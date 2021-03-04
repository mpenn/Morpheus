'''
To stream from Kafka, please refer to the streamz API here: https://streamz.readthedocs.io/en/latest/api.html#sources 
You can refer to https://kafka.apache.org/quickstart to start a local Kafka cluster.
Below is an example snippet to start a stream from Kafka.
'''
import asyncio
from functools import reduce
import cudf
from distributed.client import wait
from streamz import Stream, Source
from streamz.dataframe import DataFrame
import time
import json
from cudf_subword_helper import tokenize_text_series, Feature
import tensorrt as trt
import numpy as np
import cupy as cp
import queue
import threading
# import pycuda.autoinit
# import pycuda.driver as cuda
import docker
from ctypes import c_void_p
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import dataclasses
import typing

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

@dataclasses.dataclass
class Request:
    input_ids: cp.ndarray
    input_mask: cp.ndarray
    segment_ids: cp.ndarray
    input_str: typing.List[str]

    def from_feature(in_feature: Feature, in_df: cudf.Series):
        return Request(
            input_ids=in_feature.input_ids,
            input_mask=in_feature.input_mask,
            segment_ids=in_feature.segment_ids,
            input_str=in_df.to_arrow().to_pylist()
        )

# from tornado.ioloop import IOLoop

# from tornado.platform.asyncio import AsyncIOMainLoop
# AsyncIOMainLoop().install()

# source = Stream.from_textfile('pcap_dump_small.json')  # doctest: +SKIP
# source.map(json.loads).sink(print)  # doctest: +SKIP
# source.start()  # doctest: +SKIP

kafka_compose_name = "kafka-docker"

docker_client = docker.from_env()
bridge_net = docker_client.networks.get("bridge")
bridge_ip = bridge_net.attrs["IPAM"]["Config"][0]["Gateway"]

kafka_net = docker_client.networks.get(kafka_compose_name + "_default")

# Kafka brokers
bootstrap_servers = '172.17.0.1:49156,172.17.0.1:49157'

# Automatically determine the ports. Comment this if manual setup is desired
bootstrap_servers = ",".join([c.ports["9092/tcp"][0]["HostIp"] + ":" + c.ports["9092/tcp"][0]["HostPort"] for c in kafka_net.containers if "9092/tcp" in c.ports])

# Use this version to specify the bridge IP instead
bootstrap_servers = ",".join([bridge_ip + ":" + c.ports["9092/tcp"][0]["HostPort"] for c in kafka_net.containers if "9092/tcp" in c.ports])

# Kafka topic to read streaming data from
topic = "test_pcap"

# Kafka consumer configuration
consumer_conf = {'bootstrap.servers': bootstrap_servers,
                 'group.id': 'custreamz',
                 'session.timeout.ms': "60000"}

'''
If you changed Dask=True, please ensure you have a Dask cluster up and running. Refer to this for Dask API: https://docs.dask.org/en/latest/
If the input data is high throughput, we recommend using Dask, and starting appropriate number of Dask workers.
In case of high-throughput processing jobs, please set appropriate number of Kafka topic partitions to ensure parallelism.
To leverage the custreamz' accelerated Kafka reader (note that data in Kafka must be JSON format), use engine="cudf".
'''
source: Stream = Stream.from_kafka_batched(topic, consumer_conf, npartitions=None, start=True,
                                   asynchronous=False, dask=False, engine="cudf", max_batch_size=8)

out_source = Stream()

def filter_empty_data(messages):
    # Filter out null data columns and only return data
    messages = messages[~messages.data.isnull()]
    return messages

def to_cpu_dicts(messages):
    return messages.to_pandas().to_dict('records')

def process_batch(messages: cudf.DataFrame):

    data_series = messages["data"]

    tokenized = tokenize_text_series(data_series,128, 1, 'vocab_hash.txt')

    return Request.from_feature(tokenized, data_series)

def process_batch_pytorch(messages: cudf.DataFrame):
    """
    converts cudf.Series of strings to two torch tensors and meta_data- token ids and attention mask with padding
    """
    data_series = messages["data"]

    max_seq_len = 512
    num_strings = len(data_series)
    token_ids, mask, meta_data = data_series.str.subword_tokenize(
        "bert-base-cased-hash.txt",
        max_length=max_seq_len,
        stride=500,
        do_lower=False,
        do_truncate=True,
    )

    token_ids = token_ids.reshape(-1, max_seq_len)
    mask = mask.reshape(-1, max_seq_len)
    meta_data = meta_data.reshape(-1, 3)
    
    return Request(input_ids=token_ids,
                   input_mask=mask,
                   segment_ids=meta_data,
                   input_str=data_series.to_arrow().to_pylist())

    # return input_ids.type(torch.long), attention_mask.type(torch.long), meta_data.reshape(-1,3)

inf_queue = queue.Queue()

max_seq_length = 128

# Create the thread for running inference
def inference_worker():

    # pyc_dev = pycuda.autoinit.device
    # pyc_ctx = pyc_dev.retain_primary_context()
    # pyc_ctx.push()

    with open("/home/mdemoret/Repos/github/NVIDIA/TensorRT/demo/BERT/engines/bert_large_128-b8.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime, runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:

        # stream = cuda.Stream()
        stream = cp.cuda.Stream(non_blocking=True)

        input_shape_max = (max_seq_length, 8)
        input_nbytes_max = trt.volume(input_shape_max) * trt.int32.itemsize  

        # Allocate device memory for inputs.
        # d_inputs = [cuda.mem_alloc(input_nbytes_max) for _ in range(3)]
        d_inputs = [cp.cuda.alloc(input_nbytes_max) for _ in range(3)]

        # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
        # h_output = cuda.pagelocked_empty(tuple(context.get_binding_shape(3)), dtype=np.float32)
        # d_output = cuda.mem_alloc(h_output.nbytes)
        h_output = cp.cuda.alloc_pinned_memory(trt.volume(tuple(context.get_binding_shape(3))) * trt.float32.itemsize)
        d_output = cp.cuda.alloc(h_output.mem.size)

        # Stores the best profile for the batch size
        bs_to_profile = {}
        previous_bs = 0

        while True:

            # Get the next work item
            batch: Request = inf_queue.get(block=True)

            batch_size = batch.input_ids.shape[0]
            binding_idx_offset = 0

            # Specify input shapes. These must be within the min/max bounds of the active profile
            # Note that input shapes can be specified on a per-inference basis, but in this case, we only have a single shape.
            input_shape = (max_seq_length, batch_size)
            input_nbytes = trt.volume(input_shape) * trt.int32.itemsize

            # Now set the profile specific settings if the batch size changed
            if (engine.num_optimization_profiles > 1 and batch_size != previous_bs):
                if (batch_size not in bs_to_profile):
                    # select engine profile
                    selected_profile = -1
                    num_binding_per_profile = engine.num_bindings // engine.num_optimization_profiles
                    for idx in range(engine.num_optimization_profiles):
                        profile_shape = engine.get_profile_shape(profile_index = idx, binding = idx * num_binding_per_profile)
                        if profile_shape[0][1] <= batch_size and profile_shape[2][1] >= batch_size and profile_shape[0][0] <= max_seq_length and profile_shape[2][0] >= max_seq_length:
                            selected_profile = idx
                            break
                    if selected_profile == -1:
                        raise RuntimeError("Could not find any profile that can run batch size {}.".format(batch_size))

                    bs_to_profile[batch_size] = selected_profile

                # Set the actual profile
                context.active_optimization_profile = bs_to_profile[batch_size]
                binding_idx_offset = bs_to_profile[batch_size] * num_binding_per_profile


                for binding in range(3):
                    context.set_binding_shape(binding_idx_offset + binding, input_shape)
                assert context.all_binding_shapes_specified

                previous_bs = batch_size

            # ### Go via numpy array
            # input_ids = cuda.register_host_memory(np.ascontiguousarray(batch.input_ids.ravel().get()))
            # segment_ids = cuda.register_host_memory(np.ascontiguousarray(batch.segment_ids.ravel().get()))
            # input_mask = cuda.register_host_memory(np.ascontiguousarray(batch.input_mask.ravel().get()))

            # cuda.memcpy_htod_async(d_inputs[0], input_ids, stream)
            # cuda.memcpy_htod_async(d_inputs[1], segment_ids, stream)
            # cuda.memcpy_htod_async(d_inputs[2], input_mask, stream)


            # cuda.memcpy_dtod_async(d_inputs[0], int(batch.input_ids.data.ptr), batch.input_ids.nbytes, stream)
            # cuda.memcpy_dtod_async(d_inputs[1], int(batch.segment_ids.data.ptr), batch.segment_ids.nbytes, stream)
            # cuda.memcpy_dtod_async(d_inputs[2], int(batch.input_mask.data.ptr), batch.input_mask.nbytes, stream)

            d_inputs[0].copy_from_device_async(batch.input_ids.data, batch.input_ids.data.mem.size, stream=stream)
            d_inputs[1].copy_from_device_async(batch.segment_ids.data, batch.segment_ids.data.mem.size, stream=stream)
            d_inputs[2].copy_from_device_async(batch.input_mask.data, batch.input_mask.data.mem.size, stream=stream)

            bindings = [0] * binding_idx_offset + [int(d_inp) for d_inp in d_inputs] + [int(d_output)]

            # context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.ptr)

            # h_output_pinmem = cp.cuda.alloc_pinned_memory(d_output.nbytes)

            # h_output = np.frombuffer(h_output_pinmem, np.float32, d_output.size).reshape(d_output.shape)

            # d_output.set(h_output, stream=stream)

            # cuda.memcpy_dtoh_async(h_output, d_output, stream)
            d_output.copy_to_host_async(c_void_p(h_output.ptr), h_output.mem.size, stream=stream)

            stream.synchronize()

            inf_queue.task_done()

def inference_worker_pytorch():

    # Get model from https://drive.google.com/u/1/uc?id=1Lbj1IyHEBV9LS2Jo1z4cmlBNxtZUkYKa&export=download
    model = torch.load("placeholder_pii2").to('cuda')

    while True:

        # Get the next work item
        batch: Request = inf_queue.get(block=True)

        # convert from cupy to torch tensor using dlpack
        input_ids = from_dlpack(
            batch.input_ids.astype(cp.float).toDlpack()
        ).type(torch.long)
        attention_mask = from_dlpack(
            batch.input_mask.astype(cp.float).toDlpack()
        ).type(torch.long)

        with torch.no_grad():
            logits = model(input_ids, token_type_ids=None, attention_mask=attention_mask)[0]
            probs = torch.sigmoid(logits[:, 1])
            preds = probs.ge(0.5)

        if (preds.any().item()):
            for i, val in enumerate(preds.tolist()):
                if (val):
                    print("\033[0;31mFound PII:\033[0m {}".format(batch.input_str[i].replace("\r\\n", "\n")))

def queue_batch(messages: Feature):

    inf_queue.put(messages)

# Preprocess each batch
stream_df = source.map(filter_empty_data) \
                .filter(lambda x: len(x) > 0) \
                .map(process_batch_pytorch)

# Queue the inference work
stream_df.sink(queue_batch)

# turn-on the worker thread
threading.Thread(target=inference_worker_pytorch, daemon=True).start()

# IOLoop.current().start()
def run_asyncio_loop():
    loop = asyncio.get_event_loop()
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        print("Exited")

run_asyncio_loop()
