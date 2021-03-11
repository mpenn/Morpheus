'''
To stream from Kafka, please refer to the streamz API here: https://streamz.readthedocs.io/en/latest/api.html#sources 
You can refer to https://kafka.apache.org/quickstart to start a local Kafka cluster.
Below is an example snippet to start a stream from Kafka.
'''
import asyncio
from asyncio.events import get_event_loop
import concurrent.futures
from config import Config
import dataclasses
import io
import json
import os
import queue
import sys
import threading
import time
import typing
from collections import defaultdict, deque
from ctypes import c_void_p
from functools import reduce

import cudf
import cupy as cp
# import pycuda.autoinit
# import pycuda.driver as cuda
import docker
import numpy as np
import streamz

import torch
from distributed.client import wait
from streamz import Source, Stream
from streamz.dataframe import DataFrame
from torch.utils.dlpack import from_dlpack, to_dlpack
from tornado import gen
from tornado.ioloop import IOLoop
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile

from cudf_subword_helper import Feature, tokenize_text_series
from request import MultiRequest, SingleRequest, SingleResponse

# # Redirect printing to tqdm
orig_out_err = sys.stdout, sys.stderr


# sys.stdout, sys.stderr = map(DummyTqdmFile, (sys.stdout, sys.stderr))
@Stream.register_api()
class async_map(Stream):
    """ Apply a function to every element in the stream

    Parameters
    ----------
    func: callable
    *args :
        The arguments to pass to the function.
    **kwargs:
        Keyword arguments to pass to func

    Examples
    --------
    >>> source = Stream()
    >>> source.map(lambda x: 2*x).sink(print)
    >>> for i in range(5):
    ...     source.emit(i)
    0
    2
    4
    6
    8
    """
    def __init__(self, upstream, func, *args, **kwargs):
        self.func = func
        # this is one of a few stream specific kwargs
        stream_name = kwargs.pop('stream_name', None)
        self.kwargs = kwargs
        self.args = args

        Stream.__init__(self,
                        upstream,
                        stream_name=stream_name,
                        ensure_io_loop=True)

    @gen.coroutine
    def update(self, x, who=None, metadata=None):
        try:
            r = self.func(x, *self.args, **self.kwargs)
            result = yield r
        except Exception as e:
            # logger.exception(e)
            print(e)
            raise
        else:
            return self._emit(result, metadata=metadata)


config = Config.get()

# queue_progress = tqdm(desc="Input Stream Rate", smoothing=0.1, dynamic_ncols=True, unit="inf", mininterval=1.0, file=orig_out_err[0])

total_log_count = 43862
'''
If you changed Dask=True, please ensure you have a Dask cluster up and running. Refer to this for Dask API: https://docs.dask.org/en/latest/
If the input data is high throughput, we recommend using Dask, and starting appropriate number of Dask workers.
In case of high-throughput processing jobs, please set appropriate number of Kafka topic partitions to ensure parallelism.
To leverage the custreamz' accelerated Kafka reader (note that data in Kafka must be JSON format), use engine="cudf".
'''
# source: Stream = Stream.from_kafka_batched(topic,
#                                            consumer_conf,
#                                            npartitions=None,
#                                            start=True,
#                                            asynchronous=False,
#                                            dask=False,
#                                            engine="cudf",
#                                            config.model.max_batch_size=config.model.max_batch_size)

# with open("true_pii_as_records.json", "r") as f:
#     data = json.load(f)

# # Adjust data

# with open("true_pii_as_records-fixed.json", "w") as f2:
#     for d in data:
#         json.dump(d, f2)
#         f2.write("\n")

# def json_to_cudf(in_str: str):
#     df = cudf.io.read_json(io.StringIO("".join(in_str)), engine="cudf", lines=True)
#     return df

# source: Stream = Stream.from_textfile("pcap_dump_nonull.json", start=True, asynchronous=True).rate_limit(
#     1 / 10000).timed_window(0.1).filter(lambda x: len(x) > 0).map(json_to_cudf).buffer(1000)

# def update_progress(x):
#     if (isinstance(x, str)):
#         queue_progress.update()
#     elif (isinstance(x, cudf.DataFrame)):
#         queue_progress.update(len(x.index))
#     else:
#         queue_progress.update(len(x))


def filter_empty_data(messages):
    # Filter out null data columns and only return data
    messages = messages[~messages.data.isnull()]
    return messages


def to_cpu_dicts(messages):
    return messages.to_pandas().to_dict('records')


def process_batch(messages: cudf.DataFrame):

    data_series = messages["data"]

    tokenized = tokenize_text_series(data_series,
                                     config.model.seq_length,
                                     1,
                                     config.model.vocab_hash_file)

    return SingleRequest.from_feature(tokenized, messages)


def process_batch_pytorch(messages: cudf.DataFrame):
    """
    converts cudf.Series of strings to two torch tensors and meta_data- token ids and attention mask with padding
    """
    data_series = messages["data"]

    # max_seq_len = 512
    num_strings = len(data_series)
    token_ids, mask, meta_data = data_series.str.subword_tokenize(
        config.model.vocab_hash_file,
        max_length=config.model.seq_length,
        stride=500,
        do_lower=False,
        do_truncate=True,
    )

    token_ids = token_ids.reshape(-1, config.model.seq_length)
    mask = mask.reshape(-1, config.model.seq_length)
    meta_data = meta_data.reshape(-1, 3)

    return SingleRequest.from_feature(
        Feature(input_ids=token_ids, input_mask=mask, segment_ids=meta_data),
        data_series)

    # return Request(count=token_ids.shape[0],
    #                input_ids=token_ids,
    #                input_mask=mask,
    #                segment_ids=meta_data,
    #                input_str=data_series.to_arrow().to_pylist())


# async def process_batch_async(self, messages: cudf.DataFrame):

#     return await self.loop.run_in_executor(thread_pool, process_batch)

# # Create the thread for running inference
# def inference_worker():

#     # pyc_dev = pycuda.autoinit.device
#     # pyc_ctx = pyc_dev.retain_primary_context()
#     # pyc_ctx.push()

#     progress = tqdm(desc="Running TensorRT Inference for PII", smoothing=0.1, dynamic_ncols=True, unit="inf", mininterval=1.0, file=orig_out_err[0])

#     with open("/home/mdemoret/Repos/github/NVIDIA/TensorRT/demo/BERT/engines/bert_large_128-b16.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime, runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:

#         # stream = cuda.Stream()
#         stream = cp.cuda.Stream(non_blocking=True)

#         input_shape_max = (config.model.seq_length, config.model.max_batch_size)
#         input_nbytes_max = trt.volume(input_shape_max) * trt.int32.itemsize

#         # Allocate device memory for inputs.
#         # d_inputs = [cuda.mem_alloc(input_nbytes_max) for _ in range(3)]
#         d_inputs = [cp.cuda.alloc(input_nbytes_max) for _ in range(3)]

#         # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
#         # h_output = cuda.pagelocked_empty(tuple(context.get_binding_shape(3)), dtype=np.float32)
#         # d_output = cuda.mem_alloc(h_output.nbytes)
#         h_output = cp.cuda.alloc_pinned_memory(trt.volume(tuple(context.get_binding_shape(3))) * trt.float32.itemsize)
#         d_output = cp.cuda.alloc(h_output.mem.size)

#         # Stores the best profile for the batch size
#         bs_to_profile = {}
#         previous_bs = 0

#         while True:

#             # Get the next work item
#             batch: MultiRequest = inf_queue.get(block=True)

#             batch_size = batch.input_ids.shape[0]
#             binding_idx_offset = 0

#             # Specify input shapes. These must be within the min/max bounds of the active profile
#             # Note that input shapes can be specified on a per-inference basis, but in this case, we only have a single shape.
#             input_shape = (config.model.seq_length, batch_size)
#             input_nbytes = trt.volume(input_shape) * trt.int32.itemsize

#             # Now set the profile specific settings if the batch size changed
#             if (engine.num_optimization_profiles > 1 and batch_size != previous_bs):
#                 if (batch_size not in bs_to_profile):
#                     # select engine profile
#                     selected_profile = -1
#                     num_binding_per_profile = engine.num_bindings // engine.num_optimization_profiles
#                     for idx in range(engine.num_optimization_profiles):
#                         profile_shape = engine.get_profile_shape(profile_index=idx, binding=idx * num_binding_per_profile)
#                         if profile_shape[0][1] <= batch_size and profile_shape[2][1] >= batch_size and profile_shape[0][
#                                 0] <= config.model.seq_length and profile_shape[2][0] >= config.model.seq_length:
#                             selected_profile = idx
#                             break
#                     if selected_profile == -1:
#                         raise RuntimeError("Could not find any profile that can run batch size {}.".format(batch_size))

#                     bs_to_profile[batch_size] = selected_profile

#                 # Set the actual profile
#                 context.active_optimization_profile = bs_to_profile[batch_size]
#                 binding_idx_offset = bs_to_profile[batch_size] * num_binding_per_profile

#                 for binding in range(3):
#                     context.set_binding_shape(binding_idx_offset + binding, input_shape)
#                 assert context.all_binding_shapes_specified

#                 previous_bs = batch_size

#             # ### Go via numpy array
#             # input_ids = cuda.register_host_memory(np.ascontiguousarray(batch.input_ids.ravel().get()))
#             # segment_ids = cuda.register_host_memory(np.ascontiguousarray(batch.segment_ids.ravel().get()))
#             # input_mask = cuda.register_host_memory(np.ascontiguousarray(batch.input_mask.ravel().get()))

#             # cuda.memcpy_htod_async(d_inputs[0], input_ids, stream)
#             # cuda.memcpy_htod_async(d_inputs[1], segment_ids, stream)
#             # cuda.memcpy_htod_async(d_inputs[2], input_mask, stream)

#             # cuda.memcpy_dtod_async(d_inputs[0], int(batch.input_ids.data.ptr), batch.input_ids.nbytes, stream)
#             # cuda.memcpy_dtod_async(d_inputs[1], int(batch.segment_ids.data.ptr), batch.segment_ids.nbytes, stream)
#             # cuda.memcpy_dtod_async(d_inputs[2], int(batch.input_mask.data.ptr), batch.input_mask.nbytes, stream)

#             d_inputs[0].copy_from_device_async(batch.input_ids.data, batch.input_ids.data.mem.size, stream=stream)
#             d_inputs[1].copy_from_device_async(batch.segment_ids.data, batch.segment_ids.data.mem.size, stream=stream)
#             d_inputs[2].copy_from_device_async(batch.input_mask.data, batch.input_mask.data.mem.size, stream=stream)

#             bindings = [0] * binding_idx_offset + [int(d_inp) for d_inp in d_inputs] + [int(d_output)]

#             # context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
#             context.execute_async_v2(bindings=bindings, stream_handle=stream.ptr)

#             # h_output_pinmem = cp.cuda.alloc_pinned_memory(d_output.nbytes)

#             # h_output = np.frombuffer(h_output_pinmem, np.float32, d_output.size).reshape(d_output.shape)

#             # d_output.set(h_output, stream=stream)

#             # cuda.memcpy_dtoh_async(h_output, d_output, stream)
#             d_output.copy_to_host_async(c_void_p(h_output.ptr), h_output.mem.size, stream=stream)

#             stream.synchronize()

#             inf_queue.task_done()

#             progress.update(n=batch_size)

# def inference_worker_pytorch():

#     # Get model from https://drive.google.com/u/1/uc?id=1Lbj1IyHEBV9LS2Jo1z4cmlBNxtZUkYKa&export=download
#     model = torch.load("placeholder_pii2").to('cuda')

#     progress = tqdm.tqdm(desc="Running PyTorch Inference for PII",
#                          smoothing=0.0,
#                          dynamic_ncols=True,
#                          unit="inf",
#                          mininterval=1.0,
#                          file=orig_out_err[0])

#     while True:

#         # Get the next work item
#         batch: MultiRequest = inf_queue.get(block=True)

#         # convert from cupy to torch tensor using dlpack
#         input_ids = from_dlpack(batch.input_ids.astype(cp.float).toDlpack()).type(torch.long)
#         attention_mask = from_dlpack(batch.input_mask.astype(cp.float).toDlpack()).type(torch.long)

#         with torch.no_grad():
#             logits = model(input_ids, token_type_ids=None, attention_mask=attention_mask)[0]
#             probs = torch.sigmoid(logits[:, 1])
#             preds = probs.ge(0.5)

#         progress.update(n=batch.count)

#         if (preds.any().item()):
#             for i, val in enumerate(preds.tolist()):
#                 if (val):
#                     print("\033[0;31mFound PII:\033[0m {}".format(batch.input_str[i].replace("\r\\n", "\n")))

# def inference_worker_triton():

#     import tritonclient.grpc as tritonclient
#     from tritonclient.grpc.model_config_pb2 import DataType

#     triton_client = tritonclient.InferenceServerClient(url="localhost:8001", verbose=False)

#     progress = tqdm(desc="Running Triton Inference for PII", smoothing=0.0, dynamic_ncols=True, unit="inf", mininterval=1.0, total=total_log_count)

#     while True:

#         # Get the next work item
#         batch: MultiRequest = inf_queue.get(block=True)

#         def infer_callback(result, error):
#             if (error):
#                 print(error)
#             else:
#                 progress.update(n=batch.count)

#         inputs: typing.List[tritonclient.InferInput] = [
#             tritonclient.InferInput("input_ids", list(batch.input_ids.shape), "INT32"),
#             tritonclient.InferInput("segment_ids", list(batch.input_ids.shape), "INT32"),
#             tritonclient.InferInput("input_mask", list(batch.input_mask.shape), "INT32"),
#         ]

#         input_ids_np = batch.input_ids.astype(np.int32).get()

#         inputs[0].set_data_from_numpy(input_ids_np)
#         inputs[1].set_data_from_numpy(np.zeros_like(input_ids_np))
#         inputs[2].set_data_from_numpy(batch.input_mask.astype(np.int32).get())

#         outputs = [
#             tritonclient.InferRequestedOutput("cls_squad_logits"),
#         ]

#         # Inference call
#         triton_client.async_infer(model_name="bert_trt", inputs=inputs, callback=infer_callback, outputs=outputs)


def batchsingle_to_multi(single_requests: typing.List[SingleRequest]):
    return MultiRequest.from_singles(single_requests)


async def main_loop():

    inf_queue = queue.Queue()
    progress = tqdm(desc="Running Inference for PII",
                    smoothing=0.1,
                    dynamic_ncols=True,
                    unit="inf",
                    mininterval=1.0)

    def queue_batch(messages: MultiRequest):

        # queue_progress.update(messages.count)

        inf_queue.put(messages)

    def build_source():
        # def json_to_cudf(in_str: str):
        #     df = cudf.io.read_json(io.StringIO("".join(in_str)), engine="cudf", lines=True)
        #     return df

        # source: Source = Stream.from_textfile("pcap_dump_nonull.json", asynchronous=True).rate_limit(
        #     1 / 10000).timed_window(0.1).filter(lambda x: len(x) > 0).map(json_to_cudf).buffer(1000)

        # Automatically determine the ports. Comment this if manual setup is desired
        if (config.kafka.bootstrap_servers == "auto"):
            kafka_compose_name = "kafka-docker"

            docker_client = docker.from_env()
            bridge_net = docker_client.networks.get("bridge")
            bridge_ip = bridge_net.attrs["IPAM"]["Config"][0]["Gateway"]

            kafka_net = docker_client.networks.get(kafka_compose_name +
                                                   "_default")

            config.kafka.bootstrap_servers = ",".join([
                c.ports["9092/tcp"][0]["HostIp"] + ":" +
                c.ports["9092/tcp"][0]["HostPort"]
                for c in kafka_net.containers if "9092/tcp" in c.ports
            ])

            # Use this version to specify the bridge IP instead
            config.kafka.bootstrap_servers = ",".join([
                bridge_ip + ":" + c.ports["9092/tcp"][0]["HostPort"]
                for c in kafka_net.containers if "9092/tcp" in c.ports
            ])

            print("Auto determined Bootstrap Servers: {}".format(config.kafka.bootstrap_servers))

        # Kafka consumer configuration
        consumer_conf = {
            'bootstrap.servers': config.kafka.bootstrap_servers,
            'group.id': 'custreamz',
            'session.timeout.ms': "60000"
        }

        source: Stream = Stream.from_kafka_batched(
            config.kafka.input_topic,
            consumer_conf,
            npartitions=None,
            start=False,
            asynchronous=True,
            dask=False,
            engine="cudf",
            max_batch_size=config.model.max_batch_size)

        return source

    source: Source = build_source()

    def setup():
        # # Preprocess each batch
        # stream_df = source.map(filter_empty_data) \
        #                 .filter(lambda x: len(x) > 0) \
        #                 .map(process_batch) \
        #                 .partition_batch(config.model.max_batch_size, timeout=1)

        # # source.sink(lambda x: queue_progress.update())

        # # Queue the inference work
        # stream_df.sink(queue_batch)

        # turn-on the worker thread
        if (config.general.pipeline == "triton"):
            from inference_triton import inference_worker
        elif (config.general.pipeline == "pytorch"):
            from inference_pytorch import inference_worker
        elif (config.general.pipeline == "tensorrt"):
            from inference_tensorrt import inference_worker
        else:
            raise Exception("Unknown inference pipeline: '{}'".format(
                config.general.pipeline))

        threading.Thread(target=inference_worker,
                         daemon=True,
                         args=(
                             source.loop,
                             inf_queue,
                         )).start()

    setup()

    # source.sink(update_progress)

    req_stream: Stream = source

    # Preprocess each batch into stream of Request
    req_stream = req_stream.map(filter_empty_data)
    req_stream = req_stream.filter(lambda x: len(x) > 0)
    req_stream = req_stream.map(process_batch)
    req_stream = req_stream.flatten()

    # Do batching here if needed
    req_stream = req_stream.partition(config.model.max_batch_size, timeout=0.1)
    req_stream = req_stream.map(batchsingle_to_multi)

    # # Queue the inference work
    # req_stream.sink(queue_batch)
    async def future_map_batch(req: MultiRequest):
        # Get the current event loop.
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        inf_queue.put((req, fut))

        res = await fut

        return res

    # def out_sink(inf_out: typing.Tuple[MultiRequest, typing.Any]):
    #     progress.update(inf_out[0].count)

    def out_sink(message):
        progress.update(message.count)
        # print(message)

    res_stream = req_stream.async_map(future_map_batch)

    res_stream.sink(out_sink)

    res_stream = res_stream.map(lambda x: SingleResponse.from_multi(x))
    res_stream = res_stream.flatten()
    res_stream = res_stream.filter(lambda x: x.probs.any().item())

    # Post processing
    def post_process(x: SingleResponse):
        # Convert to final form
        message = {
            "type": "PII",
            "message": x.input_str,
            "result": x.probs[0].get().tolist()
        }

        return json.dumps(message)

    res_stream = res_stream.map(post_process)

    producer_conf = {'bootstrap.servers': config.kafka.bootstrap_servers}

    res_stream.to_kafka(config.kafka.output_topic, producer_conf)

    # Sleep for 1 sec to give something to await on
    await asyncio.sleep(1.0)

    source.start()


def run_pipeline():
    loop = asyncio.get_event_loop()
    asyncio.ensure_future(main_loop())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        print("Exited")


# if __name__ == '__main__':
#     cli(obj={}, auto_envvar_prefix='CLX')

#     print("Config: ")
#     print(dataclasses.asdict(Config.get()))

#     # run_asyncio_loop()
