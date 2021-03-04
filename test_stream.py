'''
To stream from Kafka, please refer to the streamz API here: https://streamz.readthedocs.io/en/latest/api.html#sources 
You can refer to https://kafka.apache.org/quickstart to start a local Kafka cluster.
Below is an example snippet to start a stream from Kafka.
'''
import asyncio
import tqdm
from functools import reduce
from collections import deque, defaultdict
import io
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
from tornado import gen

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

@dataclasses.dataclass
class Request:
    count: int
    input_ids: cp.ndarray
    input_mask: cp.ndarray
    segment_ids: cp.ndarray
    input_str: typing.List[str]

    def split(self, n: int):

        assert n < self.count

        # Chop the top off
        remain_count = self.count - n

        remain = Request(
            count=self.count - n,
            input_ids=self.input_ids[n:, :],
            input_mask=self.input_mask[n:, :],
            segment_ids=self.segment_ids[n:, :],
            input_str=self.input_str[n:],
        )

        # reduce
        self.count = n
        self.input_ids = self.input_ids[:n, :]
        self.input_mask = self.input_mask[:n, :]
        self.segment_ids = self.segment_ids[:n, :]
        self.input_str = self.input_str[:n]

        return remain

    def from_feature(in_feature: Feature, in_df: cudf.Series):
        return Request(
            count=in_feature.input_ids.shape[0],
            input_ids=in_feature.input_ids,
            input_mask=in_feature.input_mask,
            segment_ids=in_feature.segment_ids,
            input_str=in_df.to_arrow().to_pylist()
        )

@Stream.register_api()
class partition_batch(Stream):
    """ Partition stream into tuples of equal size

    Parameters
    ----------
    n: int
        Maximum partition size
    timeout: int or float, optional
        Number of seconds after which a partition will be emitted,
        even if its size is less than ``n``. If ``None`` (default),
        a partition will be emitted only when its size reaches ``n``.
    key: hashable or callable, optional
        Emit items with the same key together as a separate partition.
        If ``key`` is callable, partition will be identified by ``key(x)``,
        otherwise by ``x[key]``. Defaults to ``None``.

    Examples
    --------
    >>> source = Stream()
    >>> source.partition(3).sink(print)
    >>> for i in range(10):
    ...     source.emit(i)
    (0, 1, 2)
    (3, 4, 5)
    (6, 7, 8)

    >>> source = Stream()
    >>> source.partition(2, key=lambda x: x % 2).sink(print)
    >>> for i in range(4):
    ...     source.emit(i)
    (0, 2)
    (1, 3)

    >>> from time import sleep
    >>> source = Stream()
    >>> source.partition(5, timeout=1).sink(print)
    >>> for i in range(3):
    ...     source.emit(i)
    >>> sleep(1)
    (0, 1, 2)
    """
    _graphviz_shape = 'diamond'

    def __init__(self, upstream, n, timeout=None, key=None, **kwargs):
        self.n = n
        self._timeout = timeout
        self._key = key
        self._buffer = defaultdict(lambda: [])
        self._metadata_buffer = defaultdict(lambda: [])
        self._callbacks = {}
        kwargs["ensure_io_loop"] = True
        Stream.__init__(self, upstream, **kwargs)

    def _get_key(self, x):
        if self._key is None:
            return None
        if callable(self._key):
            return self._key(x)
        return x[self._key]

    @gen.coroutine
    def _flush(self, key):
        # Clear the current buffer
        result, self._buffer[key] = self._buffer[key], []
        metadata_result, self._metadata_buffer[key] = self._metadata_buffer[key], []

        # Break up the current results into n sized chunks
        output, remaining_buffer, metadata, remaining_metadata = self._to_batch(result, metadata_result)

        assert len(remaining_buffer) == len(remaining_metadata)

        yield self._emit(output, metadata=metadata)

        # Requeue any remaining
        for i in range(len(remaining_buffer)):
            self.update(remaining_buffer[i], metadata=remaining_metadata[i])

        self._release_refs(metadata)

            

    @gen.coroutine
    def update(self, x, who=None, metadata=None):
        self._retain_refs(metadata)
        key = self._get_key(x)
        buffer = self._buffer[key]
        metadata_buffer = self._metadata_buffer[key]
        buffer.append(x)
        metadata_buffer.append(metadata)

        if self._count(buffer) >= self.n:
            if self._timeout is not None and self.n > 1:
                self._callbacks[key].cancel()
            yield self._flush(key)
            return
        if len(buffer) == 1 and self._timeout is not None:
            self._callbacks[key] = self.loop.call_later(
                self._timeout, self._flush, key
            )

    def _count(self, buffer: typing.List[Request]):
        return reduce(lambda x, y: x + y.count, buffer, 0)

    def _to_batch(self, buffer: typing.List[Request], metadata: list):

        # Quick exit if we only have one and its too small
        if (len(buffer) == 1 and buffer[0].count <= self.n):
            return buffer[0], [], metadata[0], []
        elif (buffer[0].count == self.n):
            return buffer[0], buffer[1:], metadata[0], metadata[1:]

        # Are we too big
        if (buffer[0].count > self.n):
            # Chop the top off
            # remain_count = buffer[0].count - self.n

            # remain = Request(
            #     count=buffer[0].count - self.n,
            #     input_ids=buffer[0].input_ids[self.n:, :],
            #     input_mask=buffer[0].input_mask[self.n:, :],
            #     segment_ids=buffer[0].segment_ids[self.n:, :],
            #     input_str=buffer[0].input_str[self.n:],
            # )

            # # reduce
            # output = Request(
            #     count=self.n,
            #     input_ids=buffer[0].input_ids[:self.n, :],
            #     input_mask=buffer[0].input_mask[:self.n, :],
            #     segment_ids=buffer[0].segment_ids[:self.n, :],
            #     input_str=buffer[0].input_str[:self.n],
            # )
            output = buffer[0]
            remain = output.split(self.n)

            return output, [remain] + buffer[1:], metadata[0], metadata
        else:
            total_count = self._count(buffer)

            # We have multiple that are too small.
            final_count = min(total_count, self.n)

            output = Request(
                count=0,
                input_ids=cp.empty_like(buffer[0].input_ids, shape=(final_count, buffer[0].input_ids.shape[1]), order="C"),
                input_mask=cp.empty_like(buffer[0].input_mask, shape=(final_count, buffer[0].input_mask.shape[1]), order="C"),
                segment_ids=cp.empty_like(buffer[0].segment_ids, shape=(final_count, buffer[0].segment_ids.shape[1]), order="C"),
                input_str=[""] * final_count,
            )

            curr_idx = 0
            remain = []
            out_meta = []
            remain_meta = []

            for curr_idx, b in enumerate(buffer):
                if (output.count + b.count > self.n):
                    remain.append(b.split(self.n - output.count))
                    remain_meta.append(metadata[curr_idx])

                start = output.count
                stop = start + b.count

                output.input_ids[start:stop, :] = b.input_ids
                output.input_mask[start:stop, :] = b.input_mask
                output.segment_ids[start:stop, :] = b.segment_ids
                output.input_str[start:stop] = b.input_str
                output.count += b.count

                out_meta.extend(metadata[curr_idx])

                if (output.count == self.n):
                    break

            curr_idx += 1

            return output, remain + buffer[curr_idx:], out_meta, remain_meta + metadata[curr_idx:]


queue_progress = tqdm.tqdm(desc="Input Stream Rate", smoothing=0.1, dynamic_ncols=True, unit="inf", mininterval=1.0)


# from tornado.ioloop import IOLoop

# from tornado.platform.asyncio import AsyncIOMainLoop
# AsyncIOMainLoop().install()

# source = Stream.from_textfile('pcap_dump_small.json')  # doctest: +SKIP
# source.map(json.loads).sink(print)  # doctest: +SKIP
# source.start()  # doctest: +SKIP

max_batch_size = 16

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
# source: Stream = Stream.from_kafka_batched(topic, consumer_conf, npartitions=None, start=True,
#                                    asynchronous=False, dask=False, engine="cudf", max_batch_size=max_batch_size)

def json_to_cudf(in_str: str):
    df = cudf.io.read_json(io.StringIO("".join(in_str)), engine="cudf", lines=True)
    return df

source: Stream = Stream.from_textfile("pcap_dump.json", start=True).rate_limit(1/10000).timed_window(0.1).filter(lambda x: len(x) > 0).map(json_to_cudf).buffer(1000)

def update_progress(x):
    if (isinstance(x, str)):
        queue_progress.update()
    elif (isinstance(x, cudf.DataFrame)):
        queue_progress.update(len(x.index))
    else:
        queue_progress.update(len(x))

source.sink(update_progress)

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
    
    return Request(count=token_ids.shape[0],
                   input_ids=token_ids,
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

    progress = tqdm.tqdm(desc="Running TensorRT Inference for PII", smoothing=0.1, dynamic_ncols=True, unit="inf")

    with open("/home/mdemoret/Repos/github/NVIDIA/TensorRT/demo/BERT/engines/bert_large_128-b16.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime, runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:

        # stream = cuda.Stream()
        stream = cp.cuda.Stream(non_blocking=True)

        input_shape_max = (max_seq_length, max_batch_size)
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

            progress.update(n=batch_size)

def inference_worker_pytorch():

    # Get model from https://drive.google.com/u/1/uc?id=1Lbj1IyHEBV9LS2Jo1z4cmlBNxtZUkYKa&export=download
    model = torch.load("placeholder_pii2").to('cuda')

    progress = tqdm.tqdm(desc="Running PyTorch Inference for PII", smoothing=0.1, dynamic_ncols=True, unit="inf")

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

        progress.update(n=batch.count)

        if (preds.any().item()):
            for i, val in enumerate(preds.tolist()):
                if (val):
                    progress.write("\033[0;31mFound PII:\033[0m {}".format(batch.input_str[i].replace("\r\\n", "\n")))



def queue_batch(messages: Request):

    # queue_progress.update(messages.count)

    inf_queue.put(messages)

# Preprocess each batch
stream_df = source.map(filter_empty_data) \
                .filter(lambda x: len(x) > 0) \
                .map(process_batch) \
                .partition_batch(max_batch_size, timeout=1)

# source.sink(lambda x: queue_progress.update())

# Queue the inference work
stream_df.sink(queue_batch)

# turn-on the worker thread
threading.Thread(target=inference_worker, daemon=True).start()

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
