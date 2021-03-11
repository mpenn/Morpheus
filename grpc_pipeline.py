'''
To stream from Kafka, please refer to the streamz API here: https://streamz.readthedocs.io/en/latest/api.html#sources 
You can refer to https://kafka.apache.org/quickstart to start a local Kafka cluster.
Below is an example snippet to start a stream from Kafka.
'''
import asyncio
import concurrent.futures
import dataclasses
import io
import json
import os
import queue
import sys
import threading
import time
import typing
from asyncio.events import get_event_loop
from collections import defaultdict, deque
from ctypes import c_void_p
from functools import reduce

import cudf
import cupy as cp
# import pycuda.autoinit
# import pycuda.driver as cuda
import docker
import grpc.aio
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

from config import Config
from cudf_subword_helper import Feature, tokenize_text_series
from request import MultiRequest, SingleRequest, SingleResponse

# Add generated proto output to the path. Stupid. See https://github.com/protocolbuffers/protobuf/issues/1491
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "services")))

from services import request_pb2, request_pb2_grpc

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

        Stream.__init__(self, upstream, stream_name=stream_name, ensure_io_loop=True)

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

total_log_count = 43862


def grpc_to_cupy(in_grpc: request_pb2.CudaArrayPayload):

    # return cp.zeros(tuple(in_grpc.shape), dtype=np.dtype(in_grpc.typestr))

    class CudaArrayPayloadWrapper():
        def __init__(self, grpc: request_pb2.CudaArrayPayload):
            self._grpc = grpc

        @property
        def __cuda_array_interface__(self):
            output = {
                "shape": tuple(self._grpc.shape),
                "typestr": str(self._grpc.typestr),
                "data": (int(self._grpc.data), bool(self._grpc.readonly)),
                "version": int(self._grpc.version),
                "strides": None if len(self._grpc.strides) == 0 else list(self._grpc.strides),
            }
            return output

    # tmp = {
    #     "__cuda_array_interface__": {
    #         "shape": list(in_grpc.shape),
    #         "typestr": str(in_grpc.typestr),
    #         "data": (int(in_grpc.data), bool(in_grpc.readonly)),
    #         "version": int(in_grpc.version),
    #         "strides": None if len(in_grpc.strides) == 0 else list(in_grpc.strides),
    #     }
    # }

    out_cp = cp.asarray(CudaArrayPayloadWrapper(in_grpc))

    return out_cp


def filter_empty_data(messages):
    # Filter out null data columns and only return data
    messages = messages[~messages.data.isnull()]
    return messages


def to_cpu_dicts(messages):
    return messages.to_pandas().to_dict('records')


def process_batch(messages: cudf.DataFrame) -> typing.List[SingleRequest]:

    data_series = messages["data"]

    tokenized = tokenize_text_series(data_series, config.model.seq_length, 1, config.model.vocab_hash_file)

    return SingleRequest.from_feature(tokenized, data_series)


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

    return SingleRequest.from_feature(Feature(input_ids=token_ids, input_mask=mask, segment_ids=meta_data), data_series)


def batchsingle_to_multi(single_requests: typing.List[SingleRequest]):
    return MultiRequest.from_singles(single_requests)


async def main_loop():

    inf_queue = queue.Queue()
    progress = tqdm(desc="Running Inference for PII", smoothing=0.1, dynamic_ncols=True, unit="inf", mininterval=1.0)

    def queue_batch(messages: MultiRequest):

        # queue_progress.update(messages.count)

        inf_queue.put(messages)

    def build_source():

        source: Stream = None

        use_kafka = False

        if (not use_kafka):

            def json_to_cudf(in_str: str):
                df = cudf.io.read_json(io.StringIO("".join(in_str)), engine="cudf", lines=True)
                return df

            source: Stream = Stream.from_textfile(".tmp/pcap_dump_nonull.json", asynchronous=True, loop=IOLoop.current()).rate_limit(
                1 / 10000).timed_window(0.1).filter(lambda x: len(x) > 0).map(json_to_cudf).buffer(1000)
        else:

            # Kafka consumer configuration
            consumer_conf = {
                'bootstrap.servers': config.kafka.bootstrap_servers, 'group.id': 'custreamz', 'session.timeout.ms': "60000"
            }

            source: Stream = Stream.from_kafka_batched(config.kafka.input_topic,
                                                    consumer_conf,
                                                    npartitions=None,
                                                    start=False,
                                                    asynchronous=True,
                                                    dask=False,
                                                    engine="cudf",
                                                    loop=IOLoop.current(),
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
            raise Exception("Unknown inference pipeline: '{}'".format(config.general.pipeline))

        threading.Thread(target=inference_worker, daemon=True, args=(
            source.loop,
            inf_queue,
        )).start()

    setup()

    # source.sink(update_progress)

    req_stream: Stream = source

    def to_cpu_json(message: cudf.DataFrame):
        json_str_array = cudf.io.json.to_json(message, orient="records", lines=True).split("\n")
        data_array = message["data"].to_arrow().to_pylist()

        return json_str_array, data_array

    # Preprocess each batch into stream of Request
    req_stream = req_stream.map(filter_empty_data)
    req_stream = req_stream.filter(lambda x: len(x) > 0)
    req_stream = req_stream.map(to_cpu_json)

    channel = grpc.aio.insecure_channel('localhost:50051')

    await asyncio.wait_for(channel.channel_ready(), timeout=10.0)
    print("Connected to Preprocessing Server!")

    stub = request_pb2_grpc.PipelineStub(channel)

    # Send request to grpc server
    async def preprocess_grpc(x: typing.Tuple[typing.List[str], typing.List[str]]):

        messages = x[0]
        data = x[1]

        in_req = request_pb2.StageInput(id=1, count=len(messages))
        in_req.payload["messages"].str_array.value.extend(messages)
        in_req.payload["data"].str_array.value.extend(data)

        response = await stub.QueuePreprocessing(in_req)

        out_res = MultiRequest.create(input_ids=grpc_to_cupy(response.payload["input_ids"].cupy_array),
                                      input_mask=grpc_to_cupy(response.payload["input_mask"].cupy_array),
                                      segment_ids=grpc_to_cupy(response.payload["segment_ids"].cupy_array),
                                      input_str=list(response.payload["input_str"].str_array.value))

        return out_res.to_singles()

    req_stream = req_stream.async_map(preprocess_grpc)

    # req_stream = req_stream.map(process_batch)
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
        message = {"type": "PII", "message": x.input_str, "result": x.probs[0].get().tolist()}

        return json.dumps(message)

    res_stream = res_stream.map(post_process)

    producer_conf = {'bootstrap.servers': config.kafka.bootstrap_servers}

    res_stream.to_kafka(config.kafka.output_topic, producer_conf)

    # Sleep for 1 sec to give something to await on
    await asyncio.sleep(1.0)

    source.start()


def run_pipeline():

    loop = asyncio.get_event_loop()

    from grpc_preprocessing import serve

    asyncio.ensure_future(serve())

    asyncio.ensure_future(main_loop())
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        print("Exited")
