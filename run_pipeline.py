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

total_log_count = 43862


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
