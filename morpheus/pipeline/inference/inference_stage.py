from abc import abstractmethod
from functools import reduce
from morpheus.pipeline.pipeline import StreamFuture, StreamPair
import queue
import time
import typing
from streamz.core import Stream
from tornado.ioloop import IOLoop
from morpheus.pipeline import Stage
from morpheus.pipeline.messages import InferenceMemory, MessageMeta, MultiInferenceMessage, MultiMessage, MultiResponseMessage, ResponseMemory
from morpheus.config import Config
from tqdm import tqdm
import cudf
import threading
import json
from morpheus.utils.cudf_subword_helper import tokenize_text_series
import cupy as cp
import asyncio
import numpy as np
import typing_utils

class InferenceStage(Stage):
    def __init__(self, c: Config):
        super().__init__(c)

        self._post_sink_fn = self.post_timestamps
        self._seq_length = c.model_seq_length

        self._thread = None
        self._inf_queue = queue.Queue()

        self._max_batch_size = c.model_max_batch_size


    @property
    def name(self) -> str:
        return "inference"

    def accepted_types(self) -> typing.Tuple:
        return (MultiInferenceMessage, StreamFuture[MultiInferenceMessage])

    @abstractmethod
    def _get_inference_fn(self) -> typing.Callable:
        pass

    async def _build(self, input_stream: StreamPair) -> StreamPair:

        wait_events = []

        for i in range(1):
            ready_event = asyncio.Event()

            threading.Thread(target=self._get_inference_fn(),
                             daemon=True,
                             args=(
                                 IOLoop.current(),
                                 self._inf_queue,
                                 ready_event,
                             )).start()

            # Wait for the inference thread to be ready
            wait_events.append(ready_event.wait())

        asyncio.gather(*wait_events, return_exceptions=True)

        stream = input_stream[0]
        out_type = MultiResponseMessage

        if (typing_utils.issubtype(input_stream[1], StreamFuture)):
            # First convert to manageable batches. If the batches are too large, we cant process them
            stream = stream.map(self._split_batches, max_batch_size=self._max_batch_size).gather()

            # Queue the inference work
            stream = stream.async_map(self._queue_inf_work)

            stream = stream.scatter().map(self._convert_response)
            out_type = StreamFuture[MultiResponseMessage]
        else:
            # First convert to manageable batches. If the batches are too large, we cant process them
            stream = stream.async_map(self._split_batches, executor=self._pipeline.thread_pool, max_batch_size=self._max_batch_size)

            # Queue the inference work
            stream = stream.async_map(self._queue_inf_work)

            stream = stream.async_map(self._convert_response, executor=self._pipeline.thread_pool)

        return stream, out_type

    @staticmethod
    def _split_batches(x: MultiInferenceMessage, max_batch_size: int) -> typing.List[MultiInferenceMessage]:

        out_batches = []

        id_array = cp.concatenate([cp.array([-1]), x.seq_ids[:, 0], cp.array([-1])])

        diff_ids = cp.where(id_array[1:] != id_array[:-1])[0]

        diff_ids = diff_ids.tolist()

        head = 0
        tail = 0

        for i in range(1, len(diff_ids)):

            poss_count = diff_ids[i] - diff_ids[head]

            if (poss_count > max_batch_size):
                out_batches.append((diff_ids[head], diff_ids[tail]))

                head = tail

            tail = i

        out_batches.append((diff_ids[head], diff_ids[tail]))

        out_resp = []

        for start, stop in out_batches:

            out_resp.append(x.get_slice(start, stop))

        assert len(out_resp) > 0

        return out_resp

    async def _queue_inf_work(self, x: typing.List[MultiInferenceMessage]):
        # Get the current event loop.
        loop = asyncio.get_running_loop()

        futures = []

        for y in x:
            fut = loop.create_future()

            self._inf_queue.put((y, fut))

            futures.append(fut)

        res = await asyncio.gather(*futures, return_exceptions=True)

        return x, res

    @staticmethod
    def _convert_response(x: typing.Tuple[typing.List[MultiInferenceMessage], typing.List[ResponseMemory]]):

        # Convert a MultiResponse into a MultiResponseMessage
        in_message = x[0]
        out_message = x[1]


        assert len(in_message) == len(out_message)

        # Get the total output size
        total_mess_count = reduce(lambda y, z: y + z.mess_count, in_message, 0)

        # Create a message data to store the entire list
        memory = ResponseMemory(count=total_mess_count, probs=cp.zeros((total_mess_count, out_message[0].probs.shape[1])))

        saved_meta = in_message[0].meta
        saved_offset = in_message[0].mess_offset
        saved_count = 0

        for inf, res in zip(in_message, out_message):

            # Ensure they all share the same meta object. Otherwise this doesnt work
            assert inf.meta == saved_meta

            # Make sure we have a continuous list
            assert inf.mess_offset == saved_offset + saved_count

            # Two scenarios:
            if (inf.mess_count == inf.count):
                # In message and out message have same count. Just use probs as is
                memory.probs[inf.mess_offset:inf.mess_offset + inf.mess_count, :] = res.probs
            else:
                assert inf.count == res.count

                mess_ids = inf.seq_ids[:, 0].get().tolist()

                # Out message has more reponses, so we have to do key based blending of probs
                for i, id in enumerate(mess_ids):
                    memory.probs[id, :] = cp.maximum(memory.probs[id, :], res.probs[i, :])

            saved_count += inf.mess_count

        # For now, we assume that this is the whole group of messages so it must start at 0
        assert saved_offset == 0

        return MultiResponseMessage(meta=saved_meta,
                                    mess_offset=saved_offset,
                                    mess_count=saved_count,
                                    memory=memory,
                                    offset=0,
                                    count=memory.count)

    def post_timestamps(self, x: MultiResponseMessage):

        curr_time = time.time()

        x.set_meta("ts_" + self.name, curr_time)