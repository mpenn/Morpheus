import asyncio
from config import Config
from functools import partial
from pipeline import InferenceStage
import tritonclient.grpc as tritonclient
from tritonclient.grpc.model_config_pb2 import DataType
from tqdm import tqdm
import typing
from request import MultiInferenceMessage, MultiRequest, MultiResponse, ResponseData, ResponseMemory
import queue
import numpy as np
import cupy as cp


def inference_worker(loop: asyncio.BaseEventLoop, inf_queue: queue.Queue, ready_event: asyncio.Event = None):

    triton_client = tritonclient.InferenceServerClient(url="localhost:8001", verbose=False)

    while True:

        # Get the next work item
        message: typing.Tuple[MultiInferenceMessage, asyncio.Future] = inf_queue.get(block=True)

        batch = message[0]
        fut = message[1]

        def infer_callback(f, result, error):
            def tmp(r, e):
                if (error):
                    f.set_exception(e)
                else:
                    f.set_result(
                        MultiResponse(
                            data=ResponseData(count=batch.count,
                                              probs=cp.zeros((batch.count, 10), dtype=np.int32),
                                              input_str=batch.input_str,
                                              timestamp=batch.timestamp),
                            offset=0,
                            count=batch.count,
                        ))

            # if (error):
            #     loop.call_soon_threadsafe(fut.set_exception, error)
            # else:
            #     # progress.update(n=batch.count)
            #     loop.call_soon_threadsafe(fut.set_result, result)
            loop.asyncio_loop.call_soon_threadsafe(tmp, result, error)

        inputs: typing.List[tritonclient.InferInput] = [
            tritonclient.InferInput("input_ids", list(batch.input_ids.shape), "INT32"),
            tritonclient.InferInput("segment_ids", list(batch.input_ids.shape), "INT32"),
            tritonclient.InferInput("input_mask", list(batch.input_mask.shape), "INT32"),
        ]

        input_ids_np = batch.input_ids.astype(np.int32).get()

        inputs[0].set_data_from_numpy(input_ids_np)
        inputs[1].set_data_from_numpy(np.zeros_like(input_ids_np))
        inputs[2].set_data_from_numpy(batch.input_mask.astype(np.int32).get())

        outputs = [
            tritonclient.InferRequestedOutput("cls_squad_logits"),
        ]

        # Inference call
        triton_client.async_infer(model_name="bert_trt", inputs=inputs, callback=partial(infer_callback, fut), outputs=outputs)


class TritonInferenceStage(InferenceStage):
    def __init__(self, c: Config):
        super().__init__(c)

        self._model_name = "mini_bert_trt"

        self._requires_seg_ids = False

    def _get_inference_fn(self) -> typing.Callable:

        model_name = self._model_name
        requires_seg_ids = self._requires_seg_ids

        def worker(loop: asyncio.BaseEventLoop, inf_queue: queue.Queue, ready_event: asyncio.Event = None):

            triton_client = tritonclient.InferenceServerClient(url="localhost:8001", verbose=False)

            if (ready_event is not None):
                loop.asyncio_loop.call_soon_threadsafe(ready_event.set)

            while True:

                # Get the next work item
                message: typing.Tuple[MultiInferenceMessage, asyncio.Future] = inf_queue.get(block=True)

                batch = message[0]
                fut = message[1]

                def infer_callback(f, result, error):
                    def tmp(r, e):
                        if (e):
                            f.set_exception(e)
                        else:
                            logits = r.as_numpy("output")

                            probs = 1.0 / (1.0 + np.exp(-logits))

                            f.set_result(
                                ResponseMemory(
                                    count=probs.shape[0],
                                    probs=cp.array(probs),
                                ))

                    # if (error):
                    #     loop.call_soon_threadsafe(fut.set_exception, error)
                    # else:
                    #     # progress.update(n=batch.count)
                    #     loop.call_soon_threadsafe(fut.set_result, result)
                    loop.asyncio_loop.call_soon_threadsafe(tmp, result, error)

                inputs: typing.List[tritonclient.InferInput] = [
                    tritonclient.InferInput("input_ids", list(batch.input_ids.shape), "INT32"),
                    tritonclient.InferInput("attention_mask", list(batch.input_mask.shape), "INT32"),
                ]

                if (requires_seg_ids):
                    inputs.append(tritonclient.InferInput("token_type_ids", list(batch.input_ids.shape), "INT32"))

                input_ids_np = batch.input_ids.astype(np.int32).get()

                inputs[0].set_data_from_numpy(input_ids_np)
                inputs[1].set_data_from_numpy(np.zeros_like(input_ids_np))

                if (requires_seg_ids):
                    inputs[2].set_data_from_numpy(batch.input_mask.astype(np.int32).get())

                outputs = [
                    tritonclient.InferRequestedOutput("output"),
                ]

                # Inference call
                triton_client.async_infer(model_name=model_name,
                                          inputs=inputs,
                                          callback=partial(infer_callback, fut),
                                          outputs=outputs)

        return worker
