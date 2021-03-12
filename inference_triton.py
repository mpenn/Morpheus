import asyncio
from functools import partial
import tritonclient.grpc as tritonclient
from tritonclient.grpc.model_config_pb2 import DataType
from tqdm import tqdm
import typing
from request import MultiRequest, MultiResponse, ResponseData
import queue
import numpy as np
import cupy as cp


def inference_worker(loop: asyncio.BaseEventLoop, inf_queue: queue.Queue):

    triton_client = tritonclient.InferenceServerClient(url="localhost:8001", verbose=False)

    progress = tqdm(desc="Running Triton Inference for PII", smoothing=0.0, dynamic_ncols=True, unit="inf", mininterval=1.0)

    while True:

        # Get the next work item
        message: typing.Tuple[MultiRequest, asyncio.Future] = inf_queue.get(block=True)

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
                                              probs=cp.zeros((batch.count, 1), dtype=np.int32),
                                              input_str=batch.input_str),
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
