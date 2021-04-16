import asyncio
import queue
import typing
from functools import partial

import cupy as cp
import numpy as np
import tritonclient.grpc as tritonclient
from scipy.special import expit
from tqdm import tqdm
from tritonclient.grpc.model_config_pb2 import DataType

from request import MultiRequest
from request import MultiResponse
from request import ResponseData


def inference_worker(loop: asyncio.BaseEventLoop, inf_queue: queue.Queue):

    triton_client = tritonclient.InferenceServerClient(url="localhost:8001", verbose=False)

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
                                              probs=cp.asarray(expit(r.as_numpy("output"))),
                                              input_str=batch.input_str,
                                              timestamp=batch.timestamp),
                            offset=0,
                            count=batch.count,
                        ))

            loop.asyncio_loop.call_soon_threadsafe(tmp, result, error)

        inputs: typing.List[tritonclient.InferInput] = [
            tritonclient.InferInput("input_ids", list(batch.input_ids.shape), "INT32"),
            tritonclient.InferInput("attention_mask", list(batch.input_mask.shape), "INT32"),
            # tritonclient.InferInput("token_type_ids", list(batch.input_ids.shape), "INT64"),
        ]

        input_ids_np = batch.input_ids.astype(np.int32).get()

        inputs[0].set_data_from_numpy(input_ids_np)
        inputs[1].set_data_from_numpy(batch.input_mask.astype(np.int32).get())
        # inputs[2].set_data_from_numpy(np.zeros_like(input_ids_np))
        

        outputs = [
            tritonclient.InferRequestedOutput("output"),
            # tritonclient.InferRequestedOutput("output_1"),
        ]

        # Inference call
        triton_client.async_infer(model_name="mini_bert_trt", inputs=inputs, callback=partial(infer_callback, fut), outputs=outputs)
