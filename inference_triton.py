import tritonclient.grpc as tritonclient
from tritonclient.grpc.model_config_pb2 import DataType
from tqdm import tqdm
import typing
from request import MultiRequest
import queue
import numpy as np

def inference_worker(inf_queue: queue.Queue):


    triton_client = tritonclient.InferenceServerClient(url="localhost:8001", verbose=False)

    progress = tqdm(desc="Running Triton Inference for PII", smoothing=0.0, dynamic_ncols=True, unit="inf", mininterval=1.0)

    while True:

        # Get the next work item
        batch: MultiRequest = inf_queue.get(block=True)

        def infer_callback(result, error):
            if (error):
                print(error)
            else:
                progress.update(n=batch.count)

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
        triton_client.async_infer(model_name="bert_trt", inputs=inputs, callback=infer_callback, outputs=outputs)