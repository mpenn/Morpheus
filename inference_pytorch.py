import asyncio
import typing
from tqdm import tqdm
from request import MultiRequest, MultiResponse, ResponseData
import queue
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
import cupy as cp


def inference_worker(loop: asyncio.BaseEventLoop, inf_queue: queue.Queue):

    # Get model from https://drive.google.com/u/1/uc?id=1Lbj1IyHEBV9LS2Jo1z4cmlBNxtZUkYKa&export=download
    model = torch.load(".tmp/ph_label_model.bin").to('cuda')

    while True:

        # Get the next work item
        message: typing.Tuple[MultiRequest,
                              asyncio.Future] = inf_queue.get(block=True)

        batch = message[0]
        fut = message[1]

        # convert from cupy to torch tensor using dlpack
        input_ids = from_dlpack(batch.input_ids.astype(
            cp.float).toDlpack()).type(torch.long)
        attention_mask = from_dlpack(
            batch.input_mask.astype(cp.float).toDlpack()).type(torch.long)

        with torch.no_grad():
            logits = model(input_ids,
                           token_type_ids=None,
                           attention_mask=attention_mask)[0]
            probs = torch.sigmoid(logits)
            preds = probs.ge(0.5)

        probs_cp = cp.fromDlpack(to_dlpack(probs))

        # Ensure that we are of the shape `[Batch Size, Num Labels]`
        if (len(probs_cp.shape) == 1):
            probs_cp = cp.expand_dims(probs_cp, axis=1)

        fut.set_result(
            MultiResponse(
                data=ResponseData(count=batch.count,
                                  probs=probs_cp,
                                  input_str=batch.input_str,
                                  timestamp=batch.timestamp),
                offset=0,
                count=batch.count,
            ))

        # if (preds.any().item()):
        #     for i, val in enumerate(preds.tolist()):
        #         if (val):
        #             print("\033[0;31mFound PII:\033[0m {}".format(batch.input_str[i].replace("\r\\n", "\n")))