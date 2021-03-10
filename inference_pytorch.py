from tqdm import tqdm
from request import MultiRequest
import queue
import torch
from torch.utils.dlpack import from_dlpack
import cupy as cp

def inference_worker(inf_queue: queue.Queue):

    # Get model from https://drive.google.com/u/1/uc?id=1Lbj1IyHEBV9LS2Jo1z4cmlBNxtZUkYKa&export=download
    model = torch.load("placeholder_pii2").to('cuda')

    progress = tqdm(desc="Running PyTorch Inference for PII",
                         smoothing=0.0,
                         dynamic_ncols=True,
                         unit="inf",
                         mininterval=1.0)

    while True:

        # Get the next work item
        batch: MultiRequest = inf_queue.get(block=True)

        # convert from cupy to torch tensor using dlpack
        input_ids = from_dlpack(batch.input_ids.astype(cp.float).toDlpack()).type(torch.long)
        attention_mask = from_dlpack(batch.input_mask.astype(cp.float).toDlpack()).type(torch.long)

        with torch.no_grad():
            logits = model(input_ids, token_type_ids=None, attention_mask=attention_mask)[0]
            probs = torch.sigmoid(logits[:, 1])
            preds = probs.ge(0.5)

        progress.update(n=batch.count)

        # if (preds.any().item()):
        #     for i, val in enumerate(preds.tolist()):
        #         if (val):
        #             print("\033[0;31mFound PII:\033[0m {}".format(batch.input_str[i].replace("\r\\n", "\n")))