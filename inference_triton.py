import asyncio
import base64
import threading

from tornado.ioloop import IOLoop
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
import tritonclient.utils.cuda_shared_memory as cudashm
from tritonclient.utils import triton_to_np_dtype


class RecursiveQueue(queue.Queue):
    def __init__(self, maxsize=0):
        super().__init__(maxsize=maxsize)

        # Override the mutex and conditions with a recursive lock
        self.mutex = threading.RLock()

        self.not_empty = threading.Condition(self.mutex)
        self.not_full = threading.Condition(self.mutex)
        self.all_tasks_done = threading.Condition(self.mutex)


class ResourcePool:
    def __init__(self, create_fn: typing.Callable[[], typing.Any], max_size: int = 1000):
        self._create_fn = create_fn
        self._max_size = max_size
        self._added_count = 0

        self._queue = RecursiveQueue()

        self._adding_condition = threading.Condition(self._queue.mutex)

        self._outstanding = []

    def _borrow(self):
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            # Now try and create one
            with self._queue.mutex:

                # Only add it if we have room
                if (self._added_count < self._max_size):
                    self._queue.put(self._create_fn())
                    self._added_count += 1

            return self._queue.get()

    def borrow(self):
        obj = self._borrow()

        return obj

    def return_obj(self, obj):
        self._queue.put(obj)


class ShmWrapper:
    total_count = 0

    def __init__(self, client: tritonclient.InferenceServerClient, model_name: str, config: dict):

        self._config = config

        self._total_bytes = 0

        for key in self._config.keys():
            self._config[key]["offset"] = self._total_bytes
            self._total_bytes += self._config[key]["bytes"]

        self.region_name = model_name + "_{}".format(ShmWrapper.total_count)
        ShmWrapper.total_count += 1

        # Allocate the total memory
        self._memory: cp.cuda.Memory = cp.cuda.alloc(self._total_bytes).mem

        # Get memory pointers for each object
        for key in self._config.keys():
            self._config[key]["ptr"] = cp.cuda.MemoryPointer(self._memory, self._config[key]["offset"])

        # Now get the registered IPC handle
        self._ipc_handle = cp.cuda.runtime.ipcGetMemHandle(self._memory.ptr)

        # Finally, regester this memory with the server. Must be base64 for some reason???
        client.register_cuda_shared_memory(self.region_name, base64.b64encode(self._ipc_handle), 0, self._total_bytes)

    def get_bytes(self, name: str):

        return self._config[name]["bytes"]

    def get_offset(self, name: str):

        return self._config[name]["offset"]

    def build_input(self, name: str, data: cp.ndarray) -> tritonclient.InferInput:
        # Create the input
        triton_input = tritonclient.InferInput(name, list(data.shape), self._config[name]["type"])

        # Set the data
        self[name].copy_from_device(data.data, data.nbytes)

        # Configure the shared memory
        triton_input.set_shared_memory(self.region_name, data.nbytes, self.get_offset(name))

        return triton_input

    def __getitem__(self, name: str) -> cp.cuda.MemoryPointer:
        return self._config[name]["ptr"]


def inference_worker(loop: IOLoop, inf_queue: queue.Queue, ready_event: asyncio.Event = None):

    triton_client = tritonclient.InferenceServerClient(url="localhost:8001", verbose=False)

    mem_pool = ResourcePool(lambda: 0, max_size=1000)

    while True:

        # Get the next work item
        message: typing.Tuple[MultiInferenceMessage, asyncio.Future] = inf_queue.get(block=True)

        batch = message[0]
        fut = message[1]

        # Acquire a shared memory wrapper
        mem = mem_pool.borrow()

        def infer_callback(f, result, error):
            def tmp(r, e):
                try:
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
                finally:
                    mem_pool.return_obj(mem)

            loop.add_callback(tmp, result, error)

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


# This class is exclusively run in the worker thread. Separating the classes helps keeps the threads separate
class TritonInference:
    def __init__(self, c: Config, model_name: str, server_url: str):
        self._model_name = model_name
        self._server_url = server_url

        self._requires_seg_ids = False

        self._max_batch_size = c.model_max_batch_size
        self._seq_length = c.model_seq_length

    def init(self, loop: IOLoop):

        self._loop = loop

        self.triton_client = tritonclient.InferenceServerClient(url=self._server_url, verbose=False)

        # To make sure no shared memory regions are registered with the server.
        self.triton_client.unregister_system_shared_memory()
        self.triton_client.unregister_cuda_shared_memory()

        model_config = self.triton_client.get_model_metadata(self._model_name, as_json=True)

        shm_config = {}

        for x in model_config["inputs"] + model_config["outputs"]:

            b = np.dtype(triton_to_np_dtype(x["datatype"])).itemsize

            for y in x["shape"]:
                y_int = int(y)

                if (y_int == -1):
                    y_int = self._max_batch_size

                b *= y_int

            shm_config[x["name"]] = {
                "bytes": b,
                "type": x["datatype"],
            }

        def create_wrapper():
            return ShmWrapper(self.triton_client, self._model_name, shm_config)

        self._mem_pool = ResourcePool(create_fn=create_wrapper, max_size=1000)

    def process(self, batch: MultiInferenceMessage, fut: asyncio.Future):

        mem: ShmWrapper = self._mem_pool.borrow()

        def infer_callback(f, m, result, error):
            def tmp(r, e):
                if (e):
                    f.set_exception(e)
                else:
                    logits = r.as_numpy("output")

                    probs = 1.0 / (1.0 + np.exp(-logits))

                    f.set_result(ResponseMemory(
                        count=probs.shape[0],
                        probs=cp.array(probs),
                    ))
                self._mem_pool.return_obj(m)

            # if (error):
            #     loop.call_soon_threadsafe(fut.set_exception, error)
            # else:
            #     # progress.update(n=batch.count)
            #     loop.call_soon_threadsafe(fut.set_result, result)
            self._loop.add_callback(tmp, result, error)

        inputs: typing.List[tritonclient.InferInput] = [
            # tritonclient.InferInput("input_ids", list(batch.input_ids.shape), "INT32"),
            # tritonclient.InferInput("attention_mask", list(batch.input_mask.shape), "INT32"),
        ]

        if (self._requires_seg_ids):
            inputs.append(tritonclient.InferInput("token_type_ids", list(batch.input_ids.shape), "INT32"))

        # input_ids_np = batch.input_ids.astype(np.int32).get()

        # inputs[0].set_data_from_numpy(input_ids_np)
        # inputs[1].set_data_from_numpy(np.zeros_like(input_ids_np))

        # if (self._requires_seg_ids):
        #     inputs[2].set_data_from_numpy(batch.input_mask.astype(np.int32).get())

        outputs = [
            tritonclient.InferRequestedOutput("output"),
        ]

        # # Copy the incoming memory to shared
        # mem["input_ids"].copy_from_device(batch.input_ids.data, batch.input_ids.nbytes)
        # mem["attention_mask"].copy_from_device(batch.input_mask.data, batch.input_mask.nbytes)

        # inputs[0].set_shared_memory(mem.region_name, mem.get_bytes("input_ids"), mem.get_offset("input_ids"))
        # inputs[1].set_shared_memory(mem.region_name, mem.get_bytes("attention_mask"), mem.get_offset("attention_mask"))

        # outputs[0].set_shared_memory(mem.region_name, mem.get_bytes("output"), mem.get_offset("output"))

        inputs.append(mem.build_input("input_ids", batch.input_ids))
        inputs.append(mem.build_input("attention_mask", batch.input_mask))

        # Inference call
        self.triton_client.async_infer(model_name=self._model_name,
                                       inputs=inputs,
                                       callback=partial(infer_callback, fut, mem),
                                       outputs=outputs)

    def main_loop(self, loop: IOLoop, inf_queue: queue.Queue, ready_event: asyncio.Event = None):

        self.init(loop)

        if (ready_event is not None):
            loop.asyncio_loop.call_soon_threadsafe(ready_event.set)

        while True:

            # Get the next work item
            message: typing.Tuple[MultiInferenceMessage, asyncio.Future] = inf_queue.get(block=True)

            batch = message[0]
            fut = message[1]

            self.process(batch, fut)


class TritonInferenceStage(InferenceStage):
    def __init__(self, c: Config, model_name: str, server_url: str):
        super().__init__(c)

        self._model_name = model_name
        self._server_url=server_url

        self._requires_seg_ids = False

        # self._worker = TritonInference(c, model_name=model_name, server_url=server_url)

    def _get_inference_fn(self) -> typing.Callable:

        worker = TritonInference(Config.get(), model_name=self._model_name, server_url=self._server_url)

        return worker.main_loop
