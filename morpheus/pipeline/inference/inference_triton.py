from abc import abstractmethod
import asyncio
import base64
import collections
import dataclasses
import queue
import threading
import typing
from functools import partial
import warnings

import cupy as cp
import numpy as np
import tritonclient.grpc as tritonclient
from morpheus.config import Config, PipelineModes
from morpheus.pipeline.inference.inference_stage import InferenceStage
from morpheus.pipeline.messages import (MultiInferenceMessage, MultiResponseMessage, ResponseMemory)
from tornado.ioloop import IOLoop
from tqdm import tqdm
from tritonclient.utils import InferenceServerException, triton_to_np_dtype


@dataclasses.dataclass()
class TritonInOut:
    name: str  # Name of the input/output in the model
    bytes: int  # Total bytes
    datatype: str  # Triton string for datatype
    shape: typing.List[int]
    mapped_name: str  # Name of the input/output in the pipeline
    offset: int = 0
    ptr: cp.cuda.MemoryPointer = None


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

    def __init__(self, client: tritonclient.InferenceServerClient, model_name: str, config: typing.Dict[str, TritonInOut]):

        self._config = config.copy()

        self._total_bytes = 0

        for key in self._config.keys():
            self._config[key].offset = self._total_bytes
            self._total_bytes += self._config[key].bytes

        self.region_name = model_name + "_{}".format(ShmWrapper.total_count)
        ShmWrapper.total_count += 1

        # Allocate the total memory
        self._memory: cp.cuda.Memory = cp.cuda.alloc(self._total_bytes).mem

        # Get memory pointers for each object
        for key in self._config.keys():
            self._config[key].ptr = cp.cuda.MemoryPointer(self._memory, self._config[key].offset)

        # Now get the registered IPC handle
        self._ipc_handle = cp.cuda.runtime.ipcGetMemHandle(self._memory.ptr)

        # Finally, regester this memory with the server. Must be base64 for some reason???
        client.register_cuda_shared_memory(self.region_name, base64.b64encode(self._ipc_handle), 0, self._total_bytes)

    def get_bytes(self, name: str):

        return self._config[name].bytes

    def get_offset(self, name: str):

        return self._config[name].offset

    def build_input(self, name: str, data: cp.ndarray) -> tritonclient.InferInput:
        # Create the input
        triton_input = tritonclient.InferInput(name, list(data.shape), self._config[name].datatype)

        # Set the data
        self[name].copy_from_device(data.data, data.nbytes)

        # Configure the shared memory
        triton_input.set_shared_memory(self.region_name, data.nbytes, self.get_offset(name))

        return triton_input

    def __getitem__(self, name: str) -> cp.cuda.MemoryPointer:
        return self._config[name].ptr


# This class is exclusively run in the worker thread. Separating the classes helps keeps the threads separate
class TritonInference:
    def __init__(self, c: Config, model_name: str, server_url: str, inout_mapping: typing.Dict[str, str]):
        self._model_name = model_name
        self._server_url = server_url
        self._inout_mapping = inout_mapping

        self._requires_seg_ids = False

        self._max_batch_size = c.model_max_batch_size
        self._fea_length = c.feature_length

        # Whether or not the returned value needs a logits calc for the response
        self._needs_logits = c.mode == PipelineModes.NLP

        self._inputs: typing.Dict[str, TritonInOut] = {}
        self._outputs: typing.Dict[str, TritonInOut] = {}

    def init(self, loop: IOLoop):

        self._loop = loop

        self.triton_client = tritonclient.InferenceServerClient(url=self._server_url, verbose=False)

        try:
            assert self.triton_client.is_server_live() and self.triton_client.is_server_ready(), "Server is not in ready state"

            assert self.triton_client.is_model_ready(self._model_name), f"Triton model {self._model_name} is not ready"

            # To make sure no shared memory regions are registered with the server.
            self.triton_client.unregister_system_shared_memory()
            self.triton_client.unregister_cuda_shared_memory()

            model_meta = self.triton_client.get_model_metadata(self._model_name, as_json=True)
            model_config = self.triton_client.get_model_config(self._model_name, as_json=True)["config"]

            # Make sure the inputs/outputs match our config
            if (int(model_meta["inputs"][0]["shape"][-1]) != self._fea_length):
                raise RuntimeError("Mismatched Sequence Length. Config specified {} but model specified {}".format(
                    self._fea_length, int(model_meta["inputs"][0]["shape"][-1])))

            # Check batch size
            if (model_config.get("max_batch_size", 0) != self._max_batch_size):

                # If the model is more, thats fine. Gen warning
                if (model_config["max_batch_size"] > self._max_batch_size):
                    warnings.warn(
                        "Model max batch size ({}) is more than configured max batch size ({}). May result in sub optimal performance"
                        .format(model_config["max_batch_size"], self._max_batch_size))

                # If the model is less, raise error. Cant send more to Triton than the max batch size
                if (model_config["max_batch_size"] < self._max_batch_size):
                    raise RuntimeError(
                        "Model max batch size ({}) is less than configured max batch size ({}). Reduce max batch size to be less than or equal to model max batch size."
                        .format(model_config["max_batch_size"], self._max_batch_size))

            shm_config = {}

            def build_inout(x: dict):
                b = np.dtype(triton_to_np_dtype(x["datatype"])).itemsize

                shape = []

                for y in x["shape"]:
                    y_int = int(y)

                    if (y_int == -1):
                        y_int = self._max_batch_size

                    shape.append(y_int)

                    b *= y_int

                mapped_name = x["name"] if x["name"] not in self._inout_mapping else self._inout_mapping[x["name"]]

                return TritonInOut(name=x["name"], bytes=b, datatype=x["datatype"], shape=shape, mapped_name=mapped_name)

            for x in model_meta["inputs"]:

                self._inputs[x["name"]] = build_inout(x)

            for x in model_meta["outputs"]:

                assert x["name"] not in self._inputs, "Input/Output names must be unique from eachother"

                self._outputs[x["name"]] = build_inout(x)

            # Combine the inputs/outputs for the shared memory
            shm_config = {**self._inputs, **self._outputs}

            def create_wrapper():
                return ShmWrapper(self.triton_client, self._model_name, shm_config)

            self._mem_pool = ResourcePool(create_fn=create_wrapper, max_size=1000)

        except InferenceServerException as ex:
            tqdm.write("Exception occurred while coordinating with Triton. Exception message: \n{}\n".format(ex))
            raise ex

    @abstractmethod
    def _build_response(self, result: tritonclient.InferResult) -> ResponseMemory:
        pass

    def _infer_callback(self,
                        f: asyncio.Future,
                        m: ShmWrapper,
                        result: tritonclient.InferResult,
                        error: tritonclient.InferenceServerException):

        # If its an error, return that here
        if (error is not None):
            self._loop.add_callback(f.set_exception, error)
            return

        # Build response
        response_mem = self._build_response(result)

        def tmp(mem: ResponseMemory):
            # Set result on future
            f.set_result(mem)

            # Return mempool obj
            self._mem_pool.return_obj(m)

        # We have to schedule a callback here to set the future result on the asyncio thread
        self._loop.add_callback(tmp, response_mem)

    def process(self, batch: MultiInferenceMessage, fut: asyncio.Future):

        mem: ShmWrapper = self._mem_pool.borrow()

        inputs: typing.List[tritonclient.InferInput] = [
            mem.build_input(input.name, batch.get_input(input.mapped_name)) for input in self._inputs.values()
        ]

        outputs = [tritonclient.InferRequestedOutput(output.name) for output in self._outputs.values()]

        # Inference call
        self.triton_client.async_infer(model_name=self._model_name,
                                       inputs=inputs,
                                       callback=partial(self._infer_callback, fut, mem),
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


class TritonInferenceNLP(TritonInference):
    def __init__(self, c: Config, model_name: str, server_url: str, inout_mapping: typing.Dict[str, str] = {}):

        # Some models use different names for the same thing. Set that here but allow user customization
        default_mapping = {
            "attention_mask": "input_mask",
        }

        default_mapping.update(inout_mapping)

        super().__init__(c, model_name, server_url, default_mapping)

    def _build_response(self, result: tritonclient.InferResult) -> ResponseMemory:

        output = {output.mapped_name: result.as_numpy(output.name) for output in self._outputs.values()}

        if (self._needs_logits):
            output = {key: 1.0 / (1.0 + np.exp(-val)) for key, val in output.items()}

        mem = ResponseMemory(
            count=output[list(output.keys())[0]].shape[0],
            probs=cp.array(output[list(output.keys())[0]]),  # For now, only support one output
        )

        return mem


class TritonInferenceFIL(TritonInference):
    def __init__(self, c: Config, model_name: str, server_url: str, inout_mapping: typing.Dict[str, str] = {}):
        super().__init__(c, model_name, server_url, inout_mapping)

    def _build_response(self, result: tritonclient.InferResult) -> ResponseMemory:

        output = {output.mapped_name: result.as_numpy(output.name) for output in self._outputs.values()}

        mem = ResponseMemory(
            count=output[list(output.keys())[0]].shape[0],
            probs=cp.array(output[list(output.keys())[0]]),  # For now, only support one output
        )

        return mem


class TritonInferenceStage(InferenceStage):
    def __init__(self, c: Config, model_name: str, server_url: str):
        super().__init__(c)

        self._model_name = model_name
        self._server_url = server_url

        self._requires_seg_ids = False

    def _get_inference_fn(self) -> typing.Callable:

        if (Config.get().mode == PipelineModes.NLP):
            worker = TritonInferenceNLP(Config.get(), model_name=self._model_name, server_url=self._server_url)
        elif (Config.get().mode == PipelineModes.FIL):
            worker = TritonInferenceFIL(Config.get(), model_name=self._model_name, server_url=self._server_url)
        else:
            raise NotImplementedError("Unknown config mode")

        return worker.main_loop
