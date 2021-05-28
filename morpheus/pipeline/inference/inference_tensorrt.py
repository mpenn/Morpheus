import asyncio
import os
import queue
import typing
from ctypes import c_void_p

import cupy as cp
import tensorrt as trt
from tornado.ioloop import IOLoop
from tqdm import tqdm

import torch
from config import Config
from pipeline import InferenceStage
from request import MultiInferenceMessage
from request import MultiRequest
from request import MultiResponse
from request import ResponseData
from request import ResponseMemoryProbs
from torch.utils.dlpack import from_dlpack

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

config = Config.get()


def onnx_to_engine(runtime, onnx_file):

    # First check for a matching engine file
    matching_engine = os.path.splitext(onnx_file)[0] + ".engine"

    if (os.path.exists(matching_engine)):
        print("Found matching engine file at '{}'. Loading the engine instead...".format(matching_engine))

        # Deserialize and return the engine
        with open(matching_engine, "rb") as f:

            return runtime.deserialize_cuda_engine(f.read())

    # Otherwise we are creating a new model
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnx_file, "rb") as model_file:
            if (not parser.parse(model_file.read())):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise Exception("Count not parse Onnx file. See log.")

        # Now we need to build and serialize the model
        with builder.create_builder_config() as builder_config:

            builder_config.max_workspace_size = 16000 * (1024 * 1024)
            builder_config.set_flag(trt.BuilderFlag.FP16)

            # Create the optimization files
            prev_batch_size = 0
            batch_sizes = [8, 16, 32]
            for batch_size in sorted(batch_sizes):
                profile = builder.create_optimization_profile()

                min_shape = (prev_batch_size + 1, config.model.seq_length)
                shape = (batch_size, config.model.seq_length)

                for i in range(network.num_inputs):
                    in_tensor = network.get_input(i)
                    profile.set_shape(in_tensor.name, min=min_shape, opt=shape, max=shape)

                builder_config.add_optimization_profile(profile)

            # Actually build the engine
            print("Building engine. This may take a while...")
            engine = builder.build_engine(network, builder_config)

            # Now save a copy to prevent building next time
            print("Writing engine to: {}".format(matching_engine))
            serialized_engine = engine.serialize()

            with open(matching_engine, "wb") as f:
                f.write(serialized_engine)

            return engine


def build_engine(runtime, model_file):

    model_ext = os.path.splitext(model_file)[1]

    if (model_ext == ".engine"):
        with open(model_file, "rb") as f:

            return runtime.deserialize_cuda_engine(f.read())
    elif (model_ext == ".onnx"):
        return onnx_to_engine(runtime, model_file)


def inference_worker(loop: IOLoop, inf_queue: queue.Queue, ready_event: asyncio.Event = None):

    # pyc_dev = pycuda.autoinit.device
    # pyc_ctx = pyc_dev.retain_primary_context()
    # pyc_ctx.push()

    # progress = tqdm(desc="Running TensorRT Inference for PII",
    #                 smoothing=0.1,
    #                 dynamic_ncols=True,
    #                 unit="inf",
    #                 mininterval=1.0)

    locked_profile = 0  # -1 to unlock
    bs_to_profile = {}

    def choose_profile(engine, context) -> int:

        if (locked_profile >= 0):
            return locked_profile

        if (batch_size not in bs_to_profile):
            # select engine profile
            selected_profile = -1

            for idx in range(engine.num_optimization_profiles):
                profile_shape = engine.get_profile_shape(profile_index=idx, binding=idx * num_binding_per_profile)
                if profile_shape[0][1] <= batch_size and profile_shape[2][1] >= batch_size and profile_shape[0][
                        0] <= config.model.seq_length and profile_shape[2][0] >= config.model.seq_length:
                    selected_profile = idx
                    break
            if selected_profile == -1:
                raise RuntimeError("Could not find any profile that can run batch size {}.".format(batch_size))

            bs_to_profile[batch_size] = selected_profile

            return selected_profile

    # model_file = "triton_models/bert_trt/1/triton_bert_large_256-b1_8-b1_16-b1_32.engine"
    # model_file = "triton_models/bert_trt/1/triton_bert_large_128-b8-b9.engine"
    # model_file = "triton_models/pii_onnx/1/model_10labels_256seqV3.onnx"
    model_file = ".tmp/models_onnx/mini_bert_128seq.onnx"

    with trt.Runtime(TRT_LOGGER) as runtime, \
         build_engine(runtime, model_file=model_file) as engine, \
         engine.create_execution_context() as context:

        # stream = cuda.Stream()
        stream = cp.cuda.Stream(non_blocking=True)

        stream.use()

        input_nbytes_max = 0
        output_nbytes_max = 0
        output_shape_max = None
        output_dtype = None
        max_profile = 0
        num_binding_per_profile = engine.num_bindings // engine.num_optimization_profiles

        num_inputs = 0

        for i in range(num_binding_per_profile):
            if (engine.binding_is_input(i)):
                num_inputs += 1

        output_binding_offset = num_inputs

        def get_binding(profile: int, binding_offset: int):
            return profile * num_binding_per_profile + binding_offset

        def get_output_binding(profile: int):
            return get_binding(profile, output_binding_offset)

        # Get the max size from the engine
        for idx in range(engine.num_optimization_profiles):
            profile_in_shape = engine.get_profile_shape(profile_index=idx, binding=get_binding(idx, 0))

            # Activate the profile
            context.active_optimization_profile = idx

            # Ensure all inputs are set to the max value to calc the output
            for i in range(num_inputs):
                context.set_binding_shape(get_binding(idx, i), profile_in_shape[2])

            prof_in_nbytes = trt.volume(profile_in_shape[2]) * engine.get_binding_dtype(get_binding(idx, 0)).itemsize

            assert context.all_binding_shapes_specified

            # Calc the output
            prof_out_nbytes = trt.volume(context.get_binding_shape(get_output_binding(idx))) * engine.get_binding_dtype(
                get_output_binding(idx)).itemsize

            if (prof_in_nbytes > input_nbytes_max):
                input_nbytes_max = prof_in_nbytes
                output_nbytes_max = prof_out_nbytes
                max_profile = idx

        if (locked_profile >= 0):
            context.active_optimization_profile = locked_profile

            # Set the max size so we can determine the output max size
            for i in range(num_inputs):
                context.set_binding_shape(
                    get_binding(locked_profile, i),
                    engine.get_profile_shape(profile_index=locked_profile, binding=get_binding(locked_profile, i))[2])

            assert context.all_binding_shapes_specified

            input_nbytes_max = trt.volume(context.get_binding_shape(get_binding(locked_profile, 0))) * engine.get_binding_dtype(
                get_binding(locked_profile, 0)).itemsize

            output_shape_max = context.get_binding_shape(get_output_binding(locked_profile))
            output_dtype = engine.get_binding_dtype(get_output_binding(locked_profile))
            output_nbytes_max = trt.volume(output_shape_max) * output_dtype.itemsize

            # Convert to usable types
            output_shape_max = tuple(output_shape_max)
            output_dtype = trt.nptype(output_dtype)

        # Allocate device memory for inputs.
        # d_inputs = [cuda.mem_alloc(input_nbytes_max) for _ in range(3)]
        d_inputs = [cp.cuda.alloc(input_nbytes_max) for _ in range(num_inputs)]

        # Set segment IDs to 0 since we never actually use this input
        d_inputs[1].memset(0, input_nbytes_max)

        # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
        # h_output = cuda.pagelocked_empty(tuple(context.get_binding_shape(3)), dtype=np.float32)
        # d_output = cuda.mem_alloc(h_output.nbytes)
        h_output = cp.cuda.alloc_pinned_memory(output_nbytes_max)
        d_output = cp.cuda.alloc(h_output.mem.size)

        d_output.memset(0, output_nbytes_max)

        # Stores the best profile for the batch size
        previous_bs = 0
        previous_profile = context.active_optimization_profile

        prev_event = cp.cuda.Event(block=True, disable_timing=True)
        stream.record(prev_event)

        if (ready_event is not None):
            loop.add_callback(ready_event.set)
            # loop.asyncio_loop.call_soon_threadsafe(ready_event.set)

        while True:

            # Get the next work item
            message: typing.Tuple[MultiInferenceMessage, asyncio.Future] = inf_queue.get(block=True)

            batch = message[0]
            fut = message[1]

            # def infer_callback(f):
            #     def tmp(r):
            #         f.set_result(
            #             MultiResponse(
            #                 data=ResponseData(count=batch.count,
            #                                   probs=cp.zeros((batch.count, 1), dtype=cp.int32),
            #                                   input_str=batch.input_str,
            #                                   timestamp=batch.timestamp),
            #                 offset=0,
            #                 count=batch.count,
            #             ))

            #     loop.asyncio_loop.call_soon_threadsafe(tmp, None)

            batch_size = batch.count

            # Specify input shapes. These must be within the min/max bounds of the active profile
            # Note that input shapes can be specified on a per-inference basis, but in this case, we only have a single shape.
            input_shape = (batch_size, config.model.seq_length)
            input_nbytes = trt.volume(input_shape) * trt.int32.itemsize

            curr_profile = choose_profile(engine, context)

            # Ensure the profile is correct
            if (curr_profile != previous_profile):

                context.set_optimization_profile_async(curr_profile, stream_handle=stream.ptr)

                previous_profile = curr_profile

            # Now set the batch settings
            if (batch_size != previous_bs or True):

                for binding in range(num_inputs):
                    context.set_binding_shape(get_binding(curr_profile, binding), input_shape)
                assert context.all_binding_shapes_specified

                previous_bs = batch_size

            # ### Go via numpy array
            # input_ids = cuda.register_host_memory(np.ascontiguousarray(batch.input_ids.ravel().get()))
            # segment_ids = cuda.register_host_memory(np.ascontiguousarray(batch.segment_ids.ravel().get()))
            # input_mask = cuda.register_host_memory(np.ascontiguousarray(batch.input_mask.ravel().get()))

            # cuda.memcpy_htod_async(d_inputs[0], input_ids, stream)
            # cuda.memcpy_htod_async(d_inputs[1], segment_ids, stream)
            # cuda.memcpy_htod_async(d_inputs[2], input_mask, stream)

            # cuda.memcpy_dtod_async(d_inputs[0], int(batch.input_ids.data.ptr), batch.input_ids.nbytes, stream)
            # cuda.memcpy_dtod_async(d_inputs[1], int(batch.segment_ids.data.ptr), batch.segment_ids.nbytes, stream)
            # cuda.memcpy_dtod_async(d_inputs[2], int(batch.input_mask.data.ptr), batch.input_mask.nbytes, stream)

            assert batch.input_ids.nbytes <= input_nbytes_max
            assert batch.input_mask.nbytes <= input_nbytes_max

            if (len(d_inputs) == 3):
                d_inputs[0].copy_from_device_async(batch.input_ids.data, batch.input_ids.nbytes, stream=stream)
                d_inputs[2].copy_from_device_async(batch.input_mask.data, batch.input_mask.nbytes, stream=stream)
            elif (len(d_inputs) == 2):
                d_inputs[0].copy_from_device_async(batch.input_ids.data, batch.input_ids.nbytes, stream=stream)
                d_inputs[1].copy_from_device_async(batch.input_mask.data, batch.input_mask.nbytes, stream=stream)
            else:
                raise Exception("Unexpected number of inputs")

            bindings = [0] * get_binding(curr_profile, 0) + [int(d_inp) for d_inp in d_inputs] + [int(d_output)]

            # context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.ptr)

            # h_output_pinmem = cp.cuda.alloc_pinned_memory(d_output.nbytes)

            # h_output = np.frombuffer(h_output_pinmem, np.float32, d_output.size).reshape(d_output.shape)

            # d_output.set(h_output, stream=stream)

            # Sync here to prevent overwriting the host and also allow add_callback to fire each iteration
            # prev_event.synchronize()
            # prev_event = cp.cuda.Event(block=True, disable_timing=True)
            # stream.record(prev_event)

            # cuda.memcpy_dtoh_async(h_output, d_output, stream)
            # d_output.copy_to_host_async(c_void_p(h_output.ptr), h_output.mem.size, stream=stream)

            logits = cp.ndarray(shape=output_shape_max, dtype=output_dtype, memptr=d_output)[:batch_size, :]

            # probs = 1.0 / (1.0 + cp.exp(-logits))
            probs = logits

            stream.synchronize()

            fut.set_result(ResponseMemoryProbs(
                count=probs.shape[0],
                probs=probs,
            ))
            # infer_callback(fut)

            # stream.add_callback(infer_callback, fut)

            # stream.synchronize()

            # stream.launch_host_func(infer_callback, fut)


class TensorRTInferenceStage(InferenceStage):
    def __init__(self, c: Config):
        super().__init__(c)

    def _get_inference_fn(self) -> typing.Callable:

        return inference_worker
