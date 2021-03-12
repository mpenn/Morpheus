import asyncio
import typing
from tqdm import tqdm
from request import MultiRequest, MultiResponse, ResponseData
import queue
import torch
from torch.utils.dlpack import from_dlpack
import cupy as cp
import tensorrt as trt
from config import Config
from ctypes import c_void_p

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

config = Config.get()


def inference_worker(loop: asyncio.BaseEventLoop, inf_queue: queue.Queue):

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


    with open("triton_models/bert_trt/1/triton_bert_large_128-b8-b9.engine", "rb") as f, \
            trt.Runtime(TRT_LOGGER) as runtime, \
            runtime.deserialize_cuda_engine(f.read()) as engine, \
            engine.create_execution_context() as context:

        # stream = cuda.Stream()
        stream = cp.cuda.Stream(non_blocking=True)

        input_nbytes_max = 0
        output_nbytes_max = 0
        max_profile = 0
        num_binding_per_profile = engine.num_bindings // engine.num_optimization_profiles

        # Get the max size from the engine
        for idx in range(engine.num_optimization_profiles):
            profile_in_shape = engine.get_profile_shape(profile_index=idx, binding=idx * num_binding_per_profile)

            # Activate the profile
            context.active_optimization_profile = idx

            # Ensure all inputs are set to the max value to calc the output
            for i in range(3):
                context.set_binding_shape(idx * num_binding_per_profile + i, profile_in_shape[2])

            prof_in_nbytes = trt.volume(profile_in_shape[2]) * engine.get_binding_dtype(idx * num_binding_per_profile).itemsize

            assert context.all_binding_shapes_specified

            # Calc the output
            prof_out_nbytes = trt.volume(
                context.get_binding_shape(idx * num_binding_per_profile +
                                          3)) * engine.get_binding_dtype(idx * num_binding_per_profile + 3).itemsize

            if (prof_in_nbytes > input_nbytes_max):
                input_nbytes_max = prof_in_nbytes
                output_nbytes_max = prof_out_nbytes
                max_profile = idx

        if (locked_profile >= 0):
            context.active_optimization_profile = locked_profile

            # Set the max size so we can determine the output max size
            for i in range(3):
                context.set_binding_shape(locked_profile * num_binding_per_profile + i,
                                          engine.get_profile_shape(profile_index=locked_profile, binding=i)[2])

            assert context.all_binding_shapes_specified

            input_nbytes_max = trt.volume(context.get_binding_shape(
                locked_profile * num_binding_per_profile)) * engine.get_binding_dtype(
                    locked_profile * num_binding_per_profile).itemsize
            output_nbytes_max = trt.volume(
                context.get_binding_shape(locked_profile * num_binding_per_profile +
                                          3)) * engine.get_binding_dtype(locked_profile * num_binding_per_profile + 3).itemsize

        # Allocate device memory for inputs.
        # d_inputs = [cuda.mem_alloc(input_nbytes_max) for _ in range(3)]
        d_inputs = [cp.cuda.alloc(input_nbytes_max) for _ in range(3)]

        # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
        # h_output = cuda.pagelocked_empty(tuple(context.get_binding_shape(3)), dtype=np.float32)
        # d_output = cuda.mem_alloc(h_output.nbytes)
        h_output = cp.cuda.alloc_pinned_memory(output_nbytes_max)
        d_output = cp.cuda.alloc(h_output.mem.size)

        # Stores the best profile for the batch size
        previous_bs = 0
        previous_profile = context.active_optimization_profile
        
        prev_event = cp.cuda.Event(block=True, disable_timing=True)
        stream.record(prev_event)

        while True:

            # Get the next work item
            message: typing.Tuple[MultiRequest, asyncio.Future] = inf_queue.get(block=True)

            batch = message[0]
            fut = message[1]

            def infer_callback(f):
                def tmp(r):
                    f.set_result(
                        MultiResponse(
                            data=ResponseData(count=batch.count,
                                              probs=cp.zeros((batch.count, 1), dtype=cp.int32),
                                              input_str=batch.input_str,
                                              timestamp=batch.timestamp),
                            offset=0,
                            count=batch.count,
                        ))

                loop.asyncio_loop.call_soon_threadsafe(tmp, None)

            batch_size = batch.count
            binding_idx_offset = 0

            # Specify input shapes. These must be within the min/max bounds of the active profile
            # Note that input shapes can be specified on a per-inference basis, but in this case, we only have a single shape.
            input_shape = (batch_size, config.model.seq_length)
            input_nbytes = trt.volume(input_shape) * trt.int32.itemsize

            curr_profile = choose_profile(engine, context)

            # Ensure the profile is correct
            if (curr_profile != previous_profile):

                context.set_optimization_profile_async(curr_profile, stream_handle=stream.ptr)
                binding_idx_offset = curr_profile * num_binding_per_profile

                previous_profile = curr_profile

            # Now set the batch settings
            if (batch_size != previous_bs or True):

                for binding in range(3):
                    context.set_binding_shape(binding_idx_offset + binding, input_shape)
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
            assert batch.segment_ids.nbytes <= input_nbytes_max
            assert batch.input_mask.nbytes <= input_nbytes_max

            d_inputs[0].copy_from_device_async(batch.input_ids.data, batch.input_ids.nbytes, stream=stream)
            d_inputs[1].copy_from_device_async(batch.segment_ids.data, batch.segment_ids.nbytes, stream=stream)
            d_inputs[2].copy_from_device_async(batch.input_mask.data, batch.input_mask.nbytes, stream=stream)

            bindings = [0] * binding_idx_offset + [int(d_inp) for d_inp in d_inputs] + [int(d_output)]

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
            d_output.copy_to_host_async(c_void_p(h_output.ptr), h_output.mem.size, stream=stream)

            stream.synchronize()

            fut.set_result(
                MultiResponse(
                    data=ResponseData(count=batch.count,
                                      probs=cp.zeros((batch.count, 1), dtype=cp.int32),
                                      input_str=batch.input_str,
                                      timestamp=batch.timestamp),
                    offset=0,
                    count=batch.count,
                ))
            # infer_callback(fut)

            # stream.add_callback(infer_callback, fut)

            # stream.synchronize()

            # stream.launch_host_func(infer_callback, fut)
