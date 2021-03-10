from tqdm import tqdm
from request import MultiRequest
import queue
import torch
from torch.utils.dlpack import from_dlpack
import cupy as cp
import tensorrt as trt
from config import Config
from ctypes import c_void_p

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

config = Config.get()


def inference_worker(inf_queue: queue.Queue):

    # pyc_dev = pycuda.autoinit.device
    # pyc_ctx = pyc_dev.retain_primary_context()
    # pyc_ctx.push()

    progress = tqdm(desc="Running TensorRT Inference for PII",
                    smoothing=0.1,
                    dynamic_ncols=True,
                    unit="inf",
                    mininterval=1.0)

    with open("/home/mdemoret/Repos/github/NVIDIA/TensorRT/demo/BERT/engines/bert_large_128-b16.engine", "rb") as f, \
            trt.Runtime(TRT_LOGGER) as runtime, \
            runtime.deserialize_cuda_engine(f.read()) as engine, \
            engine.create_execution_context() as context:

        # stream = cuda.Stream()
        stream = cp.cuda.Stream(non_blocking=True)

        input_shape_max = (config.model.seq_length,
                           config.model.max_batch_size)
        input_nbytes_max = trt.volume(input_shape_max) * trt.int32.itemsize

        # Allocate device memory for inputs.
        # d_inputs = [cuda.mem_alloc(input_nbytes_max) for _ in range(3)]
        d_inputs = [cp.cuda.alloc(input_nbytes_max) for _ in range(3)]

        # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
        # h_output = cuda.pagelocked_empty(tuple(context.get_binding_shape(3)), dtype=np.float32)
        # d_output = cuda.mem_alloc(h_output.nbytes)
        h_output = cp.cuda.alloc_pinned_memory(
            trt.volume(tuple(context.get_binding_shape(3))) *
            trt.float32.itemsize)
        d_output = cp.cuda.alloc(h_output.mem.size)

        # Stores the best profile for the batch size
        bs_to_profile = {}
        previous_bs = 0

        while True:

            # Get the next work item
            batch: MultiRequest = inf_queue.get(block=True)

            batch_size = batch.input_ids.shape[0]
            binding_idx_offset = 0

            # Specify input shapes. These must be within the min/max bounds of the active profile
            # Note that input shapes can be specified on a per-inference basis, but in this case, we only have a single shape.
            input_shape = (config.model.seq_length, batch_size)
            input_nbytes = trt.volume(input_shape) * trt.int32.itemsize

            # Now set the profile specific settings if the batch size changed
            if (engine.num_optimization_profiles > 1
                    and batch_size != previous_bs):
                if (batch_size not in bs_to_profile):
                    # select engine profile
                    selected_profile = -1
                    num_binding_per_profile = engine.num_bindings // engine.num_optimization_profiles
                    for idx in range(engine.num_optimization_profiles):
                        profile_shape = engine.get_profile_shape(
                            profile_index=idx,
                            binding=idx * num_binding_per_profile)
                        if profile_shape[0][1] <= batch_size and profile_shape[
                                2][1] >= batch_size and profile_shape[0][
                                    0] <= config.model.seq_length and profile_shape[
                                        2][0] >= config.model.seq_length:
                            selected_profile = idx
                            break
                    if selected_profile == -1:
                        raise RuntimeError(
                            "Could not find any profile that can run batch size {}."
                            .format(batch_size))

                    bs_to_profile[batch_size] = selected_profile

                # Set the actual profile
                context.active_optimization_profile = bs_to_profile[batch_size]
                binding_idx_offset = bs_to_profile[
                    batch_size] * num_binding_per_profile

                for binding in range(3):
                    context.set_binding_shape(binding_idx_offset + binding,
                                              input_shape)
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

            d_inputs[0].copy_from_device_async(batch.input_ids.data,
                                               batch.input_ids.data.mem.size,
                                               stream=stream)
            d_inputs[1].copy_from_device_async(batch.segment_ids.data,
                                               batch.segment_ids.data.mem.size,
                                               stream=stream)
            d_inputs[2].copy_from_device_async(batch.input_mask.data,
                                               batch.input_mask.data.mem.size,
                                               stream=stream)

            bindings = [0] * binding_idx_offset + [
                int(d_inp) for d_inp in d_inputs
            ] + [int(d_output)]

            # context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            context.execute_async_v2(bindings=bindings,
                                     stream_handle=stream.ptr)

            # h_output_pinmem = cp.cuda.alloc_pinned_memory(d_output.nbytes)

            # h_output = np.frombuffer(h_output_pinmem, np.float32, d_output.size).reshape(d_output.shape)

            # d_output.set(h_output, stream=stream)

            # cuda.memcpy_dtoh_async(h_output, d_output, stream)
            d_output.copy_to_host_async(c_void_p(h_output.ptr),
                                        h_output.mem.size,
                                        stream=stream)

            stream.synchronize()

            inf_queue.task_done()

            progress.update(n=batch_size)
