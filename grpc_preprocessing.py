import asyncio
import concurrent.futures
import os
import sys
import typing

import cudf
import grpc.aio
import cupy as cp

from config import Config
from cudf_subword_helper import tokenize_text_series
from request import MultiRequest, SingleRequest

# Add generated proto output to the path. Stupid. See https://github.com/protocolbuffers/protobuf/issues/1491
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "services")))

from services import request_pb2, request_pb2_grpc

config = Config.get()

def cupy_to_grpc(in_cp: cp.ndarray, out_grpc: request_pb2.CudaArrayPayload = None):

    if (out_grpc is None):
        out_grpc = request_pb2.CudaArrayPayload()

    out_grpc.Clear()
    out_grpc.shape.extend(list(in_cp.__cuda_array_interface__["shape"]))
    out_grpc.typestr = in_cp.__cuda_array_interface__["typestr"]
    out_grpc.data = in_cp.__cuda_array_interface__["data"][0]
    out_grpc.readonly = in_cp.__cuda_array_interface__["data"][1]
    out_grpc.version = in_cp.__cuda_array_interface__["version"]

    if ("strides" in in_cp.__cuda_array_interface__ and in_cp.__cuda_array_interface__["strides"] is not None):
        out_grpc.strides.extend(list(in_cp.__cuda_array_interface__["strides"]))

    return out_grpc

def process_batch(messages: cudf.DataFrame) -> typing.List[SingleRequest]:

    data_series = messages["data"]

    tokenized = tokenize_text_series(data_series, config.model.seq_length, 1, config.model.vocab_hash_file)

    return SingleRequest.from_feature(tokenized, data_series)

class PreprocessingServicer(request_pb2_grpc.PipelineServicer):
    """Provides methods that implement functionality of route guide server."""
    def __init__(self) -> None:
        self.hold = []

    def QueuePreprocessing(self, request: request_pb2.StageInput, unused_context) -> request_pb2.StageOutput:

        response = request_pb2.StageOutput(id=request.id, count=request.count)

        try:
            df = cudf.DataFrame({"data": request.payload["data"].str_array.value})

            single_requests = process_batch(df)

            # To multi
            multi_request = MultiRequest(offset=0, count=len(single_requests), data=single_requests[0].data)

            # TEMP: Hold onto a reference to this multi request so the data doesnt go out of scope.
            self.hold.append(multi_request)

            # Now fire off to next stage

            # For now, just return
            cupy_to_grpc(multi_request.input_ids, response.payload["input_ids"].cupy_array)
            cupy_to_grpc(multi_request.input_mask, response.payload["input_mask"].cupy_array)
            cupy_to_grpc(multi_request.segment_ids, response.payload["segment_ids"].cupy_array)
            response.payload["input_str"].str_array.value.extend(multi_request.input_str)

            return single_requests
        except Exception as ex:
            print(ex)
        finally:
            return response


async def serve() -> None:
    server = grpc.aio.server(migration_thread_pool=concurrent.futures.ThreadPoolExecutor(max_workers=10))
    request_pb2_grpc.add_PipelineServicer_to_server(PreprocessingServicer(), server)
    ip_port = '[::]:50051'
    server.add_insecure_port(ip_port)
    await server.start()
    print("Server started! Listening on: {}".format(ip_port))
    await server.wait_for_termination()


def run():
    asyncio.get_event_loop().run_until_complete(serve())
    
