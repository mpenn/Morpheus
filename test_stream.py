'''
To stream from Kafka, please refer to the streamz API here: https://streamz.readthedocs.io/en/latest/api.html#sources 
You can refer to https://kafka.apache.org/quickstart to start a local Kafka cluster.
Below is an example snippet to start a stream from Kafka.
'''
import asyncio
from functools import reduce
import cudf
from distributed.client import wait
from streamz import Stream, Source
from streamz.dataframe import DataFrame
import time
import json
from cudf_subword_helper import tokenize_text_series, Feature
import tensorrt as trt
import numpy as np
import cupy as cp
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

# from tornado.ioloop import IOLoop

# from tornado.platform.asyncio import AsyncIOMainLoop
# AsyncIOMainLoop().install()

# source = Stream.from_textfile('pcap_dump_small.json')  # doctest: +SKIP
# source.map(json.loads).sink(print)  # doctest: +SKIP
# source.start()  # doctest: +SKIP


# Kafka topic to read streaming data from
topic = "test_pcap"

# Kafka brokers
bootstrap_servers = '172.17.0.1:49156,172.17.0.1:49157'

# Kafka consumer configuration
consumer_conf = {'bootstrap.servers': bootstrap_servers,
                 'group.id': 'custreamz',
                 'session.timeout.ms': "60000"}

'''
If you changed Dask=True, please ensure you have a Dask cluster up and running. Refer to this for Dask API: https://docs.dask.org/en/latest/
If the input data is high throughput, we recommend using Dask, and starting appropriate number of Dask workers.
In case of high-throughput processing jobs, please set appropriate number of Kafka topic partitions to ensure parallelism.
To leverage the custreamz' accelerated Kafka reader (note that data in Kafka must be JSON format), use engine="cudf".
'''
source: Stream = Stream.from_kafka_batched(topic, consumer_conf, npartitions=None, start=True,
                                   asynchronous=False, dask=False, engine="cudf", max_batch_size=8)

out_source = Stream()

def filter_empty_data(messages):
    # Filter out null data columns and only return data
    messages = messages[~messages.data.isnull()]
    return messages

def to_cpu_dicts(messages):
    return messages.to_pandas().to_dict('records')

def process_batch(messages: cudf.DataFrame):

    data_series = messages["data"]

    tokenized = tokenize_text_series(data_series,128, 1, 'vocab_hash.txt')

    return tokenized

'''
This is a helper function to do some data pre-processing.
This also prints out the word count for each batch.
'''
def process_message(message: dict):

    # Turn dict message into 

    # return batch_df
    print("Received {} messages".format(len(messages)))

    # Filter out null data columns and only return data
    messages = messages[~messages.data.isnull()]
    return messages

with open("/home/mdemoret/Repos/github/NVIDIA/TensorRT/demo/BERT/engines/bert_large_128.engine", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime, runtime.deserialize_cuda_engine(f.read()) as engine:

    with engine.create_execution_context() as context:

        stream = cp.cuda.Stream(non_blocking=True)
        output_shape = tuple(context.get_binding_shape(3))
        output_size = trt.volume(output_shape) * trt.float32.itemsize

        def do_inference(batch: Feature):

            d_inputs = [
                batch.input_ids.ravel().data.ptr,
                batch.segment_ids.ravel().data.ptr,
                batch.input_mask.ravel().data.ptr,
            ]

            d_output = cp.empty(output_shape, dtype=np.float32)

            context.execute_async_v2(bindings=[int(d_inp) for d_inp in d_inputs] + [d_output.ravel().data.ptr], stream_handle=stream.ptr)

            h_output_pinmem = cp.cuda.alloc_pinned_memory(d_output.size)

            h_output = np.frombuffer(h_output_pinmem, np.float32, d_output.size).reshape(d_output.shape)

            d_output.set(h_output, stream=stream)

            stream.synchronize()


        # Preprocess each batch
        stream_df = source.map(filter_empty_data) \
                        .filter(lambda x: len(x) > 0) \
                        .map(process_batch)

        stream_df.sink(do_inference)

# Now sink it to a trt engine

# # Create a streamz dataframe to get stateful word count
# sdf = DataFrame(stream_df, example=cudf.DataFrame({'word':[], 'count':[]}))

# # Formatting the print statements
# def print_format(sdf):
#     print("\nGlobal Word Count:")
#     return sdf

# # Print cumulative word count from the start of the stream, after every batch. 
# # One can also sink the output to a list.
# sdf.groupby('word').sum().stream.gather().map(print_format).sink(print)

stream_df.sink(print)

# IOLoop.current().start()
def run_asyncio_loop():
    loop = asyncio.get_event_loop()
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()
        print("Exited")

run_asyncio_loop()
