import inspect
import json
import time
import typing
from abc import abstractclassmethod, abstractmethod
from functools import partial

import cudf
import cupy as cp
import typing_utils
from morpheus.config import Config
from morpheus.pipeline import Stage
from morpheus.pipeline.messages import (InferenceMemory,
                                        InferenceMemoryFIL,
                                        InferenceMemoryNLP,
                                        MessageMeta,
                                        MultiInferenceFILMessage,
                                        MultiInferenceMessage,
                                        MultiInferenceNLPMessage,
                                        MultiMessage)
from morpheus.pipeline.pipeline import StreamFuture, StreamPair, get_time_ms
from morpheus.utils.cudf_subword_helper import tokenize_text_series


class DeserializeStage(Stage):
    def __init__(self, c: Config):
        super().__init__(c)

        self._use_dask = c.use_dask

        # Mark these stages to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        return "deserialize"

    def accepted_types(self) -> typing.Tuple:
        return (cudf.DataFrame, StreamFuture[cudf.DataFrame])

    @staticmethod
    def process_dataframe(x: cudf.DataFrame):

        # Convert here to pandas since this will persist after the message is done
        x_pd = x.to_pandas()

        # Now determine the list of input strings before modification
        input_json = [json.dumps(y) for y in x_pd.loc[:, x_pd.columns != 'ID'].to_dict(orient="records")]

        # Add the start_time field
        x_pd["ts_start"] = round(time.time() * 1000)

        # Try to double deserialize
        def deserialize_data(y: str):
            try:
                return str(json.loads(y))
            except:
                return y

        if ("data" in x_pd):
            x_pd["data"] = x_pd["data"].apply(deserialize_data)

        # Build the message data
        meta = MessageMeta(df=x_pd, input_json=input_json)

        return MultiMessage(meta=meta, mess_offset=0, mess_count=len(x_pd))

    async def _build(self, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]
        out_type = MultiMessage

        if (typing_utils.issubtype(input_stream[1], StreamFuture)):

            stream = stream.map(DeserializeStage.process_dataframe)
            out_type = StreamFuture[MultiMessage]
        else:
            stream = stream.async_map(DeserializeStage.process_dataframe, executor=self._pipeline.thread_pool)

        return stream, out_type

    async def post_timestamps(self, x: MultiMessage):

        curr_time = get_time_ms()

        x.set_meta("ts_" + self.name, curr_time)


class PreprocessBaseStage(Stage):
    def __init__(self, c: Config):
        super().__init__(c)

        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        return "preprocess"

    def accepted_types(self) -> typing.Tuple:
        return (MultiMessage, StreamFuture[MultiMessage])

    @abstractmethod
    def _get_preprocess_fn(self) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:
        pass

    async def _build(self, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]
        out_type = MultiInferenceMessage

        preprocess_fn = self._get_preprocess_fn()

        preproc_sig = inspect.signature(preprocess_fn)

        # If the innerfunction returns a type annotation, update the output type
        if (preproc_sig.return_annotation and typing_utils.issubtype(preproc_sig.return_annotation, out_type)):
            out_type = preproc_sig.return_annotation

        if (typing_utils.issubtype(input_stream[1], StreamFuture)):
            stream = stream.map(preprocess_fn)
            out_type = StreamFuture[out_type]
        else:
            stream = stream.async_map(preprocess_fn, executor=self._pipeline.thread_pool)

        return stream, out_type


class PreprocessNLPStage(PreprocessBaseStage):
    def __init__(self, c: Config):
        super().__init__(c)

        self._seq_length = c.feature_length
        self._vocab_hash_file = c.nlp.model_vocab_hash_file

        # Set the stride to 75%. Works well with powers of 2
        self._stride = self._seq_length // 2
        self._stride = self._stride + self._stride // 2

    @staticmethod
    def pre_process_batch(x: MultiMessage, seq_len: int, stride: int, vocab_hash_file: str) -> MultiInferenceNLPMessage:

        tokenized = tokenize_text_series(cudf.Series(x.get_meta("data")), seq_len, stride, vocab_hash_file)

        # Create the inference memory. Keep in mind count here could be > than input count
        memory = InferenceMemoryNLP(count=tokenized.input_ids.shape[0],
                                    input_ids=tokenized.input_ids,
                                    input_mask=tokenized.input_mask,
                                    seq_ids=tokenized.segment_ids)

        infer_message = MultiInferenceNLPMessage(meta=x.meta,
                                                 mess_offset=x.mess_offset,
                                                 mess_count=x.mess_count,
                                                 memory=memory,
                                                 offset=0,
                                                 count=memory.count)

        return infer_message

    def _get_preprocess_fn(self) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:
        return partial(PreprocessNLPStage.pre_process_batch,
                       stride=self._stride,
                       seq_len=self._seq_length,
                       vocab_hash_file=self._vocab_hash_file)


class PreprocessFILStage(PreprocessBaseStage):
    def __init__(self, c: Config):
        super().__init__(c)

        self._fea_length = c.feature_length

        self.features = [
            "nvidia_smi_log.gpu.pci.tx_util",
            "nvidia_smi_log.gpu.pci.rx_util",
            "nvidia_smi_log.gpu.fb_memory_usage.used",
            "nvidia_smi_log.gpu.fb_memory_usage.free",
            "nvidia_smi_log.gpu.bar1_memory_usage.total",
            "nvidia_smi_log.gpu.bar1_memory_usage.used",
            "nvidia_smi_log.gpu.bar1_memory_usage.free",
            "nvidia_smi_log.gpu.utilization.gpu_util",
            "nvidia_smi_log.gpu.utilization.memory_util",
            "nvidia_smi_log.gpu.temperature.gpu_temp",
            "nvidia_smi_log.gpu.temperature.gpu_temp_max_threshold",
            "nvidia_smi_log.gpu.temperature.gpu_temp_slow_threshold",
            "nvidia_smi_log.gpu.temperature.gpu_temp_max_gpu_threshold",
            "nvidia_smi_log.gpu.temperature.memory_temp",
            "nvidia_smi_log.gpu.temperature.gpu_temp_max_mem_threshold",
            "nvidia_smi_log.gpu.power_readings.power_draw",
            "nvidia_smi_log.gpu.clocks.graphics_clock",
            "nvidia_smi_log.gpu.clocks.sm_clock",
            "nvidia_smi_log.gpu.clocks.mem_clock",
            "nvidia_smi_log.gpu.clocks.video_clock",
            "nvidia_smi_log.gpu.applications_clocks.graphics_clock",
            "nvidia_smi_log.gpu.applications_clocks.mem_clock",
            "nvidia_smi_log.gpu.default_applications_clocks.graphics_clock",
            "nvidia_smi_log.gpu.default_applications_clocks.mem_clock",
            "nvidia_smi_log.gpu.max_clocks.graphics_clock",
            "nvidia_smi_log.gpu.max_clocks.sm_clock",
            "nvidia_smi_log.gpu.max_clocks.mem_clock",
            "nvidia_smi_log.gpu.max_clocks.video_clock",
            "nvidia_smi_log.gpu.max_customer_boost_clocks.graphics_clock",
        ]

        assert self._fea_length == len(self.features), f"Number of features in preprocessing {len(self.features)}, does not match configuration {self._fea_length}"

    @staticmethod
    def pre_process_batch(x: MultiMessage, fea_len: int, fea_cols: typing.List[str]) -> MultiInferenceFILMessage:

        # Drop some extra columns we dont need
        x.meta.df.drop(x.meta.df.columns.difference(fea_cols + ["ts_start", "ts_deserialize"]), 1, inplace=True)

        # Extract just the numbers from each feature col
        for col in fea_cols:
            x.meta.df[col] = x.meta.df[col].str.extract(r"(\d+)", expand=False).astype("float32")

        # Convert the dataframe to cupy the same way cuml does
        data = cp.asarray(cudf.from_pandas(x.meta.df[fea_cols]).as_gpu_matrix(order='C'))

        count = data.shape[0]

        seg_ids = cp.zeros((count, 3), dtype=cp.uint32)
        seg_ids[:, 0] = cp.arange(0, count, dtype=cp.uint32)
        seg_ids[:, 2] = fea_len - 1

        # Create the inference memory. Keep in mind count here could be > than input count
        memory = InferenceMemoryFIL(count=count, input__0=data, seq_ids=seg_ids)

        infer_message = MultiInferenceFILMessage(meta=x.meta,
                                                 mess_offset=x.mess_offset,
                                                 mess_count=x.mess_count,
                                                 memory=memory,
                                                 offset=0,
                                                 count=memory.count)

        return infer_message

    def _get_preprocess_fn(self) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:
        return partial(PreprocessFILStage.pre_process_batch, fea_len=self._fea_length, fea_cols=self.features)
