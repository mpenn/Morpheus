# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import inspect
import json
import time
import typing
from abc import abstractmethod
from functools import partial

import cupy as cp
import neo
import numpy as np
import pandas as pd
import typing_utils

import cudf
from cudf.core.subword_tokenizer import SubwordTokenizer

import morpheus._lib.stages as neos
from morpheus.config import Config
from morpheus.pipeline.messages import InferenceMemoryFIL
from morpheus.pipeline.messages import InferenceMemoryNLP
from morpheus.pipeline.messages import MessageMeta
from morpheus.pipeline.messages import MultiInferenceFILMessage
from morpheus.pipeline.messages import MultiInferenceMessage
from morpheus.pipeline.messages import MultiInferenceNLPMessage
from morpheus.pipeline.messages import MultiMessage
from morpheus.pipeline.pipeline import MultiMessageStage
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamPair
from morpheus.pipeline.pipeline import get_time_ms
from morpheus.utils.cudf_subword_helper import tokenize_text_series


class DeserializeStage(MultiMessageStage):
    """
    This stage deserialize the output of `FileSourceStage`/`KafkaSourceStage` into a `MultiMessage`. This
    should be one of the first stages after the `Source` object.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

    """
    def __init__(self, c: Config):
        super().__init__(c)

        self._batch_size = c.pipeline_batch_size

        self._max_concurrent = c.num_threads

        # Mark these stages to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        return "deserialize"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        """
        return (MessageMeta)

    @staticmethod
    def process_dataframe(x: MessageMeta, batch_size: int) -> typing.List[MultiMessage]:
        """
        The deserialization of the cudf is implemented in this function.

        Parameters
        ----------
        x : cudf.DataFrame
            Input rows that needs to be deserilaized.

        """

        full_message = MultiMessage(meta=x, mess_offset=0, mess_count=x.count)

        # Now break it up by batches
        output = []

        for i in range(0, full_message.mess_count, batch_size):
            output.append(full_message.get_slice(i, min(i + batch_size, full_message.mess_count)))

        return output

    @staticmethod
    def add_start_time(x: MultiMessage):

        curr_time = get_time_ms()

        x.set_meta("ts_start", curr_time)

        return x

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]
        out_type = MultiMessage

        def deserialize_fn(input: neo.Observable, output: neo.Subscriber):
            def obs_on_next(x: MessageMeta):

                message_list: typing.List[MultiMessage] = DeserializeStage.process_dataframe(x, self._batch_size)

                for y in message_list:

                    output.on_next(y)

            def obs_on_error(x):
                output.on_error(x)

            def obs_on_completed():
                output.on_completed()

            obs = neo.Observer.make_observer(obs_on_next, obs_on_error, obs_on_completed)

            input.subscribe(obs)

        if (Config.get().use_cpp):
            stream = neos.DeserializeStage(seg, self.unique_name, self._batch_size)
        else:
            stream = seg.make_node_full(self.unique_name + "-flatten", deserialize_fn)

        seg.make_edge(input_stream[0], stream)

        return stream, out_type

    def _post_build_single(self, seg: neo.Segment, out_pair: StreamPair) -> StreamPair:

        if (self._should_log_timestamps):

            stream = seg.make_node(self.unique_name + "-ts", DeserializeStage.add_start_time)
            seg.make_edge(out_pair[0], stream)

            # Only have one port
            out_pair = (stream, out_pair[1])

        return super()._post_build_single(seg, out_pair)


class DropNullStage(SinglePortStage):
    """
    Drop null/empty data input entries.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    column : str
        Column name to perform null check.

    """
    def __init__(self, c: Config, column: str):
        super().__init__(c)

        self._column = column

        # Mark these stages to log timestamps if requested
        self._should_log_timestamps = True

    @property
    def name(self) -> str:
        return "dropna"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple
            Accepted input types

        """
        return (MessageMeta, )

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:
        stream = input_stream[0]

        # Finally, flatten to a single stream
        def node_fn(input: neo.Observable, output: neo.Subscriber):
            def obs_on_next(x: MessageMeta):
                
                x.df = x.df[~x.df[self._column].isna()]

                if (not x.empty):
                    output.on_next(x)

            def obs_on_error(x):
                output.on_error(x)

            def obs_on_completed():
                output.on_completed()

            obs = neo.Observer.make_observer(obs_on_next, obs_on_error, obs_on_completed)

            input.subscribe(obs)

        node = seg.make_node_full(self.unique_name, node_fn)
        seg.make_edge(stream, node)
        stream = node

        return stream, input_stream[1]


class PreprocessBaseStage(MultiMessageStage):
    """
    This is a base pre-processing class holding general functionality for all preprocessing stages.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

    """
    def __init__(self, c: Config):
        super().__init__(c)

        self._should_log_timestamps = True

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        """
        return (MultiMessage, )

    @abstractmethod
    def _get_preprocess_fn(self) -> typing.Callable[[MultiMessage], MultiInferenceMessage]:
        pass

    @abstractmethod
    def _get_preprocess_node(self, seg: neo.Segment):
        pass

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]
        out_type = MultiInferenceMessage

        preprocess_fn = self._get_preprocess_fn()

        preproc_sig = inspect.signature(preprocess_fn)

        # If the innerfunction returns a type annotation, update the output type
        if (preproc_sig.return_annotation and typing_utils.issubtype(preproc_sig.return_annotation, out_type)):
            out_type = preproc_sig.return_annotation

        if (Config.get().use_cpp):
            stream = self._get_preprocess_node(seg)
        else:
            stream = seg.make_node(self.unique_name, preprocess_fn)

        seg.make_edge(input_stream[0], stream)

        return stream, out_type


class PreprocessNLPStage(PreprocessBaseStage):
    """
    NLP usecases are preprocessed with this stage class.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

    """
    def __init__(self,
                 c: Config,
                 vocab_hash_file: str,
                 truncation: bool,
                 do_lower_case: bool,
                 add_special_tokens: bool,
                 stride: int = -1):
        super().__init__(c)

        self._seq_length = c.feature_length
        self._vocab_hash_file = vocab_hash_file

        if (stride <= 0):
            # Set the stride to 75%. Works well with powers of 2
            self._stride = self._seq_length // 2
            self._stride = self._stride + self._stride // 2
        else:
            # Use the given value
            self._stride = stride

        self._truncation = truncation
        self._do_lower_case = do_lower_case
        self._add_special_tokens = add_special_tokens

        self._tokenizer: SubwordTokenizer = None

    @property
    def name(self) -> str:
        return "preprocess-nlp"

    @staticmethod
    def pre_process_batch(x: MultiMessage,
                          vocab_hash_file: str,
                          do_lower_case: bool,
                          seq_len: int,
                          stride: int,
                          truncation: bool,
                          add_special_tokens: bool) -> MultiInferenceNLPMessage:
        """
        For NLP category usecases, this function performs pre-processing.

        Parameters
        ----------
        x : morpheus.messages.MultiMessage
            Input rows recieved from Deserialized stage.
        seq_len : int
            Limits the length of the sequence returned. If tokenized string is shorter than max_length, output will be
            padded with 0s. If the tokenized string is longer than max_length and do_truncate == False, there will be
            multiple returned sequences containing the overflowing token-ids.
        stride : int
            If do_truncate == False and the tokenized string is larger than max_length, the sequences containing the
            overflowing token-ids can contain duplicated token-ids from the main sequence. If max_length is equal to
            stride there are no duplicated-id tokens. If stride is 80% of max_length, 20% of the first sequence will be
            repeated on the second sequence and so on until the entire sentence is encoded.
        vocab_hash_file : str
            Path to hash file containing vocabulary of words with token-ids. This can be created from the raw vocabulary
            using the cudf.utils.hash_vocab_utils.hash_vocab function.

        Returns
        -------
        morpheus.messages.MultiInferenceNLPMessage
            infer_message

        """
        text_ser = cudf.Series(x.get_meta("data"))

        tokenized = tokenize_text_series(vocab_hash_file=vocab_hash_file,
                                         do_lower_case=do_lower_case,
                                         text_ser=text_ser,
                                         seq_len=seq_len,
                                         stride=stride,
                                         truncation=truncation,
                                         add_special_tokens=add_special_tokens)
        del text_ser

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
                       vocab_hash_file=self._vocab_hash_file,
                       do_lower_case=self._do_lower_case,
                       stride=self._stride,
                       seq_len=self._seq_length,
                       truncation=self._truncation,
                       add_special_tokens=self._add_special_tokens)

    def _get_preprocess_node(self, seg: neo.Segment):
        return neos.PreprocessNLPStage(seg,
                                       self.unique_name,
                                       self._vocab_hash_file,
                                       self._seq_length,
                                       self._truncation,
                                       self._do_lower_case,
                                       self._add_special_tokens,
                                       self._stride)


class PreprocessFILStage(PreprocessBaseStage):
    """
    FIL usecases are preprocessed with this stage class.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

    """
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

        assert self._fea_length == len(self.features), \
            f"Number of features in preprocessing {len(self.features)}, does not match configuration {self._fea_length}"

    @property
    def name(self) -> str:
        return "preprocess-fil"

    @staticmethod
    def pre_process_batch(x: MultiMessage, fea_len: int, fea_cols: typing.List[str]) -> MultiInferenceFILMessage:
        """
        For FIL category usecases, this function performs pre-processing.

        Parameters
        ----------
        x : morpheus.messages.MultiMessage
            Input rows recieved from Deserialized stage.
        fea_len : int
            Number features are being used in the inference.
        fea_cols : typing.Tuple[str]
            List of columns that are used as features.

        Returns
        -------
        morpheus.messages.MultiInferenceFILMessage
            infer_message

        """

        df = x.get_meta(fea_cols)

        # Extract just the numbers from each feature col. Not great to operate on x.meta.df here but the operations will
        # only happen once.
        for col in fea_cols:
            if (df[col].dtype == np.dtype(str) or df[col].dtype == np.dtype(object)):
                # If the column is a string, parse the number
                df[col] = df[col].str.extract(r"(\d+)", expand=False).astype("float32")
            elif (df[col].dtype != np.float32):
                # Convert to float32
                df[col] = df[col].astype("float32")

        if (isinstance(df, pd.DataFrame)):
            df = cudf.from_pandas(df)

        # Convert the dataframe to cupy the same way cuml does
        data = cp.asarray(df.as_gpu_matrix(order='C'))

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

    def _get_preprocess_node(self, seg: neo.Segment):
        return neos.PreprocessFILStage(seg, self.unique_name)
