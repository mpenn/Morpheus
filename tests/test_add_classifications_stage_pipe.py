#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

import cupy as cp
import numpy as np
import pandas as pd
import pytest

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.file_types import FileTypes
from morpheus.pipeline.general_stages import AddClassificationsStage
from morpheus.pipeline.input.from_file import FileSourceStage
from morpheus.pipeline.input.utils import read_file_to_df
from morpheus.pipeline.messages import MessageMeta
from morpheus.pipeline.messages import MultiMessage
from morpheus.pipeline.messages import MultiResponseProbsMessage
from morpheus.pipeline.messages import ResponseMemoryProbs
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.preprocessing import DeserializeStage
from tests import TEST_DIRS
from tests import BaseMorpheusTest


class ConvMsg(SinglePortStage):
    def __init__(self, c: Config, expected_data_file: str):
        super().__init__(c)
        self._expected_data_file = expected_data_file

    @property
    def name(self):
        return "test"

    def accepted_types(self):
        return (MultiMessage, )

    def _conv_message(self, m):
        df = read_file_to_df(self._expected_data_file, FileTypes.CSV, df_type="cudf")
        probs = df.values
        memory = ResponseMemoryProbs(count=len(probs), probs=probs)
        return MultiResponseProbsMessage(m.meta, 0, len(probs), memory, 0, len(probs))

    def _build_single(self, seg, input_stream):
        stream = seg.make_node(self.unique_name, self._conv_message)
        seg.make_edge(input_stream[0], stream)

        return stream, MultiResponseProbsMessage


class WriteMetaToFileStage(SinglePortStage):
    def __init__(self, c, filename):
        super().__init__(c)

        self._output_file = filename
        self._columns = c.class_labels

    @property
    def name(self):
        return "meta-to-file"

    def accepted_types(self):
        return (MultiMessage, )

    def _write_meta_to_file(self, x):
        df = x.get_meta()
        with open(self._output_file, "a") as f:
            idx = df.columns.intersection(self._columns)
            df[idx].to_csv(f)

        return x

    def _build_single(self, seg, input_stream):
        stream = input_stream[0]
        to_file = seg.make_node(self.unique_name, self._write_meta_to_file)
        seg.make_edge(stream, to_file)
        stream = to_file

        return input_stream

class TestAddClassificationsStagePipe(BaseMorpheusTest):
    def _test_pipe(self, use_cpp):
        config = Config.get()
        config.class_labels = ['frogs', 'lizards', 'toads', 'turtles']
        config.use_cpp = use_cpp
        config.num_threads = 1

        # Silly data with all false values
        input_file = os.path.join(TEST_DIRS.expeced_data_dir, "filter_probs.csv")

        temp_dir = self._mk_tmp_dir()
        out_file = os.path.join(temp_dir, 'results.csv')

        threshold = 0.75

        pipe = LinearPipeline(config)
        pipe.set_source(FileSourceStage(config, filename=input_file, iterative=False))
        pipe.add_stage(DeserializeStage(config))
        pipe.add_stage(ConvMsg(config, input_file))
        pipe.add_stage(AddClassificationsStage(config, threshold=threshold))
        pipe.add_stage(WriteMetaToFileStage(config, filename=out_file))
        pipe.run()

        self.assertTrue(os.path.exists(out_file))

        input_data = np.loadtxt(input_file, delimiter=",", skiprows=1)
        expected = (input_data > threshold)

        # The output data will contain an additional id column that we will need to slice off
        output_data = pd.read_csv(out_file)
        idx = output_data.columns.intersection(config.class_labels)
        self.assertEqual(idx.to_list(), config.class_labels)

        output_np = output_data[idx].to_numpy()

        self.assertEqual(output_np.tolist(), expected.tolist())

    @pytest.mark.slow
    def test_pipe_no_cpp(self):
        self._test_pipe(False)

    @pytest.mark.slow
    def test_pipe_cpp(self):
        self._test_pipe(True)

if __name__ == '__main__':
    unittest.main()
