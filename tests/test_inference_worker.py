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

import asyncio
import unittest
from unittest import mock

import pytest

from morpheus.config import Config
from morpheus.pipeline.inference import inference_stage
from morpheus.utils.producer_consumer_queue import Closed
from morpheus.utils.producer_consumer_queue import ProducerConsumerQueue
from tests import BaseMorpheusTest


@pytest.mark.usefixtures("config_no_cpp")
class TestInferenceWorker(BaseMorpheusTest):
    def test_constructor(self):

        pq = ProducerConsumerQueue()
        iw = inference_stage.InferenceWorker(pq)
        self.assertIs(iw._inf_queue, pq)

        # Call empty methods
        iw.init()
        iw.stop()

    def test_build_output_message(self):
        # build_output_message calls calc_output_dims which is abstract
        # creating a subclass for the purpose of testing
        class TestIW(inference_stage.InferenceWorker):
            def calc_output_dims(self, _):
                return (1, 2)

        config = Config.get()
        config.use_cpp = False # C++ doesn't like our mocked messages

        pq = ProducerConsumerQueue()
        iw = TestIW(pq)

        mock_message = mock.MagicMock()
        mock_message.count = 2
        mock_message.mess_offset = 11
        mock_message.mess_count = 10
        mock_message.offset = 12

        response = iw.build_output_message(mock_message)
        self.assertEqual(response.count, 2)
        self.assertEqual(response.mess_offset, 11)
        self.assertEqual(response.mess_count, 10)
        self.assertEqual(response.offset, 12)


if __name__ == '__main__':
    unittest.main()
