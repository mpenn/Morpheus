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

import logging
import os

from morpheus.config import Config
from morpheus.config import PipelineModes
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.general_stages import AddClassificationsStage
from morpheus.pipeline.general_stages import MonitorStage
from morpheus.pipeline.inference.inference_triton import TritonInferenceStage
from morpheus.pipeline.input.from_file import FileSourceStage
from morpheus.pipeline.output.serialize import SerializeStage
from morpheus.pipeline.output.to_file import WriteToFileStage
from morpheus.pipeline.output.validation import ValidationStage
from morpheus.pipeline.preprocessing import DeserializeStage
from morpheus.pipeline.preprocessing import PreprocessNLPStage
from morpheus.utils.logging import configure_logging


def run_pipeline():
    # Enable the Morpheus logger
    configure_logging(log_level=logging.DEBUG)

    root_dir = os.environ['MORPHEUS_ROOT']

    config = Config()
    config.mode = PipelineModes.NLP
    config.class_labels = ["score", "pred"]
    config.model_max_batch_size = 32
    config.pipeline_batch_size = 1024
    config.feature_length = 128
    config.edge_buffer_size = 128
    config.num_threads = 1

    val_file_name = os.path.join(TEST_DIRS.validation_data_dir, 'phishing-email-validation-data.jsonlines')
    vocab_file_name = os.path.join(TEST_DIRS.data_dir, 'bert-base-uncased-hash.txt')
    out_file = os.path.join(tmp_path, 'results.csv')
    results_file_name = os.path.join(tmp_path, 'results.json')

    pipe = LinearPipeline(config)
    pipe.set_source(FileSourceStage(config, filename=val_file_name, iterative=False))
    pipe.add_stage(DeserializeStage(config))
    pipe.add_stage(
        PreprocessNLPStage(config,
                           vocab_hash_file=vocab_file_name,
                           truncation=True,
                           do_lower_case=True,
                           add_special_tokens=False))
    pipe.add_stage(
        TritonInferenceStage(config,
                             model_name='phishing-bert-onnx',
                             server_url='localhost:8001',
                             force_convert_inputs=True))
    pipe.add_stage(MonitorStage(config, description="Inference Rate", smoothing=0.001, unit="inf"))
    pipe.add_stage(AddClassificationsStage(config, labels=["pred"], threshold=0.7))
    pipe.add_stage(
        ValidationStage(config, val_file_name=val_file_name, results_file_name=results_file_name, rel_tol=0.05))
    pipe.add_stage(SerializeStage(config))
    pipe.add_stage(WriteToFileStage(config, filename=out_file, overwrite=False))

    pipe.run()
    results = calc_error_val(results_file_name)

if __name__ == '__main__':
    run_pipeline()
