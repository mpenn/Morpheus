#!/usr/bin/env python3
# Copyright (c) 2022, NVIDIA CORPORATION.
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

import logging
import os

import psutil
from from_rabbitmq import RabbitMQSourceStage

from morpheus.config import Config
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.file_types import FileTypes
from morpheus.pipeline.general_stages import MonitorStage
from morpheus.pipeline.output.to_file import WriteToFileStage
from morpheus.utils.logging import configure_logging


def run_pipeline():
    # Enable the Morpheus logger
    configure_logging(log_level=logging.DEBUG)

    config = Config()
    config.num_threads = psutil.cpu_count()

    # Create a linear pipeline object
    pipeline = LinearPipeline(config)

    # Set source stage
    pipeline.set_source(RabbitMQSourceStage(config, host='localhost', exchange='logs'))

    # Add monitor to record the performance of our new stages
    pipeline.add_stage(MonitorStage(config))

    # Write the to the output file
    pipeline.add_stage(WriteToFileStage(config, filename='/tmp/results.json', file_type=FileTypes.JSON, overwrite=True))

    # Run the pipeline
    pipeline.run()

if __name__ == "__main__":
    run_pipeline()
