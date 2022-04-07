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

import click
import mlflow
import pytest
from click.testing import CliRunner
from mlflow.tracking import fluent

from morpheus import cli
from morpheus.config import ConfigAutoEncoder
from morpheus.config import PipelineModes
from morpheus.pipeline import LinearPipeline
from morpheus.pipeline.general_stages import AddClassificationsStage
from morpheus.pipeline.general_stages import AddScoresStage
from morpheus.pipeline.general_stages import FilterDetectionsStage
from morpheus.pipeline.general_stages import MonitorStage
from morpheus.pipeline.inference.inference_ae import AutoEncoderInferenceStage
from morpheus.pipeline.inference.inference_identity import IdentityInferenceStage
from morpheus.pipeline.inference.inference_pytorch import PyTorchInferenceStage
from morpheus.pipeline.inference.inference_triton import TritonInferenceStage
from morpheus.pipeline.input.from_cloudtrail import CloudTrailSourceStage
from morpheus.pipeline.input.from_file import FileSourceStage
from morpheus.pipeline.input.from_kafka import KafkaSourceStage
from morpheus.pipeline.output.serialize import SerializeStage
from morpheus.pipeline.output.to_file import WriteToFileStage
from morpheus.pipeline.output.to_kafka import WriteToKafkaStage
from morpheus.pipeline.output.validation import ValidationStage
from morpheus.pipeline.postprocess.mlflow_drift import MLFlowDriftStage
from morpheus.pipeline.postprocess.timeseries import TimeSeriesStage
from morpheus.pipeline.preprocess.autoencoder import PreprocessAEStage
from morpheus.pipeline.preprocess.autoencoder import TrainAEStage
from morpheus.pipeline.preprocessing import DeserializeStage
from morpheus.pipeline.preprocessing import DropNullStage
from morpheus.pipeline.preprocessing import PreprocessFILStage
from morpheus.pipeline.preprocessing import PreprocessNLPStage
from tests import TEST_DIRS

GENERAL_ARGS = ['run', '--num_threads=12', '--pipeline_batch_size=1024', '--model_max_batch_size=1024', '--use_cpp=0']
MONITOR_ARGS = ['monitor', '--description', 'Unittest', '--smoothing=0.001', '--unit', 'inf']
VALIDATE_ARGS = [
    'validate',
    '--val_file_name',
    os.path.join(TEST_DIRS.validation_data_dir, 'hammah-role-g-validation-data.csv'),
    '--results_file_name=results.json',
    '--index_col=_index_',
    '--exclude',
    'event_dt',
    '--rel_tol=0.1'
]
TO_FILE_ARGS = ['to-file', '--filename=out.csv']
FILE_SRC_ARGS = [
    'from-file', '--filename', os.path.join(TEST_DIRS.validation_data_dir, 'abp-validation-data.jsonlines')
]
INF_TRITON_ARGS = ['inf-triton', '--model_name=test-model', '--server_url=test:123', '--force_convert_inputs=True']

KAFKA_BOOTS = ['--bootstrap_servers', 'kserv1:123,kserv2:321']
FROM_KAFKA_ARGS = ['from-kafka', '--input_topic', 'test_topic'] + KAFKA_BOOTS
TO_KAFKA_ARGS = ['to-kafka', '--output_topic', 'test_topic'] + KAFKA_BOOTS


# Fixtures specific to the cli tests
@pytest.fixture(scope="function")
def callback_values(request: pytest.FixtureRequest):
    """
    Replaces the results_callback in cli which executes the pipeline.
    Allowing us to examine/verify that cli built us a propper pipeline
    without actually running it. When run the callback will update the
    `callback_values` dictionary with the context, pipeline & stages constructed.
    """
    cv = {}

    marker = request.node.get_closest_marker("replace_callback")
    group_name = marker.args[0]
    group = getattr(cli, group_name)

    @group.result_callback(replace=True)
    @click.pass_context
    def mock_post_callback(ctx, stages, *a, **k):
        cv.update({'ctx': ctx, 'stages': stages, 'pipe': ctx.find_object(LinearPipeline)})
        ctx.exit(47)

    return cv


@pytest.fixture(scope="function")
def mlflow_uri(tmp_path):
    experiment_name = "Morpheus"
    uri = "file://{}".format(tmp_path)
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    with mlflow.start_run(run_name="Model Drift",
                          tags={"morpheus.type": "drift"},
                          experiment_id=experiment.experiment_id):
        pass

    yield uri

    num_runs = len(fluent._active_run_stack)
    for _ in range(num_runs):
        mlflow.end_run()


@pytest.mark.reload_modules(cli)
@pytest.mark.usefixtures("reload_modules")
@pytest.mark.use_python
class TestCLI:

    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['--help'])
        assert result.exit_code == 0, result.output

        result = runner.invoke(cli.cli, ['tools', '--help'])
        assert result.exit_code == 0, result.output

        result = runner.invoke(cli.cli, ['run', '--help'])
        assert result.exit_code == 0, result.output

        result = runner.invoke(cli.cli, ['run', 'pipeline-ae', '--help'])
        assert result.exit_code == 0, result.output

    def test_autocomplete(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli.cli, ['tools', 'autocomplete', 'show'], env={'HOME': str(tmp_path)})
        assert result.exit_code == 0, result.output

        # The actual results of this are specific to the implementation of click_completion
        result = runner.invoke(cli.cli, ['tools', 'autocomplete', 'install', 'bash'], env={'HOME': str(tmp_path)})
        assert result.exit_code == 0, result.output

    @pytest.mark.replace_callback('pipeline_ae')
    def test_pipeline_ae(self, config, callback_values):
        """
        Build a pipeline roughly ressembles the hammah validation script
        """
        args = GENERAL_ARGS + \
               ['pipeline-ae', '--userid_filter=user321', '--userid_column_name=user_col',
                'from-cloudtrail', '--input_glob=input_glob*.csv',
                'train-ae', '--train_data_glob=train_glob*.csv', '--seed', '47',
                'preprocess', 'inf-pytorch', 'add-scores',
                'timeseries', '--resolution=1m', '--zscore_threshold=8.0', '--hot_start'] + \
               MONITOR_ARGS + VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS

        runner = CliRunner()
        result = runner.invoke(cli.cli, args)
        assert result.exit_code == 47, result.output

        # Ensure our config is populated correctly

        assert config.mode == PipelineModes.AE
        assert not config.use_cpp
        assert config.class_labels == ["ae_anomaly_score"]
        assert config.model_max_batch_size == 1024
        assert config.pipeline_batch_size == 1024
        assert config.num_threads == 12

        assert isinstance(config.ae, ConfigAutoEncoder)
        config.ae.userid_column_name = "user_col"
        config.ae.userid_filter = "user321"

        pipe = callback_values['pipe']
        assert pipe is not None

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        [cloud_trail, train_ae, process_ae, auto_enc, add_scores, time_series, monitor, validation, serialize,
         to_file] = stages

        assert isinstance(cloud_trail, CloudTrailSourceStage)
        assert cloud_trail._input_glob == "input_glob*.csv"

        assert isinstance(train_ae, TrainAEStage)
        assert train_ae._train_data_glob == "train_glob*.csv"
        assert train_ae._seed == 47

        assert isinstance(process_ae, PreprocessAEStage)
        assert isinstance(auto_enc, AutoEncoderInferenceStage)
        assert isinstance(add_scores, AddScoresStage)

        assert isinstance(time_series, TimeSeriesStage)
        assert time_series._resolution == '1m'
        assert time_series._zscore_threshold == 8.0
        assert time_series._hot_start

        assert isinstance(monitor, MonitorStage)
        assert monitor._description == 'Unittest'
        assert monitor._smoothing == 0.001
        assert monitor._unit == 'inf'

        assert isinstance(validation, ValidationStage)
        assert validation._val_file_name == os.path.join(TEST_DIRS.validation_data_dir,
                                                         'hammah-role-g-validation-data.csv')
        assert validation._results_file_name == 'results.json'
        assert validation._index_col == '_index_'

        # Click appears to be converting this into a tuple
        assert list(validation._exclude_columns) == ['event_dt']
        assert validation._rel_tol == 0.1

        assert isinstance(serialize, SerializeStage)

        assert isinstance(to_file, WriteToFileStage)
        assert to_file._output_file == 'out.csv'

    @pytest.mark.replace_callback('pipeline_ae')
    def test_pipeline_ae_all(self, config, callback_values, tmp_path):
        """
        Attempt to add all possible stages to the pipeline_ae, even if the pipeline doesn't
        actually make sense, just test that cli could assemble it
        """
        args = GENERAL_ARGS + \
               ['pipeline-ae', '--userid_filter=user321', '--userid_column_name=user_col',
                'from-cloudtrail', '--input_glob=input_glob*.csv',
                'add-class', 'filter',
                'train-ae', '--train_data_glob=train_glob*.csv', '--seed', '47',
                'preprocess', 'inf-pytorch', 'add-scores'] + \
               INF_TRITON_ARGS + \
               ['timeseries', '--resolution=1m', '--zscore_threshold=8.0', '--hot_start'] + \
               MONITOR_ARGS + VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS + TO_KAFKA_ARGS

        runner = CliRunner()
        result = runner.invoke(cli.cli, args)

        assert result.exit_code == 47, result.output

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        [
            cloud_trail,
            add_class,
            filter_stage,
            train_ae,
            process_ae,
            auto_enc,
            add_scores,
            triton_inf,
            time_series,
            monitor,
            validation,
            serialize,
            to_file,
            to_kafka
        ] = stages

        assert isinstance(cloud_trail, CloudTrailSourceStage)
        assert cloud_trail._input_glob == "input_glob*.csv"

        assert isinstance(add_class, AddClassificationsStage)
        assert isinstance(filter_stage, FilterDetectionsStage)

        assert isinstance(train_ae, TrainAEStage)
        assert train_ae._train_data_glob == "train_glob*.csv"
        assert train_ae._seed == 47

        assert isinstance(process_ae, PreprocessAEStage)
        assert isinstance(auto_enc, AutoEncoderInferenceStage)
        assert isinstance(add_scores, AddScoresStage)

        assert isinstance(triton_inf, TritonInferenceStage)
        assert triton_inf._kwargs['model_name'] == 'test-model'
        assert triton_inf._kwargs['server_url'] == 'test:123'
        assert triton_inf._kwargs['force_convert_inputs']

        assert isinstance(time_series, TimeSeriesStage)
        assert time_series._resolution == '1m'
        assert time_series._zscore_threshold == 8.0
        assert time_series._hot_start

        assert isinstance(monitor, MonitorStage)
        assert monitor._description == 'Unittest'
        assert monitor._smoothing == 0.001
        assert monitor._unit == 'inf'

        assert isinstance(validation, ValidationStage)
        assert validation._val_file_name == os.path.join(TEST_DIRS.validation_data_dir,
                                                         'hammah-role-g-validation-data.csv')
        assert validation._results_file_name == 'results.json'
        assert validation._index_col == '_index_'

        # Click appears to be converting this into a tuple
        assert list(validation._exclude_columns) == ['event_dt']
        assert validation._rel_tol == 0.1

        assert isinstance(serialize, SerializeStage)

        assert isinstance(to_file, WriteToFileStage)
        assert to_file._output_file == 'out.csv'

        assert isinstance(to_kafka, WriteToKafkaStage)
        assert to_kafka._kafka_conf['bootstrap.servers'] == 'kserv1:123,kserv2:321'
        assert to_kafka._output_topic == 'test_topic'

    @pytest.mark.replace_callback('pipeline_fil')
    def test_pipeline_fil(self, config, callback_values, tmp_path):
        """
        Creates a pipeline roughly matching that of the abp validation test
        """
        args = GENERAL_ARGS + ['pipeline-fil'] + FILE_SRC_ARGS + ['deserialize', 'preprocess'] + INF_TRITON_ARGS + \
               MONITOR_ARGS + ['add-class'] + VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS

        runner = CliRunner()
        result = runner.invoke(cli.cli, args)
        assert result.exit_code == 47, result.output

        # Ensure our config is populated correctly
        assert config.mode == PipelineModes.FIL
        assert config.class_labels == ["mining"]

        assert config.ae is None

        pipe = callback_values['pipe']
        assert pipe is not None

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        [file_source, deserialize, process_fil, triton_inf, monitor, add_class, validation, serialize, to_file] = stages

        assert isinstance(file_source, FileSourceStage)
        assert file_source._filename == os.path.join(TEST_DIRS.validation_data_dir, 'abp-validation-data.jsonlines')
        assert not file_source._iterative

        assert isinstance(deserialize, DeserializeStage)
        assert isinstance(process_fil, PreprocessFILStage)

        assert isinstance(triton_inf, TritonInferenceStage)
        assert triton_inf._kwargs['model_name'] == 'test-model'
        assert triton_inf._kwargs['server_url'] == 'test:123'
        assert triton_inf._kwargs['force_convert_inputs']

        assert isinstance(monitor, MonitorStage)
        assert monitor._description == 'Unittest'
        assert monitor._smoothing == 0.001
        assert monitor._unit == 'inf'

        assert isinstance(add_class, AddClassificationsStage)

        assert isinstance(validation, ValidationStage)
        assert validation._val_file_name == os.path.join(TEST_DIRS.validation_data_dir,
                                                         'hammah-role-g-validation-data.csv')
        assert validation._results_file_name == 'results.json'
        assert validation._index_col == '_index_'

        # Click appears to be converting this into a tuple
        assert list(validation._exclude_columns) == ['event_dt']
        assert validation._rel_tol == 0.1

        assert isinstance(serialize, SerializeStage)
        assert isinstance(to_file, WriteToFileStage)
        assert to_file._output_file == 'out.csv'

    @pytest.mark.replace_callback('pipeline_fil')
    def test_pipeline_fil_all(self, config, callback_values, tmp_path, mlflow_uri):
        """
        Attempt to add all possible stages to the pipeline_fil, even if the pipeline doesn't
        actually make sense, just test that cli could assemble it
        """
        tmp_model = os.path.join(tmp_path, 'fake-model.file')
        with open(tmp_model, 'w') as fh:
            pass

        labels_file = os.path.join(tmp_path, 'labels.txt')
        with open(labels_file, 'w') as fh:
            fh.writelines(['frogs\n', 'lizards\n', 'toads'])

        args = GENERAL_ARGS + \
               ['pipeline-fil', '--labels_file', labels_file] + \
               FILE_SRC_ARGS + FROM_KAFKA_ARGS +\
               ['deserialize', 'filter',
                'dropna', '--column', 'xyz',
                'preprocess', 'add-scores', 'inf-identity',
                'inf-pytorch', '--model_filename', tmp_model,
                'mlflow-drift', '--tracking_uri', mlflow_uri] + \
               INF_TRITON_ARGS + MONITOR_ARGS + ['add-class'] + VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS + TO_KAFKA_ARGS

        runner = CliRunner()
        result = runner.invoke(cli.cli, args)
        assert result.exit_code == 47, result.output

        # Ensure our config is populated correctly

        assert config.mode == PipelineModes.FIL
        assert config.class_labels == ['frogs', 'lizards', 'toads']

        assert config.ae is None

        pipe = callback_values['pipe']
        assert pipe is not None

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        [
            file_source,
            from_kafka,
            deserialize,
            filter_stage,
            dropna,
            process_fil,
            add_scores,
            inf_ident,
            inf_pytorch,
            mlflow_drift,
            triton_inf,
            monitor,
            add_class,
            validation,
            serialize,
            to_file,
            to_kafka
        ] = stages

        assert isinstance(file_source, FileSourceStage)
        assert file_source._filename == os.path.join(TEST_DIRS.validation_data_dir, 'abp-validation-data.jsonlines')
        assert not file_source._iterative

        assert isinstance(from_kafka, KafkaSourceStage)
        assert from_kafka._consumer_conf['bootstrap.servers'] == 'kserv1:123,kserv2:321'
        assert from_kafka._input_topic == 'test_topic'

        assert isinstance(deserialize, DeserializeStage)
        assert isinstance(filter_stage, FilterDetectionsStage)

        assert isinstance(dropna, DropNullStage)
        assert dropna._column == 'xyz'

        assert isinstance(process_fil, PreprocessFILStage)

        assert isinstance(add_scores, AddScoresStage)
        assert isinstance(inf_ident, IdentityInferenceStage)

        assert isinstance(inf_pytorch, PyTorchInferenceStage)
        assert inf_pytorch._model_filename == tmp_model

        assert isinstance(mlflow_drift, MLFlowDriftStage)
        assert mlflow_drift._tracking_uri == mlflow_uri

        assert isinstance(triton_inf, TritonInferenceStage)
        assert triton_inf._kwargs['model_name'] == 'test-model'
        assert triton_inf._kwargs['server_url'] == 'test:123'
        assert triton_inf._kwargs['force_convert_inputs']

        assert isinstance(monitor, MonitorStage)
        assert monitor._description == 'Unittest'
        assert monitor._smoothing == 0.001
        assert monitor._unit == 'inf'

        assert isinstance(add_class, AddClassificationsStage)

        assert isinstance(validation, ValidationStage)
        assert validation._val_file_name == os.path.join(TEST_DIRS.validation_data_dir,
                                                         'hammah-role-g-validation-data.csv')
        assert validation._results_file_name == 'results.json'
        assert validation._index_col == '_index_'

        # Click appears to be converting this into a tuple
        assert list(validation._exclude_columns) == ['event_dt']
        assert validation._rel_tol == 0.1

        assert isinstance(serialize, SerializeStage)

        assert isinstance(to_file, WriteToFileStage)
        assert to_file._output_file == 'out.csv'

        assert isinstance(to_kafka, WriteToKafkaStage)
        assert to_kafka._kafka_conf['bootstrap.servers'] == 'kserv1:123,kserv2:321'
        assert to_kafka._output_topic == 'test_topic'

    @pytest.mark.replace_callback('pipeline_nlp')
    def test_pipeline_nlp(selff, config, callback_values, tmp_path):
        """
        Build a pipeline roughly ressembles the phishing validation script
        """
        labels_file = os.path.join(TEST_DIRS.data_dir, 'labels_phishing.txt')
        vocab_file_name = os.path.join(TEST_DIRS.data_dir, 'bert-base-uncased-hash.txt')
        args = GENERAL_ARGS + \
               ['pipeline-nlp', '--model_seq_length=128', '--labels_file', labels_file] + \
               FILE_SRC_ARGS +  \
               ['deserialize',
                'preprocess', '--vocab_hash_file', vocab_file_name, '--truncation=True',
                '--do_lower_case=True', '--add_special_tokens=False'] + \
               INF_TRITON_ARGS + MONITOR_ARGS + \
               ['add-class', '--label=pred', '--threshold=0.7'] + \
               VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS

        runner = CliRunner()
        result = runner.invoke(cli.cli, args)
        assert result.exit_code == 47, result.output

        # Ensure our config is populated correctly
        assert config.mode == PipelineModes.NLP
        assert config.class_labels == ["score", "pred"]
        assert config.feature_length == 128

        assert config.ae is None

        pipe = callback_values['pipe']
        assert pipe is not None

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        [file_source, deserialize, process_nlp, triton_inf, monitor, add_class, validation, serialize, to_file] = stages

        assert isinstance(file_source, FileSourceStage)
        assert file_source._filename == os.path.join(TEST_DIRS.validation_data_dir, 'abp-validation-data.jsonlines')
        assert not file_source._iterative

        assert isinstance(deserialize, DeserializeStage)

        assert isinstance(process_nlp, PreprocessNLPStage)
        assert process_nlp._vocab_hash_file == vocab_file_name
        assert process_nlp._truncation
        assert process_nlp._do_lower_case
        assert not process_nlp._add_special_tokens

        assert isinstance(triton_inf, TritonInferenceStage)
        assert triton_inf._kwargs['model_name'] == 'test-model'
        assert triton_inf._kwargs['server_url'] == 'test:123'
        assert triton_inf._kwargs['force_convert_inputs']

        assert isinstance(monitor, MonitorStage)
        assert monitor._description == 'Unittest'
        assert monitor._smoothing == 0.001
        assert monitor._unit == 'inf'

        assert isinstance(add_class, AddClassificationsStage)
        assert add_class._labels == ['pred']
        assert add_class._threshold == 0.7

        assert isinstance(validation, ValidationStage)
        assert validation._val_file_name == os.path.join(TEST_DIRS.validation_data_dir,
                                                         'hammah-role-g-validation-data.csv')
        assert validation._results_file_name == 'results.json'
        assert validation._index_col == '_index_'

        # Click appears to be converting this into a tuple
        assert list(validation._exclude_columns) == ['event_dt']
        assert validation._rel_tol == 0.1

        assert isinstance(serialize, SerializeStage)

        assert isinstance(to_file, WriteToFileStage)
        assert to_file._output_file == 'out.csv'

    @pytest.mark.replace_callback('pipeline_nlp')
    def test_pipeline_nlp_all(self, config, callback_values, tmp_path, mlflow_uri):
        """
        Attempt to add all possible stages to the pipeline_nlp, even if the pipeline doesn't
        actually make sense, just test that cli could assemble it
        """
        tmp_model = os.path.join(tmp_path, 'fake-model.file')
        with open(tmp_model, 'w') as fh:
            pass

        labels_file = os.path.join(TEST_DIRS.data_dir, 'labels_phishing.txt')
        vocab_file_name = os.path.join(TEST_DIRS.data_dir, 'bert-base-uncased-hash.txt')
        args = GENERAL_ARGS + \
               ['pipeline-nlp', '--model_seq_length=128', '--labels_file', labels_file] + \
               FILE_SRC_ARGS + FROM_KAFKA_ARGS +  \
               ['deserialize', 'filter',
                'dropna', '--column', 'xyz',
                'preprocess', '--vocab_hash_file', vocab_file_name, '--truncation=True',
                '--do_lower_case=True', '--add_special_tokens=False',
                'add-scores', 'inf-identity',
                'inf-pytorch', '--model_filename', tmp_model,
                'mlflow-drift', '--tracking_uri', mlflow_uri] + \
               INF_TRITON_ARGS + MONITOR_ARGS + \
               ['add-class', '--label=pred', '--threshold=0.7'] + \
               VALIDATE_ARGS + ['serialize'] + TO_FILE_ARGS + TO_KAFKA_ARGS

        runner = CliRunner()
        result = runner.invoke(cli.cli, args)
        assert result.exit_code == 47, result.output

        # Ensure our config is populated correctly
        assert config.mode == PipelineModes.NLP
        assert config.class_labels == ["score", "pred"]
        assert config.feature_length == 128

        assert config.ae is None

        pipe = callback_values['pipe']
        assert pipe is not None

        stages = callback_values['stages']
        # Verify the stages are as we expect them, if there is a size-mismatch python will raise a Value error
        [
            file_source,
            from_kafka,
            deserialize,
            filter_stage,
            dropna,
            process_nlp,
            add_scores,
            inf_ident,
            inf_pytorch,
            mlflow_drift,
            triton_inf,
            monitor,
            add_class,
            validation,
            serialize,
            to_file,
            to_kafka
        ] = stages

        assert isinstance(file_source, FileSourceStage)
        assert file_source._filename == os.path.join(TEST_DIRS.validation_data_dir, 'abp-validation-data.jsonlines')
        assert not file_source._iterative

        assert isinstance(from_kafka, KafkaSourceStage)
        assert from_kafka._consumer_conf['bootstrap.servers'] == 'kserv1:123,kserv2:321'
        assert from_kafka._input_topic == 'test_topic'

        assert isinstance(deserialize, DeserializeStage)
        assert isinstance(filter_stage, FilterDetectionsStage)

        assert isinstance(dropna, DropNullStage)
        assert dropna._column == 'xyz'

        assert isinstance(process_nlp, PreprocessNLPStage)
        assert process_nlp._vocab_hash_file == vocab_file_name
        assert process_nlp._truncation
        assert process_nlp._do_lower_case
        assert not process_nlp._add_special_tokens

        assert isinstance(add_scores, AddScoresStage)
        assert isinstance(inf_ident, IdentityInferenceStage)

        assert isinstance(inf_pytorch, PyTorchInferenceStage)
        assert inf_pytorch._model_filename == tmp_model

        assert isinstance(mlflow_drift, MLFlowDriftStage)
        assert mlflow_drift._tracking_uri == mlflow_uri

        assert isinstance(triton_inf, TritonInferenceStage)
        assert triton_inf._kwargs['model_name'] == 'test-model'
        assert triton_inf._kwargs['server_url'] == 'test:123'
        assert triton_inf._kwargs['force_convert_inputs']

        assert isinstance(monitor, MonitorStage)
        assert monitor._description == 'Unittest'
        assert monitor._smoothing == 0.001
        assert monitor._unit == 'inf'

        assert isinstance(add_class, AddClassificationsStage)
        assert add_class._labels == ['pred']
        assert add_class._threshold == 0.7

        assert isinstance(validation, ValidationStage)
        assert validation._val_file_name == os.path.join(TEST_DIRS.validation_data_dir,
                                                         'hammah-role-g-validation-data.csv')
        assert validation._results_file_name == 'results.json'
        assert validation._index_col == '_index_'

        # Click appears to be converting this into a tuple
        assert list(validation._exclude_columns) == ['event_dt']
        assert validation._rel_tol == 0.1

        assert isinstance(serialize, SerializeStage)
        assert isinstance(to_file, WriteToFileStage)
        assert to_file._output_file == 'out.csv'

        assert isinstance(to_kafka, WriteToKafkaStage)
        assert to_kafka._kafka_conf['bootstrap.servers'] == 'kserv1:123,kserv2:321'
        assert to_kafka._output_topic == 'test_topic'

    @pytest.mark.replace_callback('pipeline_nlp')
    def test_pipeline_alias(self, config, callback_values, tmp_path):
        """
        Verify that pipeline implies pipeline-nlp
        """
        args = GENERAL_ARGS + ['pipeline'] + FILE_SRC_ARGS + TO_FILE_ARGS

        runner = CliRunner()
        result = runner.invoke(cli.cli, args)
        assert result.exit_code == 47, result.output

        # Ensure our config is populated correctly
        assert config.mode == PipelineModes.NLP
