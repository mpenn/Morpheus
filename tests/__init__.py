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

import collections
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import unittest

import mlflow
import numpy as np
import requests
from mlflow.tracking import fluent

from morpheus.config import Config


class TestDirectories(object):
    def __init__(self, cur_file=__file__) -> None:
        self.tests_dir = os.path.dirname(cur_file)
        self.morpheus_root = os.environ.get('MORPHEUS_ROOT', os.path.dirname(self.tests_dir))
        self.data_dir = os.path.join(self.morpheus_root, 'data')
        self.models_dir = os.path.join(self.morpheus_root, 'models')
        self.datasets_dir = os.path.join(self.models_dir, 'datasets')
        self.training_data_dir = os.path.join(self.datasets_dir, 'training-data')
        self.validation_data_dir = os.path.join(self.datasets_dir, 'validation-data')
        self.expeced_data_dir = os.path.join(self.tests_dir, 'expected_data')
        self.mock_triton_servers_dir = os.path.join(self.tests_dir, 'mock_triton_server')

TEST_DIRS = TestDirectories()

class BaseMorpheusTest(unittest.TestCase):
    Results = collections.namedtuple('Results', ['total_rows', 'diff_rows', 'error_pct'])

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()
        # reset the config singleton
        Config.reset()

    def _mk_tmp_dir(self):
        """
        Creates a temporary directory for use by tests, directory is deleted after the test is run unless the
        MORPHEUS_NO_TEST_CLEANUP environment variable is defined.
        """
        tmp_dir = tempfile.mkdtemp(prefix='morpheus_test_')
        if os.environ.get('MORPHEUS_NO_TEST_CLEANUP') is None:
            self.addCleanup(shutil.rmtree, tmp_dir)

        return tmp_dir

    def _save_env_vars(self):
        """
        Save the current environment variables and restore them at the end of the test, removing any newly added values
        """
        orig_vars = os.environ.copy()
        self.addCleanup(self._restore_env_vars, orig_vars)

    def _restore_env_vars(self, orig_vars):
        # Iterating over a copy of the keys as we will potentially be deleting keys in the loop
        for key in list(os.environ.keys()):
            orig_val = orig_vars.get(key)
            if orig_val is not None:
                os.environ[key] = orig_val
            else:
                del (os.environ[key])

    def _calc_error_val(self, results_file):
        """
        Based on the calc_error_val function in val-utils.sh
        """
        with open(results_file) as fh:
            results = json.load(fh)

        total_rows = results['total_rows']
        diff_rows = results['diff_rows']
        return self.Results(total_rows=total_rows, diff_rows=diff_rows, error_pct=(diff_rows / total_rows) * 100)

    def _partition_array(self, array, chunk_size):
        return np.split(array, range(chunk_size, len(array), chunk_size))

    def _wait_for_camouflage(self, popen, root_dir, host="localhost", port=8000, timeout=5):
        ready = False
        elapsed_time = 0.0
        sleep_time = 0.1
        url = "http://{}:{}/ping".format(host, port)
        while not ready and elapsed_time < timeout and popen.poll() is None:
            try:
                r = requests.get(url, timeout=1)
                if r.status_code == 200:
                    ready = r.json()['message'] == 'I am alive.'
            except Exception as e:
                pass

            if not ready:
                time.sleep(sleep_time)
                elapsed_time += sleep_time

        if popen.poll() is not None:
            raise RuntimeError("camouflage server exited with status code={} details in: {}".\
                format(popen.poll(), os.path.join(root_dir, 'camouflage.log')))

        return ready

    def _kill_proc(self, proc, timeout=1):
        logging.info("killing pid {}".format(proc.pid))

        elapsed_time = 0.0
        sleep_time = 0.1
        stopped = False

        # It takes a little while to shutdown
        while not stopped and elapsed_time < timeout:
            proc.kill()
            stopped = (proc.poll() is not None)
            if not stopped:
                time.sleep(sleep_time)
                elapsed_time += sleep_time

    def _launch_camouflage_triton(self, root_dir=TEST_DIRS.mock_triton_servers_dir, config="config.yml", timeout=5):
        """
        Launches a mock triton server using camouflage (https://testinggospels.github.io/camouflage/) with a package
        rooted at `root_dir` and configured with `config`.

        This function will wait for up to `timeout` seconds for camoflauge to startup

        This function is a no-op if the `MORPHEUS_NO_LAUNCH_CAMOUFLAGE` environment variable is defined, which can
        be useful during test development to run camouflage by hand.
        """
        if os.environ.get('MORPHEUS_NO_LAUNCH_CAMOUFLAGE') is None:
            popen = subprocess.Popen(["camouflage", "--config", config],
                                     cwd=root_dir,
                                     stderr=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL)

            logging.info("Launching camouflage in {} with pid: {}".format(root_dir, popen.pid))
            self.addCleanup(self._kill_proc, popen)

            if timeout > 0:
                if not self._wait_for_camouflage(popen, root_dir, timeout=timeout):
                    raise RuntimeError("Failed to launch camouflage server")

    def _shutdown_mlflow(self):
        """
        Shutdown all of the active mlflow runs prior to the tmp_dir being deleted
        """
        num_runs = len(fluent._active_run_stack)
        for _ in range(num_runs):
            mlflow.end_run()

    def _get_mlflow_uri(self, experiment_name="Morpheus"):
        tmp_dir = self._mk_tmp_dir()
        uri = "file://{}".format(tmp_dir)
        mlflow.set_tracking_uri(uri)
        mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        with mlflow.start_run(run_name="Model Drift",
                              tags={"morpheus.type": "drift"},
                              experiment_id=experiment.experiment_id):
            pass

        self.addCleanup(self._shutdown_mlflow)
        return uri