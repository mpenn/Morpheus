import os
import json
import time
import cudf
import numpy as np
from pathlib import Path
from cuml import ForestInference

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])
        
        process_name_config = pb_utils.get_output_config_by_name(model_config, "process_name")
        hostname_config = pb_utils.get_output_config_by_name(model_config, "hostname")
        timestamp_config = pb_utils.get_output_config_by_name(model_config, "timestamp")
        preds_config = pb_utils.get_output_config_by_name(model_config, "preds")
        
        self.process_name_config = pb_utils.triton_string_to_numpy(process_name_config['data_type'])
        self.hostname_config = pb_utils.triton_string_to_numpy(hostname_config['data_type'])
        self.timestamp_config = pb_utils.triton_string_to_numpy(timestamp_config['data_type'])
        self.preds_config = pb_utils.triton_string_to_numpy(preds_config['data_type'])
        
        
        # Model features
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
        "nvidia_smi_log.gpu.max_customer_boost_clocks.graphics_clock",]
        
        
        # Get outputconfiguration
        #output_config = pb_utils.get_output_config_by_name(model_config, "output")
        
        model_filepath = '/models/anomaly_detection_fil_model/1/xgboost.model'
        
        self.model = ForestInference.load(model_filepath, output_class=True)
        

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        
        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests: 
            s_time = time.time()
            
            # Get raw_logs
            raw_logs = pb_utils.get_input_tensor_by_name(request, "raw_logs")
            raw_logs = raw_logs.as_numpy()
            raw_logs = [raw_log[0].decode("utf-8") for raw_log in raw_logs]
            
            events_df = cudf.DataFrame()
            events_df['raw_event'] = raw_logs
        
            for feature in self.features:
                regex = '"{}":[\s"].(\d+)'.format(feature)
                events_df[feature] = events_df["raw_event"].str.extract(regex, expand=False).astype("int64")
            process_name_series = events_df["raw_event"].str.extract(
                '"nvidia_smi_log.gpu.processes.process_info.*.process_name": "([a-zA-Z0-9_.-\s]+)"', expand=False
            )
            hostname_series = events_df["raw_event"].str.extract(
                'hostname":\s"([a-zA-Z0-9\-\.\@\s\_]+)"', expand=False
            )
            timestamp_series = events_df["raw_event"].str.extract('timestamp":\s([\.0-9]+)', expand=False)
            re_time = time.time()
            
            events_df = events_df.drop("raw_event", axis=1)
            
            # predict model
            events_df["predictions"] = self.model.predict(events_df)
            preds_arr = events_df["predictions"].to_array()
            process_name_arr = process_name_series.to_array()
            hostname_arr = hostname_series.to_array()
            timestamp_arr = timestamp_series.to_array()
            
            process_name_tensor = pb_utils.Tensor("process_name", process_name_arr.astype(self.process_name_config))
            hostname_tensor = pb_utils.Tensor("hostname", hostname_arr.astype(self.hostname_config))
            timestamp_tensor = pb_utils.Tensor("timestamp", timestamp_arr.astype(self.timestamp_config))
            preds_tensor = pb_utils.Tensor("preds", preds_arr.astype(self.preds_config))
            
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[process_name_tensor, hostname_tensor, timestamp_tensor, preds_tensor])
            responses.append(inference_response)
            e_time = time.time()
            print('Processing batch time: {}'.format(e_time-s_time))
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
