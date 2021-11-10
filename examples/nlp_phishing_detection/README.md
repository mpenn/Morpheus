<!--
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# Phishing Detection Examples Using Morpheus

Example Morpheus pipeline using Docker containers for Triton Inference server and Morpheus SDK/Client.

### Set up Triton Inference Server

##### Pull Triton Inference Server Docker Image
Pull Docker image from NGC (https://ngc.nvidia.com/catalog/containers/nvidia:tritonserver) suitable for your environment.

Example:

```
docker pull nvcr.io/nvidia/tritonserver:21.09-py3
```

##### Create your Triton Model Repository
Create a directory on your host machine that will serve as your Triton model repository. This directory will contain your models and be volume mounted to your Triton Inference Server container.
Your repository must follow the layout described here: https://github.com/triton-inference-server/server/blob/r21.03/docs/model_repository.md

An example for an URL phishing detection ONNX model and it's configuration provided at `~/morpheus/models/triton-model-repo/url-phishing-bert-onnx`. In this case, you would copy your ONNX model and configuration to `<model_repository_path>/`

An example for an Email phishing detection ONNX model and it's configuration provided at `~/morpheus/models/triton-model-repo/phishing-bert-onnx`. In this case, you would copy your ONNX model and configuration to `<model_repository_path>/`

You can find more information on Triton model configuration here: `https://github.com/triton-inference-server/server/blob/r21.03/docs/model_configuration.md`


##### Start Triton Inference Server container
```
docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v <model_repository_path>:/models nvcr.io/nvidia/tritonserver:21.09-py3 tritonserver --model-repository=/models
```

### Run Phishing Detection Pipelines

##### Build Morpheus SDK/Client Docker image
```
docker build -t morpheus-sdk:latest -f ops/Dockerfile .
```

##### Start Morpheus SDK/Client container
```
docker run -it --gpus '"device=0"' \
--rm -d \
--net=host \
morpheus-sdk:latest
```

##### Start interactive bash shell in your container
```
docker exec -it <container_name> bash
```

##### Install Morpheus python module

```
python setup.py install
```

##### Run URL phishing detection pipeline on your input data
The following command can be used to start the URL phishing detection pipeline. Your input data must be in JSON lines, with a file name `.jsonlines` extension.

Here's an example input schema for a JSON line.

```
{"data": "<url>"}
```

```
python ./examples/nlp_phishing_detection/binary/run.py \
    --num_threads 1 \
    --input_file <input json file path> \
    --output_file <output json file path>  \
    --model_vocab_hash_file=./data/bert-base-uncased-hash.txt \
    --model_seq_length=128 \
    --model_name url-phishing-bert-onnx \
    --server_url localhost:8001
```

##### Run Email phishing detection pipeline on your input data
The following command can be used to start the Email phishing detection pipeline. Your input data must be in JSON lines, with a file name `.jsonlines` extension.

Here's an example input schema for a JSON line.

```
{"data": "<email_body>"}
```

```
python ./examples/nlp_phishing_detection/binary/run.py \
    --num_threads 1 \
    --input_file <input json file path> \
    --output_file <output json file path>  \
    --model_vocab_hash_file=./data/bert-base-uncased-hash.txt \
    --model_seq_length=128 \
    --model_name phishing-bert-onnx \
    --server_url localhost:8001
```
##### Convert ONNX to TRT

Please use Morpheus tool as stated here: `/morpheus/models/triton-model-repo/url-phishing-bert-trt/1/README.md` to convert `url-phishing-bert-onnx` model to TensorRT for improved performance. Similarly we can follow the same procedure for `phishing-bert-onnx` as stated here: `/morpheus/models/triton-model-repo/phishing-bert-trt/1/README.md`

Note: Before deploying the TRT model to the Triton inference server, check that the model tree topology as shown below.

```
$ tree url-phishing-bert-trt/
```
output:

```
url-phishing-bert-trt/
├── 1
│   └── model.engine
└── config.pbtxt
```

