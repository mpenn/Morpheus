# MLFlow Triton

MLFlow plugin for deploying your models from MLFlow to Triton Inference Server. Scripts
are included for publishing TensorRT, ONNX and FIL models to your MLFlow Model Registry.

## Requirements

* MLflow (tested on 1.21.0)
* Python (tested on 3.8)

## Install Triton Docker Image

Before you can use the Triton Docker image you must install
[Docker](https://docs.docker.com/engine/install). If you plan on using
a GPU for inference you must also install the [NVIDIA Container
Toolkit](https://github.com/NVIDIA/nvidia-docker). DGX users should
follow [Preparing to use NVIDIA
Containers](http://docs.nvidia.com/deeplearning/dgx/preparing-containers/index.html).

Pull the image using the following command.

```
$ docker pull nvcr.io/nvidia/tritonserver:<xx.yy>-py3
```

Where \<xx.yy\> is the version of Triton that you want to pull.

## Set up your Triton Model Repository
Create a directory on your host machine that will serve as your Triton model repository. This directory will contain the models to be used by Morpheus and will be volume mounted to your Triton Inference Server container.

Example:

```
mkdir -p /opt/triton_models
```

## Start Triton Inference Server in EXPLICIT mode

Use the following command to run Triton with our model
repository you just created. The [NVIDIA Container
Toolkit](https://github.com/NVIDIA/nvidia-docker) must be installed
for Docker to recognize the GPU(s). The --gpus=1 flag indicates that 1
system GPU should be made available to Triton for inferencing.

```
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v /opt/triton_models:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models --model-control-mode=explicit
```

## MLflow container

Build MLFlow image from Dockerfile:

```
docker build -t mlflow-morpheus:latest -f docker/Dockerfile .
```

Create MLFlow container with volume mount to Triton model repository:

```
docker run -it -v /opt/triton_models:/triton_models \
--env TRITON_MODEL_REPO=/triton_models \
--gpus '"device=0"' \
--net=host \
--rm \
-d mlflow-morpheus:latest
```

Open Bash shell in container:

```
docker exec -it <container_name> bash
```

## Start MLflow server

```
nohup mlflow server --backend-store-uri sqlite:////tmp/mlflow-db.sqlite --default-artifact-root /mlflow/artifacts --host 0.0.0.0 &
```


## Download Morpheus reference models

You can download the Morpheus reference models by cloning the [morpheus-models](https://gitlab-master.nvidia.com/morpheus/morpheus-models) GitLab repo.

```
git clone https://gitlab-master.nvidia.com/morpheus/morpheus-models.git
```

## Publish reference models to MLflow

The `publish_model_to_mlflow` script is used to publish `onnx`, `tensorrt`, and `fil` models to MLflow.

```
cd /mlflow/scripts

python publish_model_to_mlflow.py \
  	--model_name sid-bert-onnx \
  	--model_file <path-to-morpheus-models-repo>/sid-minibert-onnx/1/model.onnx \
  	--model_config <path-to-morpheus-models-repo>/sid-minibert-onnx/config.pbtxt \
    --flavor onnx  

python publish_model_to_mlflow.py \
  	--model_name sid-minibert-trt \
  	--model_file <path-to-morpheus-models-repo>/sid-minibert-trt/1/model.plan \
  	--model_config <path-to-morpheus-models-repo>/sid-minibert-trt/config.pbtxt \
    --flavor tensorrt

python publish_model_to_mlflow.py \
  	--model_name abp-nvsmi-xgb \
  	--model_file <path-to-morpheus-models-repo>/abp-nvsmi-xgb/1/abp-nvsmi-xgb.bst \
  	--model_config <path-to-morpheus-models-repo>/abp-nvsmi-xgb/config.pbtxt \
    --flavor fil
```

## Deploy reference models to Triton

```
mlflow deployments create -t triton --flavor onnx --name ref_model_1 -m models:/ref_model_1/1 -C "version=1"

mlflow deployments create -t triton --flavor onnx --name ref_model_2 -m models:/ref_model_2/1 -C "version=1"
```

## Deployments

The following deployment functions are implemented within the plugin.
The plugin will deploy associated `config.pbtxt` with the saved model version.

### Create Deployment

To create a deployment use the following command

##### CLI
```
mlflow deployments create -t triton --flavor onnx --name mini_bert_onnx -m models:/mini_bert_onnx/1 -C "version=1"
```

##### Python API
```
from mlflow.deployments import get_deploy_client
client = get_deploy_client('triton')
client.create_deployment("mini_bert_onnx", "models:/mini_bert_onnx/1", flavor="onnx", config={"version": "1"})
```

### Delete Deployment

##### CLI
```
mlflow deployments delete -t triton --name mini_bert_onnx/1 
```

##### Python API
```
client.delete_deployment("mini_bert_onnx/1")
```

### Update Deployment

##### CLI
```
mlflow deployments update -t triton --flavor onnx --name mini_bert_onnx/1 -m models:/mini_bert_onnx/1
```

##### Python API
```
client.update_deployment("mini_bert_onnx/1", "models:/mini_bert_onnx/2", flavor="onnx")
```

### List Deployments

##### CLI
```
mlflow deployments list -t triton
```

##### Python API
```
client.list_deployments()
```

### Get Deployment

##### CLI
```
mlflow deployments get -t triton --name mini_bert_onnx
```

##### Python API
```
client.get_deployment("mini_bert_onnx")
```
