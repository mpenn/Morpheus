ARG FROM_IMAGE=gpuci/miniconda-cuda
ARG CUDA_VER=11.0
ARG LINUX_VER=ubuntu18.04
FROM ${FROM_IMAGE}:${CUDA_VER}-runtime-${LINUX_VER} AS base

# Required arguments
ARG IMAGE_TYPE=base
ARG RAPIDS_CHANNEL=rapidsai-nightly
ARG RAPIDS_VER=0.19
ARG PYTHON_VER=3.8

# Capture argument used for FROM
ARG CUDA_VER

# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

RUN /opt/conda/bin/conda create -n morpheus \
  -c rapidsai \
  -c nvidia \
  -c conda-forge \
  -c defaults \
  pip \
  cudatoolkit=${CUDA_VER} \
  cuml=${RAPIDS_VER} \
  cudf_kafka=${RAPIDS_VER} \
  custreamz=${RAPIDS_VER} \
  python=${PYTHON_VER} \
  tqdm \
  && sed -i 's/conda activate base/conda activate morpheus/g' ~/.bashrc

SHELL ["/opt/conda/bin/conda", "run", "-n", "morpheus", "/bin/bash", "-c"]

RUN pip install nvidia-pyindex \
  && pip install xgboost grpcio-channelz click configargparse docker tritonclient[all] typing-utils \
  && pip uninstall streamz \
  && pip install git+https://github.com/mdemoret-nv/streamz.git@async

WORKDIR /workspace

COPY . ./

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
