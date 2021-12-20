ARG FROM_IMAGE=gpuci/miniconda-cuda
ARG CUDA_VER=11.4
ARG LINUX_VER=ubuntu18.04
FROM ${FROM_IMAGE}:${CUDA_VER}-runtime-${LINUX_VER} AS base

WORKDIR /mlflow

COPY . ./

RUN apt-get update && apt-get install -y procps vim
RUN /opt/conda/bin/conda env create -f environment/environment.yml
RUN sed -i 's/conda activate base/conda activate mlflow/g' ~/.bashrc

SHELL ["/opt/conda/bin/conda", "run", "-n", "mlflow", "/bin/bash", "-c"]

RUN source activate mlflow && \
    python setup.py install

RUN mkdir -p /mlflow/artifacts

ENV MLFLOW_MODEL_REPO=/mlflow/artifacts
ENV MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow-db.sqlite

# Set the entrypoint to use the entrypoint.sh script which sets the conda env
ENTRYPOINT [ "/opt/conda/bin/tini", "--", "docker/entrypoint.sh" ]

CMD [ "/bin/bash" ]
