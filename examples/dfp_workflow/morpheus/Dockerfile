ARG MORPHEUS_CONTAINER=nvcr.io/nvidia/morpheus/morpheus
ARG MORPHEUS_CONTAINER_VERSION=v22.06.00-runtime

FROM ${MORPHEUS_CONTAINER}:${MORPHEUS_CONTAINER_VERSION} as common_base

# Fix the entrypoint to work with different WORKDIR
ENTRYPOINT [ "/opt/conda/bin/tini", "--", "/workspace/docker/entrypoint.sh" ]

SHELL ["/bin/bash", "-c"]
ENV UBA_COMMON="uba-autoencoder-common"

# Install vault
RUN apt-get update \
    && apt-get install -y \
      apt-utils \
      jq \
      lsb-release \
      software-properties-common \
    && curl -fsSL https://apt.releases.hashicorp.com/gpg | apt-key add - \
    && apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main" \
    && apt-get update \
    && apt-get install --reinstall -y  \
      vault \
    && apt-get clean all \
    && rm -rf /var/lib/apt/lists/*

# Fix vault install per: https://github.com/hashicorp/vault/issues/10924#issuecomment-1197259930
RUN setcap -r /usr/bin/vault

# # Install NGC CLI
# COPY ${UBA_COMMON}/utils/*.sh ./
# RUN chmod +x *.sh \
#   && bash ngc-cli-install.sh

# Install DFP dependencies
RUN source activate morpheus \
    && mamba install -y -c conda-forge \
        boto3 \
        dill \
        kfp \
        librdkafka \
        mlflow \
        papermill \
        s3fs

# COPY ${UBA_COMMON}/dfencoder-nv-updates/* \
#      /work/dfp/dfencoder-nv-updates/

# # Copy the common pipeline components to the morpheus_pipeline folder
# COPY ${UBA_COMMON}/morpheus_pipeline/* \
#      /work/dfp/morpheus_pipeline/

FROM common_base as base

WORKDIR /work/examples/dfp_workflow/morpheus

# This will get used by pipelines for the --s3_cache option
ENV DFP_S3_CACHE="/work/examples/dfp_workflow/morpheus/.s3_cache"

# Copy the sources
COPY . ./

CMD ["bash", "-c", "./launch.sh"]
