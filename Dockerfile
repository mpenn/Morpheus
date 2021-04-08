FROM rapidsai/rapidsai-clx-nightly:0.18-cuda11.0-runtime-ubuntu18.04-py3.7
LABEL maintainer="NVIDIA CORPORATION"

# SHELL ["/bin/bash", "-c"]

# Update conda to 4.8+
# RUN source activate base && conda update -y -n base -c defaults conda
RUN gpuci_conda_retry install -y -n base -c defaults \
    "conda=4.9.2"

# # From https://pythonspeed.com/articles/activate-conda-dockerfile/
# SHELL ["conda", "run", "-n", "rapids", "/bin/bash", "-c"]

# RUN conda --version

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -r -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

WORKDIR /workspace

RUN cd /workspace

RUN source activate rapids && pip install click configargparse docker

COPY . ./

USER trtuser

# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "rapids"]

CMD python cli.py