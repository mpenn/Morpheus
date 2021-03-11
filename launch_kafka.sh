#!/bin/bash

# Download from https://netq-shared.s3-us-west-2.amazonaws.com/kafka-producer.tar.gz
# Load container with `docker load --input kafka-producer.tar.gz`

# Install docker-compose if not already installed
mamba install -c conda-forge docker-compose

# Create kafka service: https://medium.com/big-data-engineering/hello-kafka-world-the-complete-guide-to-kafka-with-docker-and-python-f788e2588cfc

# First change docker-compose.yml to use 'KAFKA_ADVERTISED_HOST_NAME: 172.17.0.1'

# Launch kafka
docker-compose up -d

# Scale to 3 instances
docker-compose scale kafka=3

# Create the topic
./start-kafka-shell.sh 172.17.0.1
$KAFKA_HOME/bin/kafka-topics.sh --create --topic=test_pcap --bootstrap-server `broker-list.sh`

# Run container
# docker run --rm -it -e KAFKA_BROKER_SERVERS=172.17.0.1:49161 -e INPUT_FILE_NAME=pcap_dump.json -e TOPIC_NAME=test_pcap --mount src="$PWD,target=/app/data/,type=bind" kafka-producer:latest
docker run --rm -it -e KAFKA_BROKER_SERVERS=$(kafka-docker/broker-list.sh) -e INPUT_FILE_NAME=pcap_dump.json -e TOPIC_NAME=test_pcap --mount src="$PWD,target=/app/data/,type=bind" kafka-producer:latest

# To view the messages from the server
./start-kafka-shell.sh 172.17.0.1
$KAFKA_HOME/bin/kafka-console-consumer.sh --topic=test_pcap --bootstrap-server `broker-list.sh`



# OTHER COMMANDS USED
# Launch Triton server
docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/home/mdemoret/Repos/rapids/cyber-dev/triton_models:/models nvcr.io/nvidia/tritonserver:21.02-py3 tritonserver --model-repository=/models --model-control-mode=poll --repository-poll-secs=1

# Run inference container
docker run --rm -ti --gpus=all -e CLX_INFERENCE_PIPELINE="pytorch" -e CLX_KAFKA_BOOTSTRAP_SERVERS=$(kafka-docker/broker-list.sh) -e CLX_MODEL_SEQ_LENGTH=512 498186410471.dkr.ecr.us-east-2.amazonaws.com/gtc-cyber-demo:latest

python -m grpc_tools.protoc -I=. --python_out=./proto_out --grpc_python_out=./proto_out request.proto
--include_source_info