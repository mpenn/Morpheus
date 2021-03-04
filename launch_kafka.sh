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

# Run container
docker run --rm -it -e KAFKA_BROKER_SERVERS=172.17.0.1:49161 -e INPUT_FILE_NAME=pcap_dump.json -e TOPIC_NAME=test_pcap --mount src="$PWD,target=/app/data/,type=bind" kafka-producer:latest
docker run --rm -it -e KAFKA_BROKER_SERVERS=$(kafka-docker/broker-list.sh) -e INPUT_FILE_NAME=pcap_dump.json -e TOPIC_NAME=test_pcap --mount src="$PWD,target=/app/data/,type=bind" kafka-producer:latest

# To view the messages from the server
./start-kafka-shell.sh 172.17.0.1
$KAFKA_HOME/bin/kafka-console-consumer.sh --topic=test_pcap --bootstrap-server `broker-list.sh`