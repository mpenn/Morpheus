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

# Delete a topic
$KAFKA_HOME/bin/kafka-topics.sh --delete --topic=test_pcap --bootstrap-server `broker-list.sh`

# Run container
# docker run --rm -it -e KAFKA_BROKER_SERVERS=172.17.0.1:49161 -e INPUT_FILE_NAME=pcap_dump.json -e TOPIC_NAME=test_pcap --mount src="$PWD,target=/app/data/,type=bind" kafka-producer:latest
docker run --rm -it -e KAFKA_BROKER_SERVERS=$(kafka-docker/broker-list.sh) -e INPUT_FILE_NAME=pcap_dump.json -e TOPIC_NAME=test_pcap --mount src="$PWD,target=/app/data/,type=bind" kafka-producer:1

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

# Generate More Data (Run Once)
docker run --rm -it -e KAFKA_BROKER_SERVERS=$(kafka-docker/broker-list.sh) --mount src="$PWD/.tmp,target=/app/data/,type=bind" kafka-producer:latest python /app/main.py -m generate -n network -s 10 -p 0.1 -d 0.2 -r 10.20.0.0/16

# Actually produce data
docker run --rm -it -e KAFKA_BROKER_SERVERS=$(kafka-docker/broker-list.sh) --mount src="$PWD/.tmp,target=/app/data/,type=bind" kafka-producer:latest python /app/main.py -m produce -n network -i pcap_dump.json -j 10 -t test_pcap

# Delete all messages in topic
$KAFKA_HOME/bin/kafka-topics.sh --alter --topic=test_pcap --zookeeper 172.17.0.1:2181 --config retention.ms=1
$KAFKA_HOME/bin/kafka-topics.sh --alter --topic=test_pcap --zookeeper 172.17.0.1:2181 --config retention.ms=86400000

# Start Jupyter to run the graph preprocessing
docker run --gpus=all --rm -ti -p 8889:8888 -p 8787:8787 -p 8786:8786 -v $PWD:/rapids/notebooks/host rapidsai/rapidsai-nightly:cuda10.2-runtime-ubuntu18.04-py3.8 /bin/bash

# Start the viz container
docker-compose run --rm devel

# Run viz generation pipeline then run jupyter notebook noteboks/network_graph_viz_frames_clean.ipynb. Afterwards run:
sudo chown -R mdemoret:mdemoret noteboks/output/ && rm /home/mdemoret/Repos/rapids/node-rapids-dev/modules/demo/graph/data/network_graph_viz_frames_multi_label/* && cp -r noteboks/output/* /home/mdemoret/Repos/rapids/node-rapids-dev/modules/demo/graph/data/network_graph_viz_frames_multi_label/

# Then run the viz with (from the rapids-js container)
export FILE_COUNT=199
yarn demo modules/demo/graph --nodes=$(echo data/network_graph_viz_frames_multi_label/{0..${FILE_COUNT}}.0.nodes.csv | sed 's/ /,/g')\
 --edges=$(echo data/network_graph_viz_frames_multi_label/{0..${FILE_COUNT}}.0.edges.csv | sed 's/ /,/g')\
 --params='"autoCenter":1,"strongGravityMode":0,"jitterTolerance":0.01,"scalingRatio":1,"gravity":5,"controlsVisible":0,"outboundAttraction":1,"linLogMode":1' --delay=100 --width=1920 --height=1080

yarn demo modules/demo/graph --nodes=$(echo data/network_graph_viz_frames_multi_label/{0..159}.0.nodes.csv | sed 's/ /,/g')\
 --edges=$(echo data/network_graph_viz_frames_multi_label/{0..159}.0.edges.csv | sed 's/ /,/g')\
 --params='"autoCenter":1,"strongGravityMode":0,"jitterTolerance":0.01,"scalingRatio":1,"gravity":5,"controlsVisible":0,"outboundAttraction":1,"linLogMode":1' --delay=100 --width=1920 --height=1080

 --nodes=$(echo data/network_graph_viz_frames_multi_label_Bartley1/{0..199}.0.nodes.csv | sed 's/ /,/g') --edges=$(echo data/network_graph_viz_frames_multi_label_Bartley1/{0..199}.0.edges.csv | sed 's/ /,/g') --params='"autoCenter":1,"strongGravityMode":0,"jitterTolerance":0.01,"scalingRatio":1,"gravity":5,"controlsVisible":0,"outboundAttraction":1,"linLogMode":1' --delay=100 --width=1920 --height=1080