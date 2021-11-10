#!/bin/bash

set -e +o pipefail
# set -x
# set -v

function run_pipeline_nlp_minibert(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3

   morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=32 pipeline --model_seq_length=256 \
      from-file --filename=${INPUT_FILE} \
      deserialize \
      preprocess --vocab_hash_file=${MORPHEUS_ROOT}/data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
      ${INFERENCE_STAGE} \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class \
      serialize \
      to-file --filename=${OUTPUT_FILE} --overwrite
}

function run_pipeline_nlp_bert(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3

   morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=32 pipeline --model_seq_length=256 \
      from-file --filename=${INPUT_FILE} \
      deserialize \
      preprocess --vocab_hash_file=${MORPHEUS_ROOT}/data/bert-base-cased-hash.txt --truncation=True --do_lower_case=False --add_special_tokens=False \
      ${INFERENCE_STAGE} \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class \
      serialize \
      to-file --filename=${OUTPUT_FILE} --overwrite
}

function run_pipeline_abp_nvsmi(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3

   morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=1024 pipeline-fil \
      from-file --filename=${INPUT_FILE} \
      deserialize \
      preprocess \
      ${INFERENCE_STAGE} \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class \
      serialize \
      to-file --filename=${OUTPUT_FILE} --overwrite
}
