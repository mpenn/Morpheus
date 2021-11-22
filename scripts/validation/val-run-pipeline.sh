#!/bin/bash

set -e +o pipefail
# set -x
# set -v

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Make sure to load the utils and globals
source ${SCRIPT_DIR}/val-globals.sh
source ${SCRIPT_DIR}/val-utils.sh

function run_pipeline_sid_minibert(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3
   VAL_FILE=$4
   VAL_OUTPUT=$5

   morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=32 --use_cpp=${USE_CPP} \
      pipeline-nlp --model_seq_length=256 \
      from-file --filename=${INPUT_FILE} \
      deserialize \
      preprocess --vocab_hash_file=${MORPHEUS_ROOT}/data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
      ${INFERENCE_STAGE} \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class --prefix="si_" \
      validate --val_file_name=${VAL_FILE} --results_file_name=${VAL_OUTPUT} --overwrite \
      serialize --exclude "id" --exclude "^_ts_" \
      to-file --filename=${OUTPUT_FILE} --overwrite
}

function run_pipeline_sid_bert(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3
   VAL_FILE=$4
   VAL_OUTPUT=$5

   morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=32 --use_cpp=${USE_CPP} \
      pipeline-nlp --model_seq_length=256 \
      from-file --filename=${INPUT_FILE} \
      deserialize \
      preprocess --vocab_hash_file=${MORPHEUS_ROOT}/data/bert-base-cased-hash.txt --truncation=True --do_lower_case=False --add_special_tokens=False \
      ${INFERENCE_STAGE} \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class --prefix="si_" \
      validate --val_file_name=${VAL_FILE} --results_file_name=${VAL_OUTPUT} --overwrite \
      serialize \
      to-file --filename=${OUTPUT_FILE} --overwrite
}

function run_pipeline_abp_nvsmi(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3
   VAL_FILE=$4
   VAL_OUTPUT=$5

   morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=1024 --use_cpp=${USE_CPP} \
      pipeline-fil \
      from-file --filename=${INPUT_FILE} \
      deserialize \
      preprocess \
      ${INFERENCE_STAGE} \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class \
      validate --val_file_name=${VAL_FILE} --results_file_name=${VAL_OUTPUT} --overwrite \
      serialize \
      to-file --filename=${OUTPUT_FILE} --overwrite
}

function run_pipeline_phishing_email(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3
   VAL_FILE=$4
   VAL_OUTPUT=$5

   morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=1024 --model_max_batch_size=32 --use_cpp=${USE_CPP} \
      pipeline-nlp --model_seq_length=128 --labels_file=${MORPHEUS_ROOT}/data/labels_phishing.txt \
      from-file --filename=${INPUT_FILE} \
      deserialize \
      preprocess --vocab_hash_file=${MORPHEUS_ROOT}/data/bert-base-uncased-hash.txt --truncation=True --do_lower_case=True --add_special_tokens=False \
      ${INFERENCE_STAGE} \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      add-class --label=pred --threshold=0.7 \
      validate --val_file_name=${VAL_FILE} --results_file_name=${VAL_OUTPUT} --overwrite \
      serialize \
      to-file --filename=${OUTPUT_FILE} --overwrite
}

function run_pipeline_hammah_user123(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3
   VAL_FILE=$4
   VAL_OUTPUT=$5

   morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=128 --model_max_batch_size=128 --use_cpp=${USE_CPP} \
      pipeline-ae --userid_filter=Account-123456789 \
      from-cloudtrail --input_glob="${MORPHEUS_ROOT}/models/datasets/validation-data/hammah-*.csv" \
      train-ae --train_data_glob="${MORPHEUS_ROOT}/models/datasets/training-data/hammah-*.csv" \
      preprocess \
      ${INFERENCE_STAGE} \
      add-scores \
      timeseries --resolution=10m \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      validate --val_file_name=${VAL_FILE} --results_file_name=${VAL_OUTPUT} --index_col="_index_" --exclude "ae_zscore" --overwrite \
      serialize \
      to-file --filename=${OUTPUT_FILE} --overwrite
}

function run_pipeline_hammah_role-g(){

   INPUT_FILE=$1
   INFERENCE_STAGE=$2
   OUTPUT_FILE=$3
   VAL_FILE=$4
   VAL_OUTPUT=$5

   morpheus --log_level=DEBUG run --num_threads=1 --pipeline_batch_size=128 --model_max_batch_size=128 --use_cpp=${USE_CPP} \
      pipeline-ae --userid_filter=Account-223344556 \
      from-cloudtrail --input_glob="${MORPHEUS_ROOT}/models/datasets/validation-data/hammah-*.csv" \
      train-ae --train_data_glob="${MORPHEUS_ROOT}/models/datasets/training-data/hammah-*.csv" \
      preprocess \
      ${INFERENCE_STAGE} \
      add-scores \
      timeseries --resolution=10m \
      monitor --description "Inference Rate" --smoothing=0.001 --unit inf \
      validate --val_file_name=${VAL_FILE} --results_file_name=${VAL_OUTPUT} --index_col="_index_" --exclude "ae_zscore" --exclude "event_dt" --overwrite \
      serialize \
      to-file --filename=${OUTPUT_FILE} --overwrite
}
