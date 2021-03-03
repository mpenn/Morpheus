# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Copyright 2021 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# %% [markdown]
# <img src="https://upload.wikimedia.org/wikipedia/en/6/6d/Nvidia_image_logo.svg" style="width: 90px; float: right;">
# 
# # QA Inference on BERT using TensorRT
# %% [markdown]
# ## 1. Overview
# 
# Bidirectional Embedding Representations from Transformers (BERT), is a method of pre-training language representations which obtains state-of-the-art results on a wide array of Natural Language Processing (NLP) tasks. 
# 
# The original paper can be found here: https://arxiv.org/abs/1810.04805.
# 
# %% [markdown]
# ### 1.a Learning objectives
# 
# This notebook demonstrates:
# - Inference on Question Answering (QA) task with BERT Base/Large model
# - The use fine-tuned NVIDIA BERT models
# - Use of BERT model with TRT
# %% [markdown]
# ## 2. Requirements
# 
# Please refer to the ReadMe file
# %% [markdown]
# ## 3. BERT Inference: Question Answering
# 
# We can run inference on a fine-tuned BERT model for tasks like Question Answering.
# 
# Here we use a BERT model fine-tuned on a [SQuaD 2.0 Dataset](https://rajpurkar.github.io/SQuAD-explorer/) which contains 100,000+ question-answer pairs on 500+ articles combined with over 50,000 new, unanswerable questions.
# %% [markdown]
# 
# 
# ### 3.a Paragraph and Queries
# 
# The paragraph and the questions can be customized by changing the text below. Note that when using models with small sequence lengths, you should use a shorter paragraph:
# %% [markdown]
# #### Paragraph:

# %%
paragraph_text = "The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which accomplished landing the first humans on the Moon from 1969 to 1972. First conceived during Dwight D. Eisenhower's administration as a three-man spacecraft to follow the one-man Project Mercury which put the first Americans in space, Apollo was later dedicated to President John F. Kennedy's national goal of landing a man on the Moon and returning him safely to the Earth by the end of the 1960s, which he proposed in a May 25, 1961, address to Congress. Project Mercury was followed by the two-man Project Gemini. The first manned flight of Apollo was in 1968. Apollo ran from 1961 to 1972, and was supported by the two-man Gemini program which ran concurrently with it from 1962 to 1966. Gemini missions developed some of the space travel techniques that were necessary for the success of the Apollo missions. Apollo used Saturn family rockets as launch vehicles. Apollo/Saturn vehicles were also used for an Apollo Applications Program, which consisted of Skylab, a space station that supported three manned missions in 1973-74, and the Apollo-Soyuz Test Project, a joint Earth orbit mission with the Soviet Union in 1975."

# Short paragraph version for BERT models with max sequence length of 128
short_paragraph_text = "The Apollo program was the third United States human spaceflight program. First conceived as a three-man spacecraft to follow the one-man Project Mercury which put the first Americans in space, Apollo was dedicated to President John F. Kennedy's national goal of landing a man on the Moon. The first manned flight of Apollo was in 1968. Apollo ran from 1961 to 1972 followed by the Apollo-Soyuz Test Project a joint Earth orbit mission with the Soviet Union in 1975."

# %% [markdown]
# #### Question:

# %%
question_text = "What project put the first Americans into space?"
#question_text =  "What year did the first manned Apollo flight occur?"
#question_text =  "What President is credited with the original notion of putting Americans in space?"
#question_text =  "Who did the U.S. collaborate with on an Earth orbit mission in 1975?"

# %% [markdown]
# In this example we ask our BERT model questions related to the following paragraph:
# 
# **The Apollo Program**
# _"The Apollo program, also known as Project Apollo, was the third United States human spaceflight program carried out by the National Aeronautics and Space Administration (NASA), which accomplished landing the first humans on the Moon from 1969 to 1972. First conceived during Dwight D. Eisenhower's administration as a three-man spacecraft to follow the one-man Project Mercury which put the first Americans in space, Apollo was later dedicated to President John F. Kennedy's national goal of landing a man on the Moon and returning him safely to the Earth by the end of the 1960s, which he proposed in a May 25, 1961, address to Congress. Project Mercury was followed by the two-man Project Gemini. The first manned flight of Apollo was in 1968. Apollo ran from 1961 to 1972, and was supported by the two-man Gemini program which ran concurrently with it from 1962 to 1966. Gemini missions developed some of the space travel techniques that were necessary for the success of the Apollo missions. Apollo used Saturn family rockets as launch vehicles. Apollo/Saturn vehicles were also used for an Apollo Applications Program, which consisted of Skylab, a space station that supported three manned missions in 1973-74, and the Apollo-Soyuz Test Project, a joint Earth orbit mission with the Soviet Union in 1975."_
# 
# The questions and relative answers expected are shown below:
# 
#  - **Q1:** "What project put the first Americans into space?" 
#   - **A1:** "Project Mercury"
#  - **Q2:** "What program was created to carry out these projects and missions?"
#   - **A2:** "The Apollo program"
#  - **Q3:** "What year did the first manned Apollo flight occur?"
#   - **A3:** "1968"
#  - **Q4:** "What President is credited with the original notion of putting Americans in space?"
#   - **A4:** "John F. Kennedy"
#  - **Q5:** "Who did the U.S. collaborate with on an Earth orbit mission in 1975?"
#   - **A5:** "Soviet Union"
#  - **Q6:** "How long did Project Apollo run?"
#   - **A6:** "1961 to 1972"
#  - **Q7:** "What program helped develop space travel techniques that Project Apollo used?"
#   - **A7:** "Gemini Mission"
#  - **Q8:** "What space station supported three manned missions in 1973-1974?"
#   - **A8:** "Skylab"
# %% [markdown]
# ## Data Preprocessing utils
# Let's convert the paragraph and the question to BERT input with the help of the tokenizer:

# %%
# # The maximum number of tokens for the question. Questions longer than this will be truncated to this length.
max_query_length = 64

# # When splitting up a long document into chunks, how much stride to take between chunks.
doc_stride = 128

# # The maximum total input sequence length after WordPiece tokenization. 
# # Sequences longer than this will be truncated, and sequences shorter 
max_seq_length = 128

# %% [markdown]
# ## TensorRT Inference (Using cudf for text processing)

# %%
import sys

print(sys.executable)

# import cudf.utils.hash_vocab_utils

# cudf.utils.hash_vocab_utils.hash_vocab("/home/mdemoret/Repos/github/NVIDIA/TensorRT/demo/BERT/models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1/vocab.txt", "./vocab_hash.txt")


# %%
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


# %%
import ctypes
import os

ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)


# %%
from cudf_subword_helper import create_vocab_table, tokenize_text_series
import cudf
import pycuda.driver as cuda

import pycuda.autoinit
import collections
import numpy as np
import time
from numba import cuda as numba_cuda

pyc_dev = pycuda.autoinit.device
pyc_ctx = pyc_dev.retain_primary_context()
pyc_ctx.push()
# Load the BERT-Large Engine
with open("/home/mdemoret/Repos/github/NVIDIA/TensorRT/demo/BERT/engines/bert_large_128-b1.engine", "rb") as f,     trt.Runtime(TRT_LOGGER) as runtime,     runtime.deserialize_cuda_engine(f.read()) as engine,     engine.create_execution_context() as context:

     # We always use batch size 1.
    input_shape = (max_seq_length, 1)
    input_nbytes = trt.volume(input_shape) * trt.int32.itemsize  
    
    # Allocate device memory for inputs.
    d_inputs = [cuda.mem_alloc(input_nbytes) for binding in range(3)]
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    # Specify input shapes. These must be within the min/max bounds of the active profile (0th profile in this case)
    # Note that input shapes can be specified on a per-inference basis, but in this case, we only have a single shape.
    for binding in range(3):
        context.set_binding_shape(binding, input_shape)
    assert context.all_binding_shapes_specified
    

    # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
    h_output = cuda.pagelocked_empty(tuple(context.get_binding_shape(3)), dtype=np.float32)
    d_output = cuda.mem_alloc(h_output.nbytes)

    print("\nRunning Inference...")    
    _NetworkOutput = collections.namedtuple(  # pylint: disable=invalid-name
        "NetworkOutput",
        ["start_logits", "end_logits", "feature_index"])
    networkOutputs = []
    
    n_iterations=10
    
    preproc_time = 0
    memtrans_time = 0
    compute_time = 0
    postproc_time = 0
    eval_time_elapsed = 0

    s = cudf.Series([1,2,3])
    print(s)

    start_time = time.time()
    old_epoch = time.time()

    for feature_index in range(0,n_iterations):

        ### cudf Processing
        paragraph_ser = cudf.Series(short_paragraph_text)
        question_ser = cudf.Series(question_text)
        text =  question_ser + ' [SEP] '+ paragraph_ser

        text_series = cudf.Series(short_paragraph_text+' [SEP] '+ question_text)

        feature = tokenize_text_series(text,128,1,'vocab_hash.txt')

        new_epoch = time.time()
        preproc_time += new_epoch - old_epoch
        old_epoch = new_epoch

        ### Go via numpy array
        input_ids = cuda.register_host_memory(np.ascontiguousarray(feature.input_ids.ravel().get()))
        segment_ids = cuda.register_host_memory(np.ascontiguousarray(feature.segment_ids.ravel().get()))
        input_mask = cuda.register_host_memory(np.ascontiguousarray(feature.input_mask.ravel().get()))

        new_epoch = time.time()
        memtrans_time += new_epoch - old_epoch
        old_epoch = new_epoch

        cuda.memcpy_htod_async(d_inputs[0], input_ids, stream)
        cuda.memcpy_htod_async(d_inputs[1], segment_ids, stream)
        cuda.memcpy_htod_async(d_inputs[2], input_mask, stream)

        # Run inference
        context.execute_async_v2(bindings=[int(d_inp) for d_inp in d_inputs] + [int(d_output)], stream_handle=stream.handle)
        # Synchronize the stream
        stream.synchronize()
        
        new_epoch = time.time()
        compute_time += new_epoch - old_epoch
        old_epoch = new_epoch

        # Transfer predictions back from GPU
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

        for index, batch in enumerate(h_output):
            # Data Post-processing
            networkOutputs.append(_NetworkOutput(
                start_logits = np.array(batch.squeeze()[:, 0]),
                end_logits = np.array(batch.squeeze()[:, 1]),
                feature_index = feature_index
                ))

        new_epoch = time.time()
        postproc_time += new_epoch - old_epoch
        old_epoch = new_epoch

    total_time = time.time() - start_time

    eval_time_elapsed /= n_iterations
    
    print("-----------------------------")
    print("Running Inference at {:.3f} Sentences/Sec".format(1.0/eval_time_elapsed))
    print("------------------")
    
    ### cudf initalize
    s = cudf.Series([1,2,3])
    print(s)

pyc_ctx.pop()

# %% [markdown]
# ## Data Post-Processing
# %% [markdown]
# Now that we have the inference results let's extract the actual answer to our question

# %%
import helpers.data_processing as dp
import helpers.tokenization as tokenization
vocab_file = "models/fine-tuned/bert_tf_ckpt_large_qa_squad2_amp_128_v19.03.1/vocab.txt"


tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)



# # Extract tokens from the paragraph
doc_tokens = dp.convert_doc_tokens(short_paragraph_text)

# # Extract features from the paragraph and question
features = dp.convert_example_to_features(doc_tokens, question_text, tokenizer, max_seq_length, doc_stride, max_query_length)

# The total number of n-best predictions to generate in the nbest_predictions.json output file
n_best_size = 20

# The maximum length of an answer that can be generated. This is needed 
#  because the start and end predictions are not conditioned on one another
max_answer_length = 30

prediction, nbest_json, scores_diff_json = dp.get_predictions(doc_tokens, features,[networkOutputs[0]], n_best_size, max_answer_length)

# for index, output in enumerate(networkOutputs):
print("Processing output")
print("Answer: '{}'".format(prediction))
print("with prob: {:.3f}%".format(nbest_json[0]['probability'] * 100.0))


