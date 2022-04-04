/**
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <morpheus/stages/preprocess_nlp.hpp>

#include <morpheus/messages/multi_inference.hpp>
#include <morpheus/utilities/type_util.hpp>

#include <pyneo/node.hpp>
#include <neo/core/segment.hpp>

#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <librdkafka/rdkafkacpp.h>
#include <nvtext/subword_tokenize.hpp>

#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <utility>


namespace morpheus {
    // Component public implementations
    // ************ PreprocessNLPStage ************************* //
    PreprocessNLPStage::PreprocessNLPStage(const neo::Segment &parent, const std::string &name,
                                           std::string vocab_hash_file, uint32_t sequence_length, bool truncation,
                                           bool do_lower_case, bool add_special_token, int stride)  :
            neo::SegmentObject(parent, name),
            PythonNode(parent, name, build_operator()),
            m_vocab_hash_file(std::move(vocab_hash_file)),
            m_sequence_length(sequence_length),
            m_truncation(truncation),
            m_do_lower_case(do_lower_case),
            m_add_special_token(add_special_token),
            m_stride(stride) {}

    PreprocessNLPStage::operator_fn_t PreprocessNLPStage::build_operator()  {
        return [this](neo::Observable <reader_type_t> &input, neo::Subscriber <writer_type_t> &output) {
            uint32_t stride = m_stride;

            // Auto calc stride to be 75% of sequence length
            if (stride < 0) {
                stride = m_sequence_length / 2;
                stride = stride + stride / 2;
            }

            return input.subscribe(neo::make_observer<reader_type_t>(
                    [this, stride, &output](reader_type_t &&x) {
                        // Convert to string view
                        auto string_col = cudf::strings_column_view{x->get_meta("data").get_column(0)};

                        // Create the hashed vocab
                        thread_local std::unique_ptr<nvtext::hashed_vocabulary> vocab =
                                nvtext::load_vocabulary_file(this->m_vocab_hash_file);

                        // Perform the tokenizer
                        auto token_results = nvtext::subword_tokenize(string_col,
                                                                      *vocab,
                                                                      this->m_sequence_length,
                                                                      stride,
                                                                      this->m_do_lower_case,
                                                                      this->m_truncation,
                                                                      string_col.size() * 2);

                        // Build the results
                        auto memory = std::make_shared<InferenceMemory>(token_results.nrows_tensor);

                        int32_t length = token_results.tensor_token_ids->size() / token_results.sequence_length;
                        auto input_ids_released =
                                cudf::cast(token_results.tensor_token_ids->view(),
                                           cudf::data_type(cudf::type_id::INT32))
                                        ->release();

                        memory->inputs["input_ids"] = std::move(Tensor::create(
                                std::move(input_ids_released.data),
                                DType::create<int32_t>(),
                                std::vector<neo::TensorIndex>{length,
                                                              static_cast<int>(token_results.sequence_length)},
                                std::vector<neo::TensorIndex>{},
                                0));

                        length = token_results.tensor_attention_mask->size() / token_results.sequence_length;
                        auto input_mask_released =
                                cudf::cast(token_results.tensor_attention_mask->view(),
                                           cudf::data_type(cudf::type_id::INT32))
                                        ->release();
                        memory->inputs["input_mask"] = std::move(Tensor::create(
                                std::move(input_mask_released.data),
                                DType::create<int32_t>(),
                                std::vector<neo::TensorIndex>{length,
                                                              static_cast<int>(token_results.sequence_length)},
                                std::vector<neo::TensorIndex>{},
                                0));

                        length = token_results.tensor_metadata->size() / 3;
                        auto seq_ids_released =
                                cudf::cast(token_results.tensor_metadata->view(),
                                           cudf::data_type(cudf::type_id::INT32))
                                        ->release();
                        memory->inputs["seq_ids"] =
                                std::move(Tensor::create(std::move(seq_ids_released.data),
                                                         DType::create<int32_t>(),
                                                         std::vector<neo::TensorIndex>{length,
                                                                                       static_cast<int32_t>(3)},
                                                         std::vector<neo::TensorIndex>{},
                                                         0));

                        auto next = std::make_shared<MultiInferenceMessage>(
                                x->meta, x->mess_offset, x->mess_count, std::move(memory), 0, memory->count);

                        output.on_next(std::move(next));
                    },
                    [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                    [&]() { output.on_completed(); }));
        };
    }

    // ************ PreprocessNLPStageInterfaceProxy *********** //
    std::shared_ptr<PreprocessNLPStage>
    PreprocessNLPStageInterfaceProxy::init(neo::Segment &parent, const std::string &name, std::string vocab_hash_file,
                                           uint32_t sequence_length, bool truncation, bool do_lower_case,
                                           bool add_special_token, int stride)  {
        auto stage = std::make_shared<PreprocessNLPStage>(parent,
                                                          name,
                                                          vocab_hash_file,
                                                          sequence_length,
                                                          truncation,
                                                          do_lower_case,
                                                          add_special_token,
                                                          stride);

        parent.register_node<PreprocessNLPStage>(stage);

        return stage;
    }
}