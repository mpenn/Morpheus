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

#include <morpheus/stages/triton_inference.hpp>

#include <morpheus/messages/multi_response_probs.hpp>
#include <morpheus/objects/triton_in_out.hpp>
#include <morpheus/utilities/matx_util.hpp>
#include <morpheus/utilities/stage_util.hpp>
#include <morpheus/utilities/type_util.hpp>

#include <neo/core/segment_object.hpp>
#include <pyneo/node.hpp>

#include <glog/logging.h>
#include <http_client.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <sstream>
#include <utility>


#define CHECK_TRITON(method) ::InferenceClientStage__check_triton_errors(method, #method, __FILE__, __LINE__);

namespace {
    // Component-private free functions.
    void InferenceClientStage__check_triton_errors(triton::client::Error status,
                             const std::string &methodName,
                             const std::string &filename,
                             const int &lineNumber) {
        if (!status.IsOk()) {
            std::string err_msg =
                    CONCAT_STR("Triton Error while executing '" << methodName << "'. Error: " + status.Message() << "\n"
                                                                << filename << "(" << lineNumber << ")");
            LOG(ERROR) << err_msg;
            throw std::runtime_error(err_msg);
        }
    }
}

namespace morpheus {
    // Component public implementations
    // ************ InferenceClientStage ************************* //
    InferenceClientStage::InferenceClientStage(const neo::Segment &parent, const std::string &name,
                                               std::string model_name, std::string server_url,
                                               bool force_convert_inputs, bool use_shared_memory, bool needs_logits,
                                               std::map<std::string, std::string> inout_mapping)  :
            neo::SegmentObject(parent, name),
            PythonNode(parent, name, build_operator()),
            m_model_name(std::move(model_name)),
            m_server_url(std::move(server_url)),
            m_force_convert_inputs(force_convert_inputs),
            m_use_shared_memory(use_shared_memory),
            m_needs_logits(needs_logits),
            m_inout_mapping(std::move(inout_mapping)),
            m_options(m_model_name) {
        // Connect with the server to setup the inputs/outputs
        this->connect_with_server(); // TODO(Devin)
    }

    InferenceClientStage::operator_fn_t InferenceClientStage::build_operator()  {
        return [this](neo::Observable <reader_type_t> &input, neo::Subscriber <writer_type_t> &output) {
            std::unique_ptr<triton::client::InferenceServerHttpClient> client;

            CHECK_TRITON(triton::client::InferenceServerHttpClient::Create(&client, m_server_url, false));

            return input.subscribe(neo::make_observer<reader_type_t>(
                    [this, &output, &client](reader_type_t &&x) {
                        auto reponse_memory = std::make_shared<ResponseMemory>(x->count);

                        // Create the output memory blocks
                        for (auto &model_output: m_model_outputs) {
                            auto total_shape = model_output.shape;

                            // First dimension will always end up being the number of rows
                            total_shape[0] = x->count;

                            auto elem_count =
                                    std::accumulate(total_shape.begin(), total_shape.end(), 1, std::multiplies<>());

                            // Create the output memory
                            auto output_buffer = std::make_shared<rmm::device_buffer>(
                                    elem_count * model_output.datatype.item_size(), rmm::cuda_stream_per_thread);

                            reponse_memory->outputs[model_output.mapped_name] =
                                    Tensor::create(std::move(output_buffer),
                                                   model_output.datatype,
                                                   std::vector<neo::TensorIndex>{static_cast<int>(total_shape[0]),
                                                                                 static_cast<int>(total_shape[1])},
                                                   std::vector<neo::TensorIndex>{},
                                                   0);
                        }

                        // This will be the final output of all mini-batches
                        auto response = std::make_shared<MultiResponseProbsMessage>(
                                x->meta, x->mess_offset, x->mess_count, std::move(reponse_memory), 0,
                                reponse_memory->count);

                        for (size_t i = 0; i < x->count; i += m_max_batch_size) {
                            triton::client::InferInput *input1;

                            size_t start = i;
                            size_t stop = std::min(i + m_max_batch_size, x->count);

                            reader_type_t mini_batch_input =
                                    std::static_pointer_cast<MultiInferenceMessage>(x->get_slice(start, stop));
                            writer_type_t mini_batch_output =
                                    std::static_pointer_cast<MultiResponseProbsMessage>(
                                            response->get_slice(start, stop));

                            // Iterate on the model inputs in case the model takes less than what tensors are available
                            std::vector<std::pair<std::shared_ptr<triton::client::InferInput>, std::vector<uint8_t>>> saved_inputs =
                                    foreach_map(m_model_inputs, [this, &mini_batch_input](auto const &model_input) {
                                        DCHECK(mini_batch_input->memory->has_input(model_input.mapped_name))
                                                        << "Model input '" << model_input.mapped_name
                                                        << "' not found in InferenceMemory";

                                        auto const &inp_tensor = mini_batch_input->get_input(
                                                model_input.mapped_name);

                                        // Convert to the right type. Make shallow if necessary
                                        auto final_tensor = inp_tensor.as_type(model_input.datatype);

                                        std::vector<uint8_t> inp_data = final_tensor.get_host_data();

                                        // Test
                                        triton::client::InferInput *inp_ptr;

                                        triton::client::InferInput::Create(&inp_ptr,
                                                                           model_input.name,
                                                                           {inp_tensor.shape(0),
                                                                            inp_tensor.shape(1)},
                                                                           model_input.datatype.triton_str());
                                        std::shared_ptr<triton::client::InferInput> inp_shared;
                                        inp_shared.reset(inp_ptr);

                                        inp_ptr->AppendRaw(inp_data);

                                        return std::make_pair(inp_shared, std::move(inp_data));
                                    });

                            std::vector<std::shared_ptr<const triton::client::InferRequestedOutput>> saved_outputs =
                                    foreach_map(m_model_outputs, [this](auto const &model_output) {
                                        // Generate the outputs to be requested.
                                        triton::client::InferRequestedOutput *out_ptr;

                                        triton::client::InferRequestedOutput::Create(&out_ptr, model_output.name);
                                        std::shared_ptr<const triton::client::InferRequestedOutput> out_shared;
                                        out_shared.reset(out_ptr);

                                        return out_shared;
                                    });

                            std::vector<triton::client::InferInput *> inputs =
                                    foreach_map(saved_inputs, [](auto x) { return x.first.get(); });

                            std::vector<const triton::client::InferRequestedOutput *> outputs =
                                    foreach_map(saved_outputs, [](auto x) { return x.get(); });

                            // this->segment().resources().fiber_pool().enqueue([client, output](){});

                            triton::client::InferResult *results;

                            CHECK_TRITON(client->Infer(&results, m_options, inputs, outputs));

                            for (auto &model_output: m_model_outputs) {
                                std::vector<int64_t> output_shape;

                                CHECK_TRITON(results->Shape(model_output.name, &output_shape));

                                // Make sure we have at least 2 dims
                                while (output_shape.size() < 2) {
                                    output_shape.push_back(1);
                                }

                                const uint8_t *output_ptr = nullptr;
                                size_t output_ptr_size = 0;
                                CHECK_TRITON(results->RawData(model_output.name, &output_ptr, &output_ptr_size));

                                auto output_buffer =
                                        std::make_shared<rmm::device_buffer>(output_ptr_size,
                                                                             rmm::cuda_stream_per_thread);

                                NEO_CHECK_CUDA(
                                        cudaMemcpy(output_buffer->data(), output_ptr, output_ptr_size,
                                                   cudaMemcpyHostToDevice));

                                // If we need to do logits, do that here
                                if (m_needs_logits) {
                                    size_t element_count =
                                            std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                                            std::multiplies<>());
                                    output_buffer = MatxUtil::logits(
                                            DevMemInfo{element_count, model_output.datatype.type_id(),
                                                       output_buffer, 0});
                                }

                                mini_batch_output->set_output(
                                        model_output.mapped_name,
                                        Tensor::create(std::move(output_buffer),
                                                       model_output.datatype,
                                                       std::vector<neo::TensorIndex>{
                                                               static_cast<int>(output_shape[0]),
                                                               static_cast<int>(output_shape[1])},
                                                       std::vector<neo::TensorIndex>{},
                                                       0));
                            }
                        }
                        output.on_next(std::move(response));
                    },
                    [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                    [&]() { output.on_completed(); }));
        };
    }

    void InferenceClientStage::connect_with_server()  {
        std::string server_url = m_server_url;

        std::unique_ptr<triton::client::InferenceServerHttpClient> client;

        auto result = triton::client::InferenceServerHttpClient::Create(&client, server_url, false);

        // Now load the input/outputs for the model
        bool is_server_live = false;

        triton::client::Error status = client->IsServerLive(&is_server_live);

        if (!status.IsOk()) {
            if (this->is_default_grpc_port(server_url)) {
                LOG(WARNING) << "Failed to connect to Triton at '" << m_server_url
                             << "'. Default gRPC port of (8001) was detected but C++ "
                                "InferenceClientStage uses HTTP protocol. Retrying with default HTTP port (8000)";

                // We are using the default gRPC port, try the default HTTP
                std::unique_ptr<triton::client::InferenceServerHttpClient> unique_client;

                auto result = triton::client::InferenceServerHttpClient::Create(&unique_client, server_url, false);

                client = std::move(unique_client);

                status = client->IsServerLive(&is_server_live);
            } else if (status.Message().find("Unsupported protocol") != std::string::npos) {
                throw std::runtime_error(
                        CONCAT_STR("Failed to connect to Triton at '"
                                           << m_server_url
                                           << "'. Received 'Unsupported Protocol' error. Are you using the right port? The C++ "
                                              "InferenceClientStage uses Triton's HTTP protocol instead of gRPC. Ensure you have "
                                              "specified the HTTP port (Default 8000)."));
            }

            if (!status.IsOk())
                throw std::runtime_error(CONCAT_STR("Unable to connect to Triton at '"
                                                            << m_server_url
                                                            << "'. Check the URL and port and ensure the server is running."));
        }

        // Save this for new clients
        m_server_url = server_url;

        if (!is_server_live)
            throw std::runtime_error("Server is not live");

        bool is_server_ready = false;
        CHECK_TRITON(client->IsServerReady(&is_server_ready));

        if (!is_server_ready)
            throw std::runtime_error("Server is not ready");

        bool is_model_ready = false;
        CHECK_TRITON(client->IsModelReady(&is_model_ready, this->m_model_name));

        if (!is_model_ready)
            throw std::runtime_error("Model is not ready");

        std::string model_metadata_json;
        CHECK_TRITON(client->ModelMetadata(&model_metadata_json, this->m_model_name));

        auto model_metadata = nlohmann::json::parse(model_metadata_json);

        std::string model_config_json;
        CHECK_TRITON(client->ModelConfig(&model_config_json, this->m_model_name));

        auto model_config = nlohmann::json::parse(model_config_json);

        if (model_config.contains("max_batch_size")) {
            m_max_batch_size = model_config.at("max_batch_size").get<int>();
        }

        for (auto const &input: model_metadata.at("inputs")) {
            auto shape = input.at("shape").get<std::vector<int>>();

            auto dtype = DType::from_triton(input.at("datatype").get<std::string>());

            size_t bytes = dtype.item_size();

            for (auto &y: shape) {
                if (y == -1) {
                    y = m_max_batch_size;
                }

                bytes *= y;
            }

            std::string mapped_name = input.at("name").get<std::string>();

            if (m_inout_mapping.find(mapped_name) != m_inout_mapping.end()) {
                mapped_name = m_inout_mapping[mapped_name];
            }

            m_model_inputs.push_back(TritonInOut{input.at("name").get<std::string>(),
                                                 bytes,
                                                 DType::from_triton(input.at("datatype").get<std::string>()),
                                                 shape,
                                                 mapped_name,
                                                 0});
        }

        for (auto const &output: model_metadata.at("outputs")) {
            auto shape = output.at("shape").get<std::vector<int>>();

            auto dtype = DType::from_triton(output.at("datatype").get<std::string>());

            size_t bytes = dtype.item_size();

            for (auto &y: shape) {
                if (y == -1) {
                    y = m_max_batch_size;
                }

                bytes *= y;
            }

            std::string mapped_name = output.at("name").get<std::string>();

            if (m_inout_mapping.find(mapped_name) != m_inout_mapping.end()) {
                mapped_name = m_inout_mapping[mapped_name];
            }

            m_model_outputs.push_back(
                    TritonInOut{output.at("name").get<std::string>(), bytes, dtype, shape, mapped_name, 0});
        }
    }

    bool InferenceClientStage::is_default_grpc_port(std::string &server_url)  {
        // Check if we are the default gRPC port of 8001 and try 8000 for http client instead
        size_t colon_loc = server_url.find_last_of(':');

        if (colon_loc == -1) {
            return false;
        }

        // Check if the port matches 8001
        if (server_url.size() < colon_loc + 1 || server_url.substr(colon_loc + 1) != "8001") {
            return false;
        }

        // It matches, change to 8000
        server_url = server_url.substr(0, colon_loc) + ":8000";

        return true;
    }

    // ************ InferenceClientStageInterfaceProxy********* //
    std::shared_ptr<InferenceClientStage>
    InferenceClientStageInterfaceProxy::init(neo::Segment &parent, const std::string &name, std::string model_name,
                                             std::string server_url, bool force_convert_inputs, bool use_shared_memory,
                                             bool needs_logits, std::map<std::string, std::string> inout_mapping)  {
        auto stage = std::make_shared<InferenceClientStage>(parent,
                                                            name,
                                                            model_name,
                                                            server_url,
                                                            force_convert_inputs,
                                                            use_shared_memory,
                                                            needs_logits,
                                                            inout_mapping);

        parent.register_node<InferenceClientStage>(stage);

        return stage;
    }
}