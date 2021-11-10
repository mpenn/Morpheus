#pragma once

#include "morpheus/common.hpp"
#include "morpheus/messages.hpp"

#include <http_client.h>
#include <cudf/io/csv.hpp>
#include <cudf/io/json.hpp>
#include <cudf/unary.hpp>
#include <nlohmann/json.hpp>
#include <nvtext/subword_tokenize.hpp>
#include "pyneo/node.hpp"
#include "trtlab/neo/core/segment_object.hpp"

namespace morpheus {

namespace fs    = std::filesystem;
namespace neo   = trtlab::neo;
namespace py    = pybind11;
namespace pyneo = trtlab::neo::pyneo;
namespace tc    = triton::client;
using json      = nlohmann::json;

class FileSourceStage : public pyneo::PythonSource<std::shared_ptr<MultiMessage>>
{
  public:
    using pyneo::PythonSource<std::shared_ptr<MultiMessage>>::source_type_t;

    FileSourceStage(
        const neo::Segment& parent, const std::string& name, std::string filename, int32_t batch_size, int repeat = 1) :
      neo::SegmentObject(parent, name),
      pyneo::PythonSource<std::shared_ptr<MultiMessage>>(parent, name),
      m_filename(std::move(filename)),
      m_batch_size(batch_size),
      m_repeat(repeat)
    {
        this->set_source_observable(
            neo::Observable<source_type_t>([this](neo::Subscriber<std::shared_ptr<MultiMessage>>& sub) {
                auto data_table = this->load_table();

                // Next, create the message metadata
                auto meta = MessageMeta::create_from_cpp(std::move(data_table), std::vector<std::string>());

                for (cudf::size_type repeat_idx = 0; repeat_idx < m_repeat; ++repeat_idx)
                {
                    // Now, for each batch, create a new MultiMessage
                    for (cudf::size_type i = 0; i < meta->count(); i += m_batch_size)
                    {
                        // End index is non-inclusive. i.e. [0, 1024)
                        cudf::size_type end_idx = std::min(i + m_batch_size, meta->count());

                        auto next = std::make_shared<MultiMessage>(meta, i, end_idx - i);

                        sub.on_next(std::move(next));
                    }
                }

                sub.on_completed();
            }));
    }

  private:
    cudf::io::table_with_metadata load_table()
    {
        auto file_path = fs::path(m_filename);

        if (file_path.extension() == ".json" || file_path.extension() == ".jsonlines")
        {
            // First, load the file into json
            auto options = cudf::io::json_reader_options::builder(cudf::io::source_info{m_filename}).lines(true);

            return cudf::io::read_json(options.build());
        }
        else if (file_path.extension() == ".csv")
        {
            auto options = cudf::io::csv_reader_options::builder(cudf::io::source_info{m_filename})
                               .dtypes({cudf::data_type(cudf::type_id::FLOAT32)});

            return cudf::io::read_csv(options.build());
        }
        else
        {
            LOG(FATAL) << "Unknown extension for file: " << m_filename;
            throw std::runtime_error("Unknown extension");
        }
    }

    std::string m_filename;
    int m_repeat{1};
    int32_t m_batch_size{1024};
};

class PreprocessNLPStage
  : public pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>
{
  public:
    using base_t = pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>;
    using base_t::operator_fn_t;
    using base_t::reader_type_t;
    using base_t::writer_type_t;

    PreprocessNLPStage(const neo::Segment& parent, const std::string& name, std::string vocab_file) :
      neo::SegmentObject(parent, name),
      PythonNode(parent, name, build_operator()),
      m_vocab_file(std::move(vocab_file))
    {}

  private:
    operator_fn_t build_operator()
    {
        return [this](neo::Observable<reader_type_t>& input, neo::Subscriber<writer_type_t>& output) {
            return input.subscribe(neo::make_observer<reader_type_t>(
                [this, &output](reader_type_t&& x) {
                    // Convert to string view
                    auto string_col = cudf::strings_column_view{x->get_meta("data")};

                    // Create the hashed vocab
                    thread_local std::unique_ptr<nvtext::hashed_vocabulary> vocab =
                        nvtext::load_vocabulary_file(this->m_vocab_file);

                    // Perform the tokenizer
                    auto token_results =
                        nvtext::subword_tokenize(string_col, *vocab, 256, 192, false, true, string_col.size() * 2);

                    // Build the results
                    auto memory = std::make_shared<morpheus::InferenceMemory>(token_results.nrows_tensor);

                    int32_t length = token_results.tensor_token_ids->size() / token_results.sequence_length;
                    auto input_ids_released =
                        cudf::cast(token_results.tensor_token_ids->view(), cudf::data_type(cudf::type_id::INT32))
                            ->release();

                    memory->inputs["input_ids"] = std::move(Tensor::create(
                        std::move(input_ids_released.data),
                        "<i4",
                        std::vector<neo::TensorIndex>{length, static_cast<int>(token_results.sequence_length)},
                        std::vector<neo::TensorIndex>{},
                        0));

                    length = token_results.tensor_attention_mask->size() / token_results.sequence_length;
                    auto input_mask_released =
                        cudf::cast(token_results.tensor_attention_mask->view(), cudf::data_type(cudf::type_id::INT32))
                            ->release();
                    memory->inputs["input_mask"] = std::move(Tensor::create(
                        std::move(input_mask_released.data),
                        "<i4",
                        std::vector<neo::TensorIndex>{length, static_cast<int>(token_results.sequence_length)},
                        std::vector<neo::TensorIndex>{},
                        0));

                    length = token_results.tensor_metadata->size() / 3;
                    auto seq_ids_released =
                        cudf::cast(token_results.tensor_metadata->view(), cudf::data_type(cudf::type_id::INT32))
                            ->release();
                    memory->inputs["seq_ids"] =
                        std::move(Tensor::create(std::move(seq_ids_released.data),
                                                 "<i4",
                                                 std::vector<neo::TensorIndex>{length, static_cast<int32_t>(3)},
                                                 std::vector<neo::TensorIndex>{},
                                                 0));

                    auto next = std::make_shared<morpheus::MultiInferenceMessage>(
                        x->meta, x->mess_offset, x->mess_count, std::move(memory), 0, memory->count);

                    output.on_next(std::move(next));
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }

    std::string m_vocab_file;
};

class PreprocessFILStage
  : public pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>
{
  public:
    using base_t = pyneo::PythonNode<std::shared_ptr<MultiMessage>, std::shared_ptr<MultiInferenceMessage>>;
    using base_t::operator_fn_t;
    using base_t::reader_type_t;
    using base_t::writer_type_t;

    PreprocessFILStage(const neo::Segment& parent, const std::string& name) :
      neo::SegmentObject(parent, name),
      PythonNode(parent, name, build_operator())
    {}

  private:
    operator_fn_t build_operator()
    {
        return [this](neo::Observable<reader_type_t>& input, neo::Subscriber<writer_type_t>& output) {
            return input.subscribe(neo::make_observer<reader_type_t>(
                [this, &output](reader_type_t&& x) {
                    std::vector<std::string> fea_cols = {
                        "nvidia_smi_log.gpu.pci.tx_util",
                        "nvidia_smi_log.gpu.pci.rx_util",
                        "nvidia_smi_log.gpu.fb_memory_usage.used",
                        "nvidia_smi_log.gpu.fb_memory_usage.free",
                        "nvidia_smi_log.gpu.bar1_memory_usage.total",
                        "nvidia_smi_log.gpu.bar1_memory_usage.used",
                        "nvidia_smi_log.gpu.bar1_memory_usage.free",
                        "nvidia_smi_log.gpu.utilization.gpu_util",
                        "nvidia_smi_log.gpu.utilization.memory_util",
                        "nvidia_smi_log.gpu.temperature.gpu_temp",
                        "nvidia_smi_log.gpu.temperature.gpu_temp_max_threshold",
                        "nvidia_smi_log.gpu.temperature.gpu_temp_slow_threshold",
                        "nvidia_smi_log.gpu.temperature.gpu_temp_max_gpu_threshold",
                        "nvidia_smi_log.gpu.temperature.memory_temp",
                        "nvidia_smi_log.gpu.temperature.gpu_temp_max_mem_threshold",
                        "nvidia_smi_log.gpu.power_readings.power_draw",
                        "nvidia_smi_log.gpu.clocks.graphics_clock",
                        "nvidia_smi_log.gpu.clocks.sm_clock",
                        "nvidia_smi_log.gpu.clocks.mem_clock",
                        "nvidia_smi_log.gpu.clocks.video_clock",
                        "nvidia_smi_log.gpu.applications_clocks.graphics_clock",
                        "nvidia_smi_log.gpu.applications_clocks.mem_clock",
                        "nvidia_smi_log.gpu.default_applications_clocks.graphics_clock",
                        "nvidia_smi_log.gpu.default_applications_clocks.mem_clock",
                        "nvidia_smi_log.gpu.max_clocks.graphics_clock",
                        "nvidia_smi_log.gpu.max_clocks.sm_clock",
                        "nvidia_smi_log.gpu.max_clocks.mem_clock",
                        "nvidia_smi_log.gpu.max_clocks.video_clock",
                        "nvidia_smi_log.gpu.max_customer_boost_clocks.graphics_clock",
                    };

                    auto df_just_features = x->get_meta(fea_cols);

                    auto packed_data = std::make_shared<rmm::device_buffer>(
                        fea_cols.size() * x->mess_count * sizeof(float), rmm::cuda_stream_per_thread);

                    for (size_t i = 0; i < df_just_features.num_columns(); ++i)
                    {
                        cudaMemcpy(static_cast<float*>(packed_data->data()) + i * df_just_features.num_rows(),
                                   df_just_features.column(i).data<float>(),
                                   df_just_features.num_rows() * sizeof(float),
                                   cudaMemcpyDeviceToDevice);
                    }

                    // Build the results
                    auto memory = std::make_shared<morpheus::InferenceMemory>(x->mess_count);

                    memory->inputs["input__0"] =
                        Tensor::create(packed_data,
                                       "<f4",
                                       std::vector<neo::TensorIndex>{x->mess_count, static_cast<int>(fea_cols.size())},
                                       std::vector<neo::TensorIndex>{},
                                       0);

                    auto next = std::make_shared<morpheus::MultiInferenceMessage>(
                        x->meta, x->mess_offset, x->mess_count, std::move(memory), 0, memory->count);

                    output.on_next(std::move(next));
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }

    std::string m_vocab_file;
};

template <typename FuncT, typename SeqT>
auto foreach_map(SeqT seq, FuncT func)
{
    using value_t  = typename SeqT::value_type;
    using return_t = decltype(func(std::declval<value_t>()));

    std::vector<return_t> result{};

    std::transform(seq.begin(), seq.end(), std::back_inserter(result), func);

    return result;
}

void __checkTritonErrors(tc::Error status,
                         const std::string& methodName,
                         const std::string& filename,
                         const int& lineNumber)
{
    if (!status.IsOk())
    {
        std::string err_msg =
            CONCAT_STR("Triton Error while executing '" << methodName << "'. Error: " + status.Message() << "\n"
                                                        << filename << "(" << lineNumber << ")");
        LOG(ERROR) << err_msg;
        throw std::runtime_error(err_msg);
    }
}

#define CHECK_TRITON(method) __checkTritonErrors(method, #method, __FILE__, __LINE__);

class InferenceClientStage
  : public pyneo::PythonNode<std::shared_ptr<MultiInferenceMessage>, std::shared_ptr<MultiResponseMessage>>
{
  public:
    using base_t = pyneo::PythonNode<std::shared_ptr<MultiInferenceMessage>, std::shared_ptr<MultiResponseMessage>>;
    using base_t::operator_fn_t;
    using base_t::reader_type_t;
    using base_t::writer_type_t;

    InferenceClientStage(const neo::Segment& parent,
                         const std::string& name,
                         std::string model_name,
                         std::string server_url,
                         std::map<std::string, std::string> inout_mapping = std::map<std::string, std::string>()) :
      neo::SegmentObject(parent, name),
      PythonNode(parent, name, build_operator()),
      m_model_name(std::move(model_name)),
      m_server_url(std::move(server_url)),
      m_inout_mapping(std::move(inout_mapping))
    {}

  private:
    struct TritonInOut
    {
        std::string name;
        size_t bytes;
        std::string datatype;
        std::vector<int> shape;
        std::string mapped_name;
        size_t offset;
    };

    operator_fn_t build_operator()
    {
        return [this](neo::Observable<reader_type_t>& input, neo::Subscriber<writer_type_t>& output) {
            std::shared_ptr<tc::InferenceServerHttpClient> client;

            {
                std::unique_ptr<tc::InferenceServerHttpClient> unique_client;

                auto result = tc::InferenceServerHttpClient::Create(&unique_client, m_server_url, false);

                client = std::move(unique_client);
            }

            tc::InferOptions options(m_model_name);

            // Now load the input/outputs for the model
            bool is_server_live = false;
            CHECK_TRITON(client->IsServerLive(&is_server_live));

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

            auto model_metadata = json::parse(model_metadata_json);

            std::string model_config_json;
            CHECK_TRITON(client->ModelConfig(&model_config_json, this->m_model_name));

            auto model_config = json::parse(model_config_json);

            int max_batch_size = -1;

            if (model_config.contains("max_batch_size"))
            {
                max_batch_size = model_config.at("max_batch_size").get<int>();
            }

            std::vector<TritonInOut> model_inputs;

            for (auto const& input : model_metadata.at("inputs"))
            {
                auto shape = input.at("shape").get<std::vector<int>>();

                size_t bytes = 4;  // TODO(MDD): Should get from size of datatype

                for (auto& y : shape)
                {
                    if (y == -1)
                    {
                        y = max_batch_size;
                    }

                    bytes *= y;
                }

                std::string mapped_name = input.at("name").get<std::string>();

                if (m_inout_mapping.find(mapped_name) != m_inout_mapping.end())
                {
                    mapped_name = m_inout_mapping[mapped_name];
                }

                model_inputs.push_back(TritonInOut{input.at("name").get<std::string>(),
                                                   bytes,
                                                   input.at("datatype").get<std::string>(),
                                                   shape,
                                                   mapped_name,
                                                   0});
            }

            std::vector<std::string> model_output_names;

            for (auto const& input : model_metadata.at("outputs"))
            {
                model_output_names.push_back(input.at("name").get<std::string>());
            }

            return input.subscribe(neo::make_observer<reader_type_t>(
                [this, &output, client, &options, model_inputs, model_output_names, max_batch_size](reader_type_t&& x) {
                    auto reponse_memory = std::make_shared<morpheus::ResponseMemory>(x->count);

                    // This will be the final output of all mini-batches
                    auto response = std::make_shared<morpheus::MultiResponseMessage>(
                        x->meta, x->mess_offset, x->mess_count, std::move(reponse_memory), 0, reponse_memory->count);

                    for (size_t i = 0; i < x->count; i += max_batch_size)
                    {
                        tc::InferInput* input1;

                        reader_type_t mini_batch_input =
                            std::static_pointer_cast<MultiInferenceMessage>(x->get_slice(i, i + max_batch_size));
                        writer_type_t mini_batch_output =
                            std::static_pointer_cast<MultiResponseMessage>(response->get_slice(i, i + max_batch_size));

                        // Iterate on the model inputs in case the model takes less than what tensors are available
                        std::vector<std::shared_ptr<tc::InferInput>> saved_inputs =
                            foreach_map(model_inputs, [this, &mini_batch_input](auto const& model_input) {
                                DCHECK(mini_batch_input->memory->has_input(model_input.mapped_name))
                                    << "Model input '" << model_input.mapped_name << "' not found in InferenceMemory";

                                auto const& inp_tensor = mini_batch_input->get_input(model_input.mapped_name);

                                std::vector<uint8_t> inp_data = inp_tensor.get_host_data();

                                tc::InferInput* inp_ptr;

                                tc::InferInput::Create(&inp_ptr,
                                                       model_input.name,
                                                       {inp_tensor.shape(0), inp_tensor.shape(1)},
                                                       model_input.datatype);
                                std::shared_ptr<tc::InferInput> inp_shared;
                                inp_shared.reset(inp_ptr);

                                inp_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&inp_data[0]), inp_data.size());

                                return inp_shared;
                            });

                        std::vector<std::shared_ptr<const tc::InferRequestedOutput>> saved_outputs =
                            foreach_map(model_output_names, [this](auto const& out_name) {
                                // Generate the outputs to be requested.
                                tc::InferRequestedOutput* out_ptr;

                                tc::InferRequestedOutput::Create(&out_ptr, out_name);
                                std::shared_ptr<const tc::InferRequestedOutput> out_shared;
                                out_shared.reset(out_ptr);

                                return out_shared;
                            });

                        std::vector<tc::InferInput*> inputs = foreach_map(saved_inputs, [](auto x) { return x.get(); });

                        std::vector<const tc::InferRequestedOutput*> outputs =
                            foreach_map(saved_outputs, [](auto x) { return x.get(); });

                        // this->segment().resources().fiber_pool().enqueue([client, output](){});

                        tc::InferResult* results;

                        CHECK_TRITON(client->Infer(&results, options, inputs, outputs));

                        for (auto& output_name : model_output_names)
                        {
                            std::vector<int64_t> output_shape;

                            CHECK_TRITON(results->Shape(output_name, &output_shape));

                            // Make sure we have at least 2 dims
                            while (output_shape.size() < 2)
                            {
                                output_shape.push_back(1);
                            }

                            const uint8_t* output_ptr = nullptr;
                            size_t output_ptr_size    = 0;
                            CHECK_TRITON(results->RawData(output_name, &output_ptr, &output_ptr_size));

                            auto output_buffer =
                                std::make_shared<rmm::device_buffer>(output_ptr_size, rmm::cuda_stream_per_thread);

                            cudaMemcpy(output_buffer->data(), output_ptr, output_ptr_size, cudaMemcpyHostToDevice);

                            mini_batch_output->get_output(output_name) =
                                Tensor::create(std::move(output_buffer),
                                               "<f4",
                                               std::vector<neo::TensorIndex>{static_cast<int>(output_shape[0]),
                                                                             static_cast<int>(output_shape[1])},
                                               std::vector<neo::TensorIndex>{},
                                               0);
                        }
                    }
                    output.on_next(std::move(response));
                },
                [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
                [&]() { output.on_completed(); }));
        };
    }

    std::string m_model_name;
    std::string m_server_url;
    std::map<std::string, std::string> m_inout_mapping;
};

}  // namespace morpheus
