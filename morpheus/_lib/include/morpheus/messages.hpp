#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "morpheus/common.hpp"
#include "morpheus/table_info.hpp"

#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <cudf/copying.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

// MUST COME LAST!!!
#include "cudf_helpers_api.h"

namespace morpheus {

namespace neo   = trtlab::neo;
namespace py    = pybind11;
namespace pyneo = trtlab::neo::pyneo;
namespace fs    = std::filesystem;
// using json      = nlohmann::json;

class MessageMeta
{
  public:
    std::vector<std::string> input_json;

    py::object get_py_table() const
    {
        return m_data->get_py_table();
    }

    cudf::size_type count() const
    {
        return this->get_info().table_view.num_rows();
    }

    TableInfo get_info() const
    {
        return this->m_data->get_info();
    }

    static std::shared_ptr<MessageMeta> create_from_python(py::object&& data_table,
                                                           const std::vector<std::string>&& input_strings)
    {
        auto data = std::make_unique<MessageMetaPyImpl>(std::move(data_table));

        return std::shared_ptr<MessageMeta>(new MessageMeta(std::move(data), std::move(input_strings)));
    }

    static std::shared_ptr<MessageMeta> create_from_cpp(cudf::io::table_with_metadata&& data_table,
                                                        const std::vector<std::string>&& input_strings)
    {
        auto data = std::make_unique<MessageMetaCppImpl>(std::move(data_table));

        return std::shared_ptr<MessageMeta>(new MessageMeta(std::move(data), std::move(input_strings)));
    }

  private:
    struct MessageMetaImpl
    {
        virtual py::object get_py_table() const = 0;
        virtual TableInfo get_info() const      = 0;
    };

    MessageMeta(std::unique_ptr<MessageMetaImpl>&& data, const std::vector<std::string>&& input_strings) :
      m_data(std::move(data)),
      input_json(std::move(input_strings))
    {}

    struct MessageMetaPyImpl : public MessageMetaImpl
    {
        MessageMetaPyImpl(py::object&& pydf) : m_pydf(std::move(pydf)) {}

        py::object get_py_table() const override
        {
            return m_pydf;
        }

        TableInfo get_info() const override
        {
            py::gil_scoped_acquire gil;

            return make_table_info_from_table((PyTable*)this->m_pydf.ptr());
        }

        py::object m_pydf;
    };

    struct MessageMetaCppImpl : public MessageMetaImpl
    {
        MessageMetaCppImpl(cudf::io::table_with_metadata&& table) : m_table(std::move(table)) {}

        py::object get_py_table() const override
        {
            py::gil_scoped_acquire gil;

            // Convert to Table first
            auto converted_table = py::reinterpret_steal<py::object>(
                (PyObject*)make_table_from_view_and_meta(m_table.tbl->view(), m_table.metadata));

            return converted_table;
        }
        TableInfo get_info() const override
        {
            return TableInfo(m_table.tbl->view(), m_table.metadata);
        }

        cudf::io::table_with_metadata m_table;
    };

    std::unique_ptr<MessageMetaImpl> m_data;
};

class InferenceMemory
{
  public:
    InferenceMemory(cudf::size_type count) : count(count) {}

    cudf::size_type count{0};
    std::map<std::string, neo::TensorObject> inputs;

    bool has_input(const std::string& name) const
    {
        return this->inputs.find(name) != this->inputs.end();
    }
};

#define DATA_CLASS_PROP(name, map)                                               \
    const neo::TensorObject& get_##name() const                                  \
    {                                                                            \
        auto found = map.find(#name);                                            \
        if (found == map.end())                                                  \
        {                                                                        \
            throw std::runtime_error("Tensor: '" #name "' not found in memory"); \
        }                                                                        \
        return found->second;                                                    \
    }                                                                            \
    void set_##name(neo::TensorObject name)                                      \
    {                                                                            \
        map[#name] = std::move(name);                                            \
    }

class InferenceMemoryNLP : public InferenceMemory
{
  public:
    InferenceMemoryNLP(cudf::size_type count,
                       neo::TensorObject input_ids,
                       neo::TensorObject input_mask,
                       neo::TensorObject seq_ids) :
      InferenceMemory(count)
    {
        this->inputs["input_ids"]  = std::move(input_ids);
        this->inputs["input_mask"] = std::move(input_mask);
        this->inputs["seq_ids"]    = std::move(seq_ids);
    }

    DATA_CLASS_PROP(input_ids, this->inputs)
    DATA_CLASS_PROP(input_mask, this->inputs)
    DATA_CLASS_PROP(seq_ids, this->inputs)
};

class InferenceMemoryFIL : public InferenceMemory
{
  public:
    InferenceMemoryFIL(cudf::size_type count, neo::TensorObject input__0, neo::TensorObject seq_ids) :
      InferenceMemory(count)
    {
        this->inputs["input__0"] = std::move(input__0);
        this->inputs["seq_ids"]  = std::move(seq_ids);
    }

    DATA_CLASS_PROP(input__0, this->inputs)
    DATA_CLASS_PROP(seq_ids, this->inputs)
};

class MultiMessage
{
  public:
    MultiMessage(std::shared_ptr<morpheus::MessageMeta> m, cudf::size_type o, cudf::size_type c) :
      meta(std::move(m)),
      mess_offset(o),
      mess_count(c)
    {}

    cudf::column_view get_meta(const std::string& col_name)
    {
        auto table_view = this->get_meta(std::vector<std::string>{col_name});

        return table_view.column(0);
    }

    cudf::table_view get_meta(const std::vector<std::string>& column_names)
    {
        TableInfo info = this->meta->get_info();

        std::vector<cudf::size_type> col_indices;

        std::transform(column_names.begin(),
                       column_names.end(),
                       std::back_inserter(col_indices),
                       [this, info](const std::string& c) {
                           auto found_col =
                               std::find(info.metadata.column_names.begin(), info.metadata.column_names.end(), c);

                           if (found_col == info.metadata.column_names.end())
                           {
                               throw std::runtime_error("Unknown column: " + c);
                           }

                           return found_col - info.metadata.column_names.begin();
                       });

        auto table_slice = cudf::slice(info.table_view, {this->mess_offset, this->mess_offset + this->mess_count})[0];

        return table_slice.select(col_indices);
    }

    std::shared_ptr<morpheus::MessageMeta> meta;
    cudf::size_type mess_offset{0};
    cudf::size_type mess_count{0};
};

class MultiInferenceMessage : public MultiMessage
{
  public:
    MultiInferenceMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                          cudf::size_type mess_offset,
                          cudf::size_type mess_count,
                          std::shared_ptr<morpheus::InferenceMemory> memory,
                          cudf::size_type offset,
                          cudf::size_type count) :
      MultiMessage(meta, mess_offset, mess_count),
      memory(std::move(memory)),
      offset(offset),
      count(count)
    {}
    std::shared_ptr<morpheus::InferenceMemory> memory;
    cudf::size_type offset{0};
    cudf::size_type count{0};

    const neo::TensorObject get_input(const std::string& name) const
    {
        CHECK(this->memory->has_input(name)) << "Cound not find input: " << name;

        // check if we are getting the entire input
        if (this->offset == 0 && this->count == this->memory->count)
        {
            return this->memory->inputs[name];
        }

        // TODO(MDD): This really needs to return the slice of the tensor
        return this->memory->inputs[name].slice({this->offset, 0}, {this->offset + this->count, -1});
    }

    std::shared_ptr<MultiInferenceMessage> get_slice(size_t start, size_t stop) const
    {
        CHECK(this->mess_count == this->count) << "At this time, mess_count and count must be the same for slicing";

        auto mess_start = this->mess_offset + start;
        auto mess_stop  = this->mess_offset + stop;

        if (this->memory->has_input("seq_ids"))
        {
            auto seq_ids = this->get_input("seq_ids");

            // Convert to MatX to access elements
            mess_start = seq_ids.read_element<int32_t>({(neo::TensorIndex)start, 0});
            mess_stop  = seq_ids.read_element<int32_t>({(neo::TensorIndex)stop - 1, 0}) + 1;
        }

        return std::make_shared<MultiInferenceMessage>(
            this->meta, mess_start, mess_stop - mess_start, this->memory, start, stop - start);
    }
};

class MultiInferenceNLPMessage : public MultiInferenceMessage
{
  public:
    MultiInferenceNLPMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                             cudf::size_type mess_offset,
                             cudf::size_type mess_count,
                             std::shared_ptr<morpheus::InferenceMemory> memory,
                             cudf::size_type offset,
                             cudf::size_type count) :
      MultiInferenceMessage(meta, mess_offset, mess_count, memory, offset, count)
    {}
};

class MultiInferenceFILMessage : public MultiInferenceMessage
{
  public:
    MultiInferenceFILMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                             cudf::size_type mess_offset,
                             cudf::size_type mess_count,
                             std::shared_ptr<morpheus::InferenceMemory> memory,
                             cudf::size_type offset,
                             cudf::size_type count) :
      MultiInferenceMessage(meta, mess_offset, mess_count, memory, offset, count)
    {}
};

class ResponseMemory
{
  public:
    ResponseMemory(cudf::size_type count) : count(count) {}

    cudf::size_type count{0};
    std::map<std::string, neo::TensorObject> outputs;
};

class ResponseMemoryProbs : public ResponseMemory
{
  public:
    ResponseMemoryProbs(cudf::size_type count, neo::TensorObject probs) : ResponseMemory(count)
    {
        this->outputs["probs"] = std::move(probs);
    }

    DATA_CLASS_PROP(probs, this->outputs)
};

class MultiResponseMessage : public MultiMessage
{
  public:
    MultiResponseMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                         cudf::size_type mess_offset,
                         cudf::size_type mess_count,
                         std::shared_ptr<morpheus::ResponseMemory> memory,
                         cudf::size_type offset,
                         cudf::size_type count) :
      MultiMessage(meta, mess_offset, mess_count),
      memory(std::move(memory)),
      offset(offset),
      count(count)
    {}
    std::shared_ptr<morpheus::ResponseMemory> memory;
    cudf::size_type offset{0};
    cudf::size_type count{0};

    const neo::TensorObject& get_output(const std::string& name) const
    {
        auto found = this->memory->outputs.find(name);

        if (found == this->memory->outputs.end())
        {
            throw std::runtime_error("Cound not find output: " + name);
        }

        // TODO(MDD): This really needs to return the slice of the tensor
        return found->second;
    }

    neo::TensorObject get_output(const std::string& name)
    {
        auto found = this->memory->outputs.find(name);

        if (found == this->memory->outputs.end())
        {
            throw std::runtime_error("Cound not find output: " + name);
        }

        // TODO(MDD): This really needs to return the slice of the tensor
        return found->second;
    }

    std::shared_ptr<MultiResponseMessage> get_slice(size_t start, size_t stop) const
    {
        CHECK(this->mess_count == this->count) << "At this time, mess_count and count must be the same for slicing";

        auto mess_start = this->mess_offset + start;
        auto mess_stop  = this->mess_offset + stop;

        return std::make_shared<MultiResponseMessage>(
            this->meta, mess_start, mess_stop - mess_start, this->memory, start, stop - start);
    }
};

class MultiResponseProbsMessage : public MultiResponseMessage
{
  public:
    MultiResponseProbsMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                              cudf::size_type mess_offset,
                              cudf::size_type mess_count,
                              std::shared_ptr<morpheus::ResponseMemory> memory,
                              cudf::size_type offset,
                              cudf::size_type count) :
      MultiResponseMessage(meta, mess_offset, mess_count, memory, offset, count)
    {}
};
}  // namespace morpheus
