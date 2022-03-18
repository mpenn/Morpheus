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

#pragma once

#include <morpheus/common.hpp>
#include <morpheus/cudf_helpers.hpp>
#include <morpheus/table_info.hpp>
#include <morpheus/type_utils.hpp>

#include <neo/core/tensor.hpp>
#include <neo/utils/type_utils.hpp>

#include <cudf/copying.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace morpheus {
struct PyDataTable : public IDataTable
{
    PyDataTable(pybind11::object&& py_table) : m_py_table(std::move(py_table)) {}

    ~PyDataTable()
    {
        if (m_py_table)
        {
            pybind11::gil_scoped_acquire gil;

            // Clear out the python object
            m_py_table = pybind11::object();
        }
    }

    cudf::size_type count() const override
    {
        pybind11::gil_scoped_acquire gil;
        return m_py_table.attr("_num_rows").cast<cudf::size_type>();
    }

    TableInfo get_info() const override
    {
        pybind11::gil_scoped_acquire gil;

        auto info = make_table_info_from_table((PyTable*)m_py_table.ptr(), this->shared_from_this());

        return info;
    }

    const pybind11::object& get_py_object() const override
    {
        return m_py_table;
    }

  private:
    pybind11::object m_py_table;
};

class MessageMeta
{
  public:
    pybind11::object get_py_table() const
    {
        return m_data->get_py_object();
    }

    size_t count() const
    {
        return m_data->count();
    }

    TableInfo get_info() const
    {
        return this->m_data->get_info();
    }

    static std::shared_ptr<MessageMeta> create_from_python(pybind11::object&& data_table)
    {
        auto data = std::make_unique<PyDataTable>(std::move(data_table));

        return std::shared_ptr<MessageMeta>(new MessageMeta(std::move(data)));
    }

    static std::shared_ptr<MessageMeta> create_from_cpp(cudf::io::table_with_metadata&& data_table,
                                                        int index_col_count = 0)
    {
        // Convert to py first
        pybind11::object py_dt = cpp_to_py(std::move(data_table), index_col_count);

        auto data = std::make_unique<PyDataTable>(std::move(py_dt));

        return std::shared_ptr<MessageMeta>(new MessageMeta(std::move(data)));
    }

  private:
    struct MessageMetaImpl
    {
        virtual pybind11::object get_py_table() const = 0;
        virtual TableInfo get_info() const            = 0;
    };

    MessageMeta(std::shared_ptr<IDataTable> data) : m_data(std::move(data)) {}

    static pybind11::object cpp_to_py(cudf::io::table_with_metadata&& table, int index_col_count = 0)
    {
        pybind11::gil_scoped_acquire gil;

        // Now convert to a python TableInfo object
        auto converted_table = pybind11::reinterpret_steal<pybind11::object>(
            (PyObject*)make_table_from_table_with_metadata(std::move(table), index_col_count));

        // VLOG(10) << "Table. Num Col: " << converted_table.attr("_num_columns").str().cast<std::string>()
        //          << ", Num Ind: " << converted_table.attr("_num_columns").cast<std::string>()
        //          << ", Rows: " << converted_table.attr("_num_rows").cast<std::string>();
        // pybind11::print("Table Created. Num Rows: {}, Num Cols: {}, Num Ind: {}",
        //           converted_table.attr("_num_rows"),
        //           converted_table.attr("_num_columns"),
        //           converted_table.attr("_num_indices"));

        return converted_table;
    }

    // struct MessageMetaPyImpl : public MessageMetaImpl
    // {
    //     MessageMetaPyImpl(pybind11::object&& pydf) : m_pydf(std::move(pydf)) {}

    //     MessageMetaPyImpl(cudf::io::table_with_metadata&& table) : m_pydf(std::move(cpp_to_py(std::move(table)))) {}

    //     pybind11::object get_py_table() const override
    //     {
    //         return m_pydf;
    //     }

    //     TableInfo get_info() const override
    //     {
    //         pybind11::gil_scoped_acquire gil;

    //         return make_table_info_from_table((PyTable*)this->m_pydf.ptr());
    //     }

    //     pybind11::object m_pydf;
    // };

    // struct MessageMetaCppImpl : public MessageMetaImpl
    // {
    //     MessageMetaCppImpl(cudf::io::table_with_metadata&& table) : m_table(std::move(table)) {}

    //     pybind11::object get_py_table() const override
    //     {
    //         pybind11::gil_scoped_acquire gil;

    //         // Get a python object from this data table
    //         pybind11::object py_datatable = pybind11::cast(m_data_table);

    //         // Now convert to a python TableInfo object
    //         auto converted_table = pybind11::reinterpret_steal<pybind11::object>(
    //             (PyObject*)make_table_from_datatable(m_data_table, (PyObject*)py_datatable.ptr()));

    //         return converted_table;
    //     }
    //     TableInfo get_info() const override
    //     {
    //         return TableInfo(m_data_table);
    //     }

    //     std::shared_ptr<DataTable> m_data_table;
    // };

    // std::unique_ptr<MessageMetaImpl> m_data;
    std::shared_ptr<IDataTable> m_data;
};

class InferenceMemory
{
  public:
    InferenceMemory(size_t count) : count(count) {}

    size_t count{0};
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
    InferenceMemoryNLP(size_t count,
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
    InferenceMemoryFIL(size_t count, neo::TensorObject input__0, neo::TensorObject seq_ids) : InferenceMemory(count)
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
    MultiMessage(std::shared_ptr<morpheus::MessageMeta> m, size_t o, size_t c) :
      meta(std::move(m)),
      mess_offset(o),
      mess_count(c)
    {}

    std::shared_ptr<morpheus::MessageMeta> meta;
    size_t mess_offset{0};
    size_t mess_count{0};

    TableInfo get_meta()
    {
        auto table_info = this->get_meta(std::vector<std::string>{});

        return table_info;
    }

    TableInfo get_meta(const std::string& col_name)
    {
        auto table_view = this->get_meta(std::vector<std::string>{col_name});

        return table_view;
    }

    TableInfo get_meta(const std::vector<std::string>& column_names)
    {
        TableInfo info = this->meta->get_info();

        TableInfo sliced_info = info.get_slice(this->mess_offset,
                                               this->mess_offset + this->mess_count,
                                               column_names.empty() ? info.get_column_names() : column_names);

        return sliced_info;
    }

    void set_meta(const std::string& col_name, neo::TensorObject tensor)
    {
        set_meta(std::vector<std::string>{col_name}, std::vector<neo::TensorObject>{tensor});
    }

    void set_meta(const std::vector<std::string>& column_names, const std::vector<neo::TensorObject>& tensors)
    {
        std::vector<neo::TypeId> tensor_types{tensors.size()};
        for (size_t i = 0; i < tensors.size(); ++i)
        {
            tensor_types[i] = tensors[i].dtype().type_id();
        }

        TableInfo info = this->meta->get_info();
        info.insert_missing_columns(column_names, tensor_types);

        TableInfo table_meta = this->get_meta(column_names);
        for (size_t i = 0; i < tensors.size(); ++i)
        {
            const auto cv          = table_meta.get_column(i);
            const auto table_type  = cv.type().id();
            const auto tensor_type = DType(tensor_types[i]).cudf_type_id();
            const auto row_stride  = tensors[i].stride(0);

            CHECK(tensors[i].count() == cv.size() &&
                  (table_type == tensor_type ||
                   (table_type == cudf::type_id::BOOL8 && tensor_type == cudf::type_id::UINT8)));

            if (row_stride == 1)
            {
                // column major just use cudaMemcpy
                NEO_CHECK_CUDA(cudaMemcpy(const_cast<uint8_t*>(cv.data<uint8_t>()),
                                          tensors[i].data(),
                                          tensors[i].bytes(),
                                          cudaMemcpyDeviceToDevice));
            }
            else
            {
                const auto item_size = tensors[i].dtype().item_size();
                NEO_CHECK_CUDA(cudaMemcpy2D(const_cast<uint8_t*>(cv.data<uint8_t>()),
                                            item_size,
                                            tensors[i].data(),
                                            row_stride * item_size,
                                            item_size,
                                            cv.size(),
                                            cudaMemcpyDeviceToDevice));
            }
        }
    }

    std::shared_ptr<MultiMessage> get_slice(size_t start, size_t stop) const
    {
        // This can only cast down
        return std::static_pointer_cast<MultiMessage>(this->internal_get_slice(start, stop));
    }

  protected:
    // This internal function is used to allow virtual overriding while `get_slice` allows for hiding of base class.
    // This allows users to avoid casting every class after calling get_slice but still supports calling `get_slice`
    // from a base class. For example, the following all works:
    // std::shared_ptr<DerivedMultiMessage> derived_message = std::make_shared<DerivedMultiMessage>();
    //
    // // No cast is necessary here
    // std::shared_ptr<DerivedMultiMessage> other_derived = derived_message->get_slice(0, 10);
    //
    // // Conversion to base class
    // std::shared_ptr<MultiMessage> base_message = derived_message;
    //
    // // This also works
    // std::shared_ptr<MultiMessage> other_base = base_message->get_slice(0, 10);
    //
    // These will be logically equivalent
    // assert(std::dynamic_ptr_cast<DerivedMultiMessage>(other_base) == other_derived);
    virtual std::shared_ptr<MultiMessage> internal_get_slice(size_t start, size_t stop) const
    {
        auto mess_start = this->mess_offset + start;
        auto mess_stop  = this->mess_offset + stop;
        return std::make_shared<MultiMessage>(this->meta, mess_start, mess_stop - mess_start);
    }
};

class MultiInferenceMessage : public MultiMessage
{
  public:
    MultiInferenceMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                          size_t mess_offset,
                          size_t mess_count,
                          std::shared_ptr<morpheus::InferenceMemory> memory,
                          size_t offset,
                          size_t count) :
      MultiMessage(meta, mess_offset, mess_count),
      memory(std::move(memory)),
      offset(offset),
      count(count)
    {}
    std::shared_ptr<morpheus::InferenceMemory> memory;
    size_t offset{0};
    size_t count{0};

    const neo::TensorObject get_input(const std::string& name) const
    {
        CHECK(this->memory->has_input(name)) << "Cound not find input: " << name;

        // check if we are getting the entire input
        if (this->offset == 0 && this->count == this->memory->count)
        {
            return this->memory->inputs[name];
        }

        // TODO(MDD): This really needs to return the slice of the tensor
        return this->memory->inputs[name].slice({static_cast<cudf::size_type>(this->offset), 0},
                                                {static_cast<cudf::size_type>(this->offset + this->count), -1});
    }

    const void set_input(const std::string& name, const neo::TensorObject& value)
    {
        // Get the input slice first
        auto slice = this->get_input(name);

        // Set the value to use assignment
        slice = value;
    }

    std::shared_ptr<MultiInferenceMessage> get_slice(size_t start, size_t stop) const
    {
        // This can only cast down
        return std::static_pointer_cast<MultiInferenceMessage>(this->internal_get_slice(start, stop));
    }

  protected:
    std::shared_ptr<MultiMessage> internal_get_slice(size_t start, size_t stop) const override
    {
        CHECK(this->mess_count == this->count) << "At this time, mess_count and count must be the same for slicing";

        auto mess_start = this->mess_offset + start;
        auto mess_stop  = this->mess_offset + stop;

        // If we have more inference rows than message rows, we need to use the seq_ids to figure out the slicing. This
        // will be slow and should be avoided at all costs
        if (this->memory->has_input("seq_ids") && this->count != this->mess_count)
        {
            auto seq_ids = this->get_input("seq_ids");

            // Convert to MatX to access elements
            mess_start = this->mess_offset + seq_ids.read_element<int32_t>({(neo::TensorIndex)start, 0});
            mess_stop  = this->mess_offset + seq_ids.read_element<int32_t>({(neo::TensorIndex)stop - 1, 0}) + 1;
        }

        return std::make_shared<MultiInferenceMessage>(
            this->meta, mess_start, mess_stop - mess_start, this->memory, start, stop - start);
    }
};

#define INPUT_PROP(name)                           \
    const neo::TensorObject get_##name() const     \
    {                                              \
        return this->get_input(#name);             \
    }                                              \
    void set_##name(const neo::TensorObject& name) \
    {                                              \
        this->set_input(#name, name);              \
    }

#define OUTPUT_PROP(name)                          \
    const neo::TensorObject get_##name() const     \
    {                                              \
        return this->get_output(#name);            \
    }                                              \
    void set_##name(const neo::TensorObject& name) \
    {                                              \
        this->set_output(#name, name);             \
    }

class MultiInferenceNLPMessage : public MultiInferenceMessage
{
  public:
    MultiInferenceNLPMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                             size_t mess_offset,
                             size_t mess_count,
                             std::shared_ptr<morpheus::InferenceMemory> memory,
                             size_t offset,
                             size_t count) :
      MultiInferenceMessage(meta, mess_offset, mess_count, memory, offset, count)
    {}

    INPUT_PROP(input_ids);
    INPUT_PROP(input_mask);
    INPUT_PROP(seq_ids);
};

class MultiInferenceFILMessage : public MultiInferenceMessage
{
  public:
    MultiInferenceFILMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                             size_t mess_offset,
                             size_t mess_count,
                             std::shared_ptr<morpheus::InferenceMemory> memory,
                             size_t offset,
                             size_t count) :
      MultiInferenceMessage(meta, mess_offset, mess_count, memory, offset, count)
    {}

    INPUT_PROP(input__0);
    INPUT_PROP(seq_ids);
};

class ResponseMemory
{
  public:
    ResponseMemory(size_t count) : count(count) {}

    size_t count{0};
    std::map<std::string, neo::TensorObject> outputs;

    bool has_output(const std::string& name) const
    {
        return this->outputs.find(name) != this->outputs.end();
    }
};

class ResponseMemoryProbs : public ResponseMemory
{
  public:
    ResponseMemoryProbs(size_t count, neo::TensorObject probs) : ResponseMemory(count)
    {
        this->outputs["probs"] = std::move(probs);
    }

    DATA_CLASS_PROP(probs, this->outputs)
};

class MultiResponseMessage : public MultiMessage
{
  public:
    MultiResponseMessage(std::shared_ptr<morpheus::MessageMeta> meta,
                         size_t mess_offset,
                         size_t mess_count,
                         std::shared_ptr<morpheus::ResponseMemory> memory,
                         size_t offset,
                         size_t count) :
      MultiMessage(meta, mess_offset, mess_count),
      memory(std::move(memory)),
      offset(offset),
      count(count)
    {}
    std::shared_ptr<morpheus::ResponseMemory> memory;
    size_t offset{0};
    size_t count{0};

    neo::TensorObject get_output(const std::string& name)
    {
        CHECK(this->memory->has_output(name)) << "Cound not find output: " << name;

        // check if we are getting the entire input
        if (this->offset == 0 && this->count == this->memory->count)
        {
            return this->memory->outputs[name];
        }

        // TODO(MDD): This really needs to return the slice of the tensor
        return this->memory->outputs[name].slice({static_cast<cudf::size_type>(this->offset), 0},
                                                 {static_cast<cudf::size_type>(this->offset + this->count), -1});
    }

    const neo::TensorObject get_output(const std::string& name) const
    {
        CHECK(this->memory->has_output(name)) << "Cound not find output: " << name;

        // check if we are getting the entire input
        if (this->offset == 0 && this->count == this->memory->count)
        {
            return this->memory->outputs[name];
        }

        // TODO(MDD): This really needs to return the slice of the tensor
        return this->memory->outputs[name].slice({static_cast<cudf::size_type>(this->offset), 0},
                                                 {static_cast<cudf::size_type>(this->offset + this->count), -1});
    }

    const void set_output(const std::string& name, const neo::TensorObject& value)
    {
        // Get the input slice first
        auto slice = this->get_output(name);

        // Set the value to use assignment
        slice = value;
    }

    std::shared_ptr<MultiResponseMessage> get_slice(size_t start, size_t stop) const
    {
        // This can only cast down
        return std::static_pointer_cast<MultiResponseMessage>(this->internal_get_slice(start, stop));
    }

  protected:
    std::shared_ptr<MultiMessage> internal_get_slice(size_t start, size_t stop) const override
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
                              size_t mess_offset,
                              size_t mess_count,
                              std::shared_ptr<morpheus::ResponseMemory> memory,
                              size_t offset,
                              size_t count) :
      MultiResponseMessage(meta, mess_offset, mess_count, memory, offset, count)
    {}

    std::shared_ptr<MultiResponseProbsMessage> get_slice(size_t start, size_t stop) const
    {
        // This can only cast down
        return std::static_pointer_cast<MultiResponseProbsMessage>(this->internal_get_slice(start, stop));
    }

    OUTPUT_PROP(probs)
};
}  // namespace morpheus
