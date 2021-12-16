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
#include <memory>
#include <stdexcept>
#include <utility>

#include <cudf/copying.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

namespace pybind11 {
struct object;
}

namespace morpheus {

struct TableInfo;

// Owning object which owns a unique_ptr<cudf::table>, table_metadata, and index information
// Why this doesnt exist in cudf is beyond me
struct IDataTable : public std::enable_shared_from_this<IDataTable>
{
    IDataTable() = default;

    virtual cudf::size_type count() const = 0;

    virtual TableInfo get_info() const = 0;

    virtual const pybind11::object& get_py_object() const = 0;

    // cudf::size_type num_indices() const
    // {
    //     return index_names.size();
    // }
    // cudf::size_type num_columns() const
    // {
    //     return column_names.size();
    // }

    // cudf::table_view get_view() const
    // {
    //     return m_table->view();
    // }

    // std::vector<std::string> all_names() const
    // {
    //     std::vector<std::string> all_names;
    //     all_names.reserve(this->num_indices() + this->num_columns());

    //     all_names.insert(all_names.end(), this->index_names.begin(), this->index_names.end());
    //     all_names.insert(all_names.end(), this->column_names.begin(), this->column_names.end());

    //     return all_names;
    // }

    // cudf::io::table_metadata get_table_metadata() const
    // {
    //     cudf::io::table_metadata meta;

    //     meta.column_names = this->all_names();

    //     return meta;
    // }

  private:
    // std::unique_ptr<cudf::table> m_table;
};

struct TableInfo
{
    TableInfo() = default;
    TableInfo(std::shared_ptr<const IDataTable> parent,
              cudf::table_view view,
              std::vector<std::string> index_names,
              std::vector<std::string> column_names) :
      m_parent(std::move(parent)),
      m_table_view(std::move(view)),
      m_index_names(std::move(index_names)),
      m_column_names(std::move(column_names))
    {}

    const pybind11::object& get_parent_table() const
    {
        return m_parent->get_py_object();
    }

    const cudf::table_view& get_view() const
    {
        return m_table_view;
    }

    std::vector<std::string> get_index_names() const
    {
        return m_index_names;
    }

    std::vector<std::string> get_column_names() const
    {
        return m_column_names;
    }

    cudf::size_type num_indices() const
    {
        return this->get_index_names().size();
    }
    cudf::size_type num_columns() const
    {
        return this->get_column_names().size();
    }
    cudf::size_type num_rows() const
    {
        return this->m_table_view.num_rows();
    }

    const cudf::column_view& get_column(cudf::size_type idx) const
    {
        if (idx < 0 || idx >= this->m_table_view.num_columns())
        {
            throw std::invalid_argument("idx must satisfy 0 <= idx < num_columns()");
        }

        return this->m_table_view.column(this->m_index_names.size() + idx);
    }

    TableInfo get_slice(cudf::size_type start, cudf::size_type stop, std::vector<std::string> column_names = {}) const
    {
        std::vector<cudf::size_type> col_indices;

        std::vector<std::string> new_column_names;

        // Append the indices column idx by default
        for (cudf::size_type i = 0; i < this->m_index_names.size(); ++i)
        {
            col_indices.push_back(i);
        }

        std::transform(column_names.begin(),
                       column_names.end(),
                       std::back_inserter(col_indices),
                       [this, &new_column_names](const std::string& c) {
                           auto found_col = std::find(this->m_column_names.begin(), this->m_column_names.end(), c);

                           if (found_col == this->m_column_names.end())
                           {
                               throw std::runtime_error("Unknown column: " + c);
                           }

                           // Add the found column to the metadata
                           new_column_names.push_back(c);

                           return (found_col - this->m_column_names.begin() + this->num_indices());
                       });

        auto slice_rows = cudf::slice(m_table_view, {start, stop})[0];

        auto slice_cols = slice_rows.select(col_indices);

        return TableInfo(m_parent, slice_cols, m_index_names, new_column_names);
    }

  private:
    std::shared_ptr<const IDataTable> m_parent;
    cudf::table_view m_table_view;
    std::vector<std::string> m_column_names;
    std::vector<std::string> m_index_names;
};

}  // namespace morpheus
