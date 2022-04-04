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

#include <morpheus/stages/file_source.hpp>

#include <neo/core/segment.hpp>

#include <cudf/io/csv.hpp>
#include <cudf/io/json.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/types.hpp>
#include <nvtext/subword_tokenize.hpp>

#include <glog/logging.h>
#include <pybind11/gil.h>
#include <pybind11/pytypes.h>
#include <nlohmann/json.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <utility>

namespace morpheus {
    // Component public implementations
    // ************ FileSourceStage ************* //
    FileSourceStage::FileSourceStage(const neo::Segment &parent, const std::string &name, std::string filename,
                                     int repeat) :
            neo::SegmentObject(parent, name),
            base_t(parent, name),
            m_filename(std::move(filename)),
            m_repeat(repeat) {
        this->set_source_observable(neo::Observable<source_type_t>([this](neo::Subscriber<source_type_t> &sub) {
            auto data_table = this->load_table();

            // Using 0 will default to creating a new range index
            int index_col_count = 0;

            // Check if we have a first column with INT64 data type
            if (data_table.metadata.column_names.size() >= 1 &&
                data_table.tbl->get_column(0).type().id() == cudf::type_id::INT64) {
                std::regex index_regex(R"((unnamed: 0|id))",
                                       std::regex_constants::ECMAScript | std::regex_constants::icase);

                // Get the column name
                auto col_name = data_table.metadata.column_names[0];

                // Check it against some common terms
                if (std::regex_search(col_name, index_regex)) {
                    // Also, if its the hideous 'Unnamed: 0', then just use an empty string
                    if (col_name == "Unnamed: 0") {
                        data_table.metadata.column_names[0] = "";
                    }

                    index_col_count = 1;
                }
            }

            // Next, create the message metadata. This gets reused for repeats
            auto meta = MessageMeta::create_from_cpp(std::move(data_table), index_col_count);

            // Always push at least 1
            sub.on_next(meta);

            for (cudf::size_type repeat_idx = 1; repeat_idx < m_repeat; ++repeat_idx) {
                // Clone the previous meta object
                {
                    pybind11::gil_scoped_acquire gil;

                    // Use the copy function
                    auto df = meta->get_py_table().attr("copy")();

                    pybind11::int_ df_len = pybind11::len(df);

                    pybind11::object index = df.attr("index");

                    df.attr("index") = index + df_len;

                    meta = MessageMeta::create_from_python(std::move(df));
                }

                sub.on_next(meta);
            }

            sub.on_completed();
        }));
    }

    cudf::io::table_with_metadata FileSourceStage::load_table() {
        auto file_path = std::filesystem::path(m_filename);

        if (file_path.extension() == ".json" || file_path.extension() == ".jsonlines") {
            // First, load the file into json
            auto options = cudf::io::json_reader_options::builder(cudf::io::source_info{m_filename}).lines(true);

            auto tbl = cudf::io::read_json(options.build());

            auto found = std::find(tbl.metadata.column_names.begin(), tbl.metadata.column_names.end(), "data");

            if (found == tbl.metadata.column_names.end())
                return tbl;

            // Super ugly but cudf cant handle newlines and add extra escapes. So we need to convert
            // \\n -> \n
            // \\/ -> \/
            auto columns = tbl.tbl->release();

            size_t idx = found - tbl.metadata.column_names.begin();

            auto updated_data = cudf::strings::replace(
                    cudf::strings_column_view{columns[idx]->view()}, cudf::string_scalar("\\n"),
                    cudf::string_scalar("\n"));

            updated_data = cudf::strings::replace(
                    cudf::strings_column_view{updated_data->view()}, cudf::string_scalar("\\/"),
                    cudf::string_scalar("/"));

            columns[idx] = std::move(updated_data);

            tbl.tbl = std::move(std::make_unique<cudf::table>(std::move(columns)));

            return tbl;
        } else if (file_path.extension() == ".csv") {
            auto options = cudf::io::csv_reader_options::builder(cudf::io::source_info{m_filename});

            return cudf::io::read_csv(options.build());
        } else {
            LOG(FATAL) << "Unknown extension for file: " << m_filename;
            throw std::runtime_error("Unknown extension");
        }
    }

    // ************ FileSourceStageInterfaceProxy ************ //
    std::shared_ptr<FileSourceStage>
    FileSourceStageInterfaceProxy::init(neo::Segment &parent, const std::string &name, std::string filename,
                                        int repeat) {
        auto stage = std::make_shared<FileSourceStage>(parent, name, filename, repeat);

        parent.register_node<FileSourceStage>(stage);

        return stage;
    }
}
