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

#include <morpheus/stages/write_to_file.hpp>

#include <morpheus/utilities/matx_util.hpp>

#include <exception>
#include <memory>
#include <mutex>
#include <string>

namespace morpheus {
// Component public implementations
// ************ WriteToFileStage **************************** //
WriteToFileStage::WriteToFileStage(const neo::Segment &parent,
                                   const std::string &name,
                                   const std::string &filename,
                                   std::ios::openmode mode,
                                   FileTypes file_type) :
  neo::SegmentObject(parent, name),
  PythonNode(parent, name, build_operator()),
  m_is_first(true)
{
    if (file_type == FileTypes::Auto)
    {
        file_type = determine_file_type(filename);
    }

    using std::placeholders::_1;
    if (file_type == FileTypes::CSV)
    {
        m_write_func = [this](auto &&PH1) { write_csv(std::forward<decltype(PH1)>(PH1)); };
    }
    else if (file_type == FileTypes::JSON)
    {
        m_write_func = [this](auto &&PH1) { write_json(std::forward<decltype(PH1)>(PH1)); };
    }
    else  // FileTypes::AUTO
    {
        LOG(FATAL) << "Unknown extension for file: " << filename;
        throw std::runtime_error("Unknown extension");
    }

    m_fstream.open(filename, mode);
}

void WriteToFileStage::close()
{
    if (m_fstream.is_open())
    {
        m_fstream.close();
    }
}

WriteToFileStage::operator_fn_t WriteToFileStage::build_operator()
{
    return [this](neo::Observable<reader_type_t> &input, neo::Subscriber<writer_type_t> &output) {
        return input.subscribe(neo::make_observer<reader_type_t>(
            [this, &output](reader_type_t &&msg) {
                this->m_write_func(msg);
                m_is_first = false;
                output.on_next(std::move(msg));
            },
            [&](std::exception_ptr error_ptr) {
                this->close();
                output.on_error(error_ptr);
            },
            [&]() {
                this->close();
                output.on_completed();
            }));
    };
}

// ************ WriteToFileStageInterfaceProxy ************* //
std::shared_ptr<WriteToFileStage> WriteToFileStageInterfaceProxy::init(neo::Segment &parent,
                                                                       const std::string &name,
                                                                       const std::string &filename,
                                                                       const std::string &mode,
                                                                       FileTypes file_type)
{
    std::ios::openmode fsmode;

    if (StringUtil::str_contains(mode, "r"))
    {
        // Dont support reading
        throw std::invalid_argument("Read mode ('r') is not supported by WriteToFileStage. Mode: " + mode);
    }
    if (StringUtil::str_contains(mode, "b"))
    {
        // Dont support binary
        throw std::invalid_argument("Binary mode ('b') is not supported by WriteToFileStage. Mode: " + mode);
    }

    // Default is write
    if (mode.empty() || StringUtil::str_contains(mode, "w"))
    {
        fsmode |= std::ios::out;
    }

    // Check for appending
    if (StringUtil::str_contains(mode, "a"))
    {
        fsmode |= (std::ios::app | std::ios::out);
    }

    // Check for truncation
    if (StringUtil::str_contains(mode, "+"))
    {
        fsmode |= (std::ios::trunc | std::ios::out);
    }

    // Ensure something was set
    if (fsmode == std::ios::openmode())
    {
        throw std::runtime_error(std::string("Unsupported file mode: ") + mode);
    }

    auto stage = std::make_shared<WriteToFileStage>(parent, name, filename, fsmode, file_type);

    parent.register_node<WriteToFileStage>(stage);

    return stage;
}
}  // namespace morpheus
