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

#include <morpheus/cudf_helpers.hpp>
#include <morpheus/stages.hpp>

#include <neo/core/segment.hpp>
#include <pyneo/utils.hpp>

#include <pybind11/cast.h>
#include <pybind11/pytypes.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace morpheus {

using namespace std::literals;
namespace pyneo = neo::pyneo;
namespace py    = pybind11;
namespace fs    = std::filesystem;

cudf::io::table_with_metadata load_table(const std::string& filename)
{
    auto file_path = fs::path(filename);

    if (file_path.extension() == ".json" || file_path.extension() == ".jsonlines")
    {
        // First, load the file into json
        auto options = cudf::io::json_reader_options::builder(cudf::io::source_info{filename}).lines(true);

        return cudf::io::read_json(options.build());
    }
    else if (file_path.extension() == ".csv")
    {
        auto options = cudf::io::csv_reader_options::builder(cudf::io::source_info{filename});

        return cudf::io::read_csv(options.build());
    }
    else
    {
        LOG(FATAL) << "Unknown extension for file: " << filename;
        throw std::runtime_error("Unknown extension");
    }
}

bool str_contains(const std::string& str, const std::string& search_str)
{
    return str.find(search_str) != std::string::npos;
}

// Define the pybind11 module m, as 'pipeline'.
PYBIND11_MODULE(stages, m)
{
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------
        .. currentmodule:: scikit_build_example
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

    // Load the cudf helpers
    load_cudf_helpers();

    pyneo::import(m, "cupy");
    pyneo::import(m, "morpheus._lib.messages");

    py::class_<FileSourceStage, neo::SegmentObject, std::shared_ptr<FileSourceStage>>(
        m, "FileSourceStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent, const std::string& name, std::string filename, int repeat = 1) {
                 auto stage = std::make_shared<FileSourceStage>(parent, name, filename, repeat);

                 parent.register_node<FileSourceStage>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("repeat"));

    py::class_<KafkaSourceStage, neo::SegmentObject, std::shared_ptr<KafkaSourceStage>>(
        m, "KafkaSourceStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent,
                           const std::string& name,
                           size_t max_batch_size,
                           std::string topic,
                           int32_t batch_timeout_ms,
                           std::map<std::string, std::string> config,
                           bool disable_commits) {
                 auto stage = std::make_shared<KafkaSourceStage>(
                     parent, name, max_batch_size, topic, batch_timeout_ms, config, disable_commits);

                 parent.register_node<KafkaSourceStage>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("max_batch_size"),
             py::arg("topic"),
             py::arg("batch_timeout_ms"),
             py::arg("config"),
             py::arg("disable_commits") = false);

    py::class_<DeserializeStage, neo::SegmentObject, std::shared_ptr<DeserializeStage>>(
        m, "DeserializeStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent, const std::string& name, size_t batch_size) {
                 auto stage = std::make_shared<DeserializeStage>(parent, name, batch_size);

                 parent.register_node<DeserializeStage>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("batch_size"));

    py::class_<PreprocessNLPStage, neo::SegmentObject, std::shared_ptr<PreprocessNLPStage>>(
        m, "PreprocessNLPStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent,
                           const std::string& name,
                           std::string vocab_hash_file,
                           uint32_t sequence_length,
                           bool truncation,
                           bool do_lower_case,
                           bool add_special_token,
                           int stride = -1) {
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
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("vocab_hash_file"),
             py::arg("sequence_length"),
             py::arg("truncation"),
             py::arg("do_lower_case"),
             py::arg("add_special_token"),
             py::arg("stride"));

    py::class_<PreprocessFILStage, neo::SegmentObject, std::shared_ptr<PreprocessFILStage>>(
        m, "PreprocessFILStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent, const std::string& name) {
                 auto stage = std::make_shared<PreprocessFILStage>(parent, name);

                 parent.register_node<PreprocessFILStage>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"));

    py::class_<InferenceClientStage, neo::SegmentObject, std::shared_ptr<InferenceClientStage>>(
        m, "InferenceClientStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent,
                           const std::string& name,
                           std::string model_name,
                           std::string server_url,
                           bool force_convert_inputs,
                           bool use_shared_memory,
                           bool needs_logits,
                           std::map<std::string, std::string> inout_mapping) {
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
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("model_name"),
             py::arg("server_url"),
             py::arg("force_convert_inputs"),
             py::arg("use_shared_memory"),
             py::arg("needs_logits"),
             py::arg("inout_mapping") = py::dict());

    py::class_<FilterDetectionsStage, neo::SegmentObject, std::shared_ptr<FilterDetectionsStage>>(
        m, "FilterDetectionsStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent, const std::string& name, float threshold) {
                 auto stage = std::make_shared<FilterDetectionsStage>(parent, name, threshold);

                 parent.register_node<FilterDetectionsStage>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("threshold"));

    py::class_<AddClassificationsStage, neo::SegmentObject, std::shared_ptr<AddClassificationsStage>>(
        m, "AddClassificationsStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent,
                           const std::string& name,
                           float threshold,
                           std::size_t num_class_labels,
                           std::map<std::size_t, std::string> idx2label) {
                 auto stage =
                     std::make_shared<AddClassificationsStage>(parent, name, threshold, num_class_labels, idx2label);

                 parent.register_node<AddClassificationsStage>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("threshold"),
             py::arg("num_class_labels"),
             py::arg("idx2label"));

    py::class_<AddScoresStage, neo::SegmentObject, std::shared_ptr<AddScoresStage>>(
        m, "AddScoresStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent,
                           const std::string& name,
                           std::size_t num_class_labels,
                           std::map<std::size_t, std::string> idx2label) {
                 auto stage = std::make_shared<AddScoresStage>(parent, name, num_class_labels, idx2label);

                 parent.register_node<AddScoresStage>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("num_class_labels"),
             py::arg("idx2label"));

    py::class_<SerializeStage, neo::SegmentObject, std::shared_ptr<SerializeStage>>(
        m, "SerializeStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent,
                           const std::string& name,
                           const std::vector<std::string>& include,
                           const std::vector<std::string>& exclude,
                           bool fixed_columns) {
                 auto stage = std::make_shared<SerializeStage>(parent, name, include, exclude, fixed_columns);

                 parent.register_node<SerializeStage>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("include"),
             py::arg("exclude"),
             py::arg("fixed_columns") = true);

    py::class_<WriteToFileStage, neo::SegmentObject, std::shared_ptr<WriteToFileStage>>(
        m, "WriteToFileStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent,
                           const std::string& name,
                           const std::string& filename,
                           const std::string& mode = "w",
                           FileTypes file_type     = FileTypes::Auto) {
                 std::ios::openmode fsmode;

                 if (str_contains(mode, "r"))
                 {
                     // Dont support reading
                     throw std::invalid_argument("Read mode ('r') is not supported by WriteToFileStage. Mode: " + mode);
                 }
                 if (str_contains(mode, "b"))
                 {
                     // Dont support binary
                     throw std::invalid_argument("Binary mode ('b') is not supported by WriteToFileStage. Mode: " +
                                                 mode);
                 }

                 // Default is write
                 if (mode.empty() || str_contains(mode, "w"))
                 {
                     fsmode |= std::ios::out;
                 }

                 // Check for appending
                 if (str_contains(mode, "a"))
                 {
                     fsmode |= (std::ios::app | std::ios::out);
                 }

                 // Check for truncation
                 if (str_contains(mode, "+"))
                 {
                     fsmode |= (std::ios::trunc | std::ios::out);
                 }

                 // Ensure something was set
                 if (fsmode == std::ios::openmode())
                 {
                     throw std::runtime_error("Unsupported file mode: "s + mode);
                 }

                 auto stage = std::make_shared<WriteToFileStage>(parent, name, filename, fsmode);

                 parent.register_node<WriteToFileStage>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("mode")      = "w",
             py::arg("file_type") = 0);  // Setting this to FileTypes::AUTO throws a conversion error at runtime

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace morpheus
