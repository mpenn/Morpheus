#include "morpheus/stages.hpp"

#include <memory>
#include <vector>

#include "pyneo/utils.hpp"

namespace morpheus {

namespace neo = trtlab::neo;
namespace py  = pybind11;

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

    pyneo::import(m, "cupy");

    py::class_<FileSourceStage, neo::SegmentObject, std::shared_ptr<FileSourceStage>>(
        m, "FileSourceStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent,
                           const std::string& name,
                           std::string filename,
                           int32_t batch_size,
                           int repeat = 1) {
                 auto stage = std::make_shared<FileSourceStage>(parent, name, filename, batch_size, repeat);

                 parent.register_node<FileSourceStage::source_type_t, FileSourceStage::source_type_t>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("filename"),
             py::arg("batch_size"),
             py::arg("repeat"));

    py::class_<PreprocessNLPStage, neo::SegmentObject, std::shared_ptr<PreprocessNLPStage>>(
        m, "PreprocessNLPStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent, const std::string& name, std::string vocab_file) {
                 auto stage = std::make_shared<PreprocessNLPStage>(parent, name, vocab_file);

                 parent.register_node<PreprocessNLPStage::source_type_t, PreprocessNLPStage::sink_type_t>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("vocab_file"));

    py::class_<PreprocessFILStage, neo::SegmentObject, std::shared_ptr<PreprocessFILStage>>(
        m, "PreprocessFILStage", py::multiple_inheritance())
        .def(py::init<>([](neo::Segment& parent, const std::string& name) {
                 auto stage = std::make_shared<PreprocessFILStage>(parent, name);

                 parent.register_node<PreprocessFILStage::source_type_t, PreprocessFILStage::sink_type_t>(stage);

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
                           std::map<std::string, std::string> inout_mapping) {
                 auto stage =
                     std::make_shared<InferenceClientStage>(parent, name, model_name, server_url, inout_mapping);

                 parent.register_node<InferenceClientStage::source_type_t, InferenceClientStage::sink_type_t>(stage);

                 return stage;
             }),
             py::arg("parent"),
             py::arg("name"),
             py::arg("model_name"),
             py::arg("server_url"),
             py::arg("inout_mapping") = py::dict());

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace morpheus
