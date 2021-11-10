#include "morpheus/messages.hpp"

#include <memory>
#include <vector>

#include "pybind11/pybind11.h"
#include "pyneo/utils.hpp"
#include "trtlab/neo/channels/channel.h"

namespace morpheus {

namespace neo = trtlab::neo;
namespace py  = pybind11;

// Define the pybind11 module m, as 'pipeline'.
PYBIND11_MODULE(messages, m)
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
    if (import_morpheus___lib__cudf_helpers() != 0)
    {
        py::error_already_set ex;

        LOG(ERROR) << "Could not load cudf_helpers library: " << ex.what();
        throw ex;
    }

    pyneo::import(m, "cupy");

    neo::EdgeConnector<std::shared_ptr<morpheus::MessageMeta>, py::object>::register_converter();
    neo::EdgeConnector<py::object, std::shared_ptr<morpheus::MessageMeta>>::register_converter();

    py::class_<morpheus::MessageMeta, std::shared_ptr<morpheus::MessageMeta>>(m, "MessageMeta")
        .def(py::init<>([](py::object df, std::vector<std::string> input_json) {
                 return morpheus::MessageMeta::create_from_python(std::move(df), std::move(input_json));
             }),
             py::arg("df"),
             py::arg("input_json"))
        .def_property_readonly("count", [](morpheus::MessageMeta& self) { return self.count(); })
        .def_property_readonly(
            "df",
            [](morpheus::MessageMeta& self) {
                // // Get the column and convert to cudf
                // auto py_table_struct = make_table_from_view_and_meta(self.m_pydf;.tbl->view(),
                // self.m_pydf;.metadata); py::object py_table  =
                // py::reinterpret_steal<py::object>((PyObject*)py_table_struct);

                // // py_col.inc_ref();

                // return py_table;
                return self.get_py_table();
            },
            py::return_value_policy::move);

    neo::EdgeConnector<std::shared_ptr<morpheus::MultiMessage>, py::object>::register_converter();
    neo::EdgeConnector<py::object, std::shared_ptr<morpheus::MultiMessage>>::register_converter();

    py::class_<morpheus::MultiMessage, std::shared_ptr<morpheus::MultiMessage>>(m, "MultiMessage")
        .def(py::init<>([](std::shared_ptr<morpheus::MessageMeta> meta,
                           cudf::size_type mess_offset,
                           cudf::size_type mess_count) {
                 return std::make_shared<morpheus::MultiMessage>(std::move(meta), mess_offset, mess_count);
             }),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"))
        .def_readonly("meta", &morpheus::MultiMessage::meta)
        .def_readonly("mess_offset", &morpheus::MultiMessage::mess_offset)
        .def_readonly("mess_count", &morpheus::MultiMessage::mess_count)
        .def(
            "get_meta",
            [](morpheus::MultiMessage& self, const std::string& col_name) {
                // Get the column and convert to cudf
                cudf::column_view col_view = self.get_meta(col_name);

                auto py_col_struct = make_column_from_view(col_view);
                py::object py_col  = py::reinterpret_steal<py::object>((PyObject*)py_col_struct);

                // py_col.inc_ref();

                return py_col;
            },
            py::return_value_policy::move);

    py::class_<morpheus::InferenceMemory, std::shared_ptr<morpheus::InferenceMemory>>(m, "InferenceMemory")
        .def_readonly("count", &morpheus::InferenceMemory::count);

    py::class_<InferenceMemoryNLP, InferenceMemory, std::shared_ptr<InferenceMemoryNLP>>(m, "InferenceMemoryNLP")
        .def(py::init<>([](cudf::size_type count, py::object input_ids, py::object input_mask, py::object seq_ids) {
                 // Conver the cupy arrays to tensors
                 return std::make_shared<InferenceMemoryNLP>(count,
                                                             std::move(cupy_to_tensor(input_ids)),
                                                             std::move(cupy_to_tensor(input_mask)),
                                                             std::move(cupy_to_tensor(seq_ids)));
             }),
             py::arg("count"),
             py::arg("input_ids"),
             py::arg("input_mask"),
             py::arg("seq_ids"))
        .def_readonly("count", &InferenceMemoryNLP::count)
        .def_property(
            "input_ids",
            [&](InferenceMemoryNLP& self) { return tensor_to_cupy(self.get_input_ids(), m); },
            [](InferenceMemoryNLP& self, py::object cupy_value) {
                return self.set_input_ids(cupy_to_tensor(cupy_value));
            })
        .def_property(
            "input_mask",
            [&](InferenceMemoryNLP& self) { return tensor_to_cupy(self.get_input_mask(), m); },
            [](InferenceMemoryNLP& self, py::object cupy_value) {
                return self.set_input_mask(cupy_to_tensor(cupy_value));
            })
        .def_property(
            "seq_ids",
            [&](InferenceMemoryNLP& self) { return tensor_to_cupy(self.get_seq_ids(), m); },
            [](InferenceMemoryNLP& self, py::object cupy_value) {
                return self.set_seq_ids(cupy_to_tensor(cupy_value));
            });

    py::class_<InferenceMemoryFIL, InferenceMemory, std::shared_ptr<InferenceMemoryFIL>>(m, "InferenceMemoryFIL")
        .def(py::init<>([](cudf::size_type count, py::object input__0, py::object seq_ids) {
                 // Conver the cupy arrays to tensors
                 return std::make_shared<InferenceMemoryFIL>(
                     count, std::move(cupy_to_tensor(input__0)), std::move(cupy_to_tensor(seq_ids)));
             }),
             py::arg("count"),
             py::arg("input__0"),
             py::arg("seq_ids"))
        .def_readonly("count", &InferenceMemoryFIL::count)
        .def("get_tensor", [](InferenceMemoryFIL& self, const std::string& name) { return self.inputs[name]; })
        .def_property(
            "input__0",
            [&](InferenceMemoryFIL& self) { return tensor_to_cupy(self.get_input__0(), m); },
            [](InferenceMemoryFIL& self, py::object cupy_value) {
                return self.set_input__0(cupy_to_tensor(cupy_value));
            })
        .def_property(
            "seq_ids",
            [&](InferenceMemoryFIL& self) { return tensor_to_cupy(self.get_seq_ids(), m); },
            [](InferenceMemoryFIL& self, py::object cupy_value) {
                return self.set_seq_ids(cupy_to_tensor(cupy_value));
            });

    neo::EdgeConnector<std::shared_ptr<morpheus::MultiInferenceMessage>, py::object>::register_converter();
    neo::EdgeConnector<py::object, std::shared_ptr<morpheus::MultiInferenceMessage>>::register_converter();

    py::class_<morpheus::MultiInferenceMessage,
               morpheus::MultiMessage,
               std::shared_ptr<morpheus::MultiInferenceMessage>>(m, "MultiInferenceMessage")
        .def(py::init<>([](std::shared_ptr<morpheus::MessageMeta> meta,
                           cudf::size_type mess_offset,
                           cudf::size_type mess_count,
                           std::shared_ptr<morpheus::InferenceMemory> memory,
                           cudf::size_type offset,
                           cudf::size_type count) {
                 return std::make_shared<morpheus::MultiInferenceMessage>(
                     std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
             }),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"),
             py::arg("memory"),
             py::arg("offset"),
             py::arg("count"))
        .def_readonly("memory", &morpheus::MultiInferenceMessage::memory)
        .def_readonly("offset", &morpheus::MultiInferenceMessage::offset)
        .def_readonly("count", &morpheus::MultiInferenceMessage::count)
        .def("get_input",
             [&, m](MultiInferenceMessage& self, const std::string& name) {
                 const auto& py_tensor = tensor_to_cupy(self.get_input(name), m);

                 //  Need to get just our portion. TODO(MDD): THis should be handled in get_input
                 py::object sliced = py_tensor[py::make_tuple(
                     py::slice(py::int_(self.offset), py::int_(self.offset + self.count), py::none()),
                     py::slice(py::none(), py::none(), py::none()))];

                 return sliced;
             })
        .def(
            "get_slice",
            [&, m](MultiInferenceMessage& self, int start, int stop) {
                py::object seq_ids = tensor_to_cupy(self.get_input("seq_ids"), m);

                int mess_start = seq_ids[py::make_tuple(start, 0)].attr("item")().cast<int>();
                int mess_stop  = seq_ids[py::make_tuple(stop - 1, 0)].attr("item")().cast<int>() + 1;

                return std::make_shared<morpheus::MultiInferenceMessage>(
                    self.meta, mess_start, mess_stop - mess_start, self.memory, start, stop - start);
            },
            py::return_value_policy::reference_internal);

    py::class_<MultiInferenceNLPMessage, MultiInferenceMessage, std::shared_ptr<MultiInferenceNLPMessage>>(
        m, "MultiInferenceNLPMessage")
        .def(py::init<>([](std::shared_ptr<morpheus::MessageMeta> meta,
                           cudf::size_type mess_offset,
                           cudf::size_type mess_count,
                           std::shared_ptr<morpheus::InferenceMemory> memory,
                           cudf::size_type offset,
                           cudf::size_type count) {
                 return std::make_shared<MultiInferenceNLPMessage>(
                     std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
             }),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"),
             py::arg("memory"),
             py::arg("offset"),
             py::arg("count"))
        .def_readonly("memory", &MultiInferenceNLPMessage::memory)
        .def_readonly("offset", &MultiInferenceNLPMessage::offset)
        .def_readonly("count", &MultiInferenceNLPMessage::count);

    py::class_<MultiInferenceFILMessage, MultiInferenceMessage, std::shared_ptr<MultiInferenceFILMessage>>(
        m, "MultiInferenceFILMessage")
        .def(py::init<>([](std::shared_ptr<morpheus::MessageMeta> meta,
                           cudf::size_type mess_offset,
                           cudf::size_type mess_count,
                           std::shared_ptr<morpheus::InferenceMemory> memory,
                           cudf::size_type offset,
                           cudf::size_type count) {
                 return std::make_shared<MultiInferenceFILMessage>(
                     std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
             }),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"),
             py::arg("memory"),
             py::arg("offset"),
             py::arg("count"))
        .def_readonly("memory", &MultiInferenceFILMessage::memory)
        .def_readonly("offset", &MultiInferenceFILMessage::offset)
        .def_readonly("count", &MultiInferenceFILMessage::count);

    py::class_<morpheus::ResponseMemory, std::shared_ptr<morpheus::ResponseMemory>>(m, "ResponseMemory")
        .def_readonly("count", &morpheus::ResponseMemory::count);

    py::class_<ResponseMemoryProbs, ResponseMemory, std::shared_ptr<ResponseMemoryProbs>>(m, "ResponseMemoryProbs")
        .def(py::init<>([](cudf::size_type count, py::object probs) {
                 // Conver the cupy arrays to tensors
                 return std::make_shared<ResponseMemoryProbs>(count, std::move(cupy_to_tensor(probs)));
             }),
             py::arg("count"),
             py::arg("probs"))
        .def_readonly("count", &ResponseMemoryProbs::count)
        .def_property(
            "probs",
            [&, m](ResponseMemoryProbs& self) { return tensor_to_cupy(self.get_probs(), m); },
            [](ResponseMemoryProbs& self, py::object cupy_value) {
                return self.set_probs(cupy_to_tensor(cupy_value));
            });

    neo::EdgeConnector<std::shared_ptr<morpheus::MultiResponseMessage>, py::object>::register_converter();
    neo::EdgeConnector<py::object, std::shared_ptr<morpheus::MultiResponseMessage>>::register_converter();

    py::class_<morpheus::MultiResponseMessage, morpheus::MultiMessage, std::shared_ptr<morpheus::MultiResponseMessage>>(
        m, "MultiResponseMessage")
        .def(py::init<>([](std::shared_ptr<morpheus::MessageMeta> meta,
                           cudf::size_type mess_offset,
                           cudf::size_type mess_count,
                           std::shared_ptr<morpheus::ResponseMemory> memory,
                           cudf::size_type offset,
                           cudf::size_type count) {
                 return std::make_shared<morpheus::MultiResponseMessage>(
                     std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
             }),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"),
             py::arg("memory"),
             py::arg("offset"),
             py::arg("count"))
        .def_readonly("memory", &morpheus::MultiResponseMessage::memory)
        .def_readonly("offset", &morpheus::MultiResponseMessage::offset)
        .def_readonly("count", &morpheus::MultiResponseMessage::count)
        .def("get_output", [&, m](MultiResponseMessage& self, const std::string& name) {
            auto tensor = self.get_output(name);

            return tensor_to_cupy(tensor, m);
        });

    py::class_<MultiResponseProbsMessage, MultiResponseMessage, std::shared_ptr<MultiResponseProbsMessage>>(
        m, "MultiResponseProbsMessage")
        .def(py::init<>([](std::shared_ptr<morpheus::MessageMeta> meta,
                           cudf::size_type mess_offset,
                           cudf::size_type mess_count,
                           std::shared_ptr<morpheus::ResponseMemory> memory,
                           cudf::size_type offset,
                           cudf::size_type count) {
                 return std::make_shared<MultiResponseProbsMessage>(
                     std::move(meta), mess_offset, mess_count, std::move(memory), offset, count);
             }),
             py::arg("meta"),
             py::arg("mess_offset"),
             py::arg("mess_count"),
             py::arg("memory"),
             py::arg("offset"),
             py::arg("count"))
        .def_readonly("memory", &MultiResponseProbsMessage::memory)
        .def_readonly("offset", &MultiResponseProbsMessage::offset)
        .def_readonly("count", &MultiResponseProbsMessage::count);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace morpheus
