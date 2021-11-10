#include "morpheus/common.hpp"

#include <pybind11/pybind11.h>
#include <memory>
#include <vector>

// namespace neo = trtlab::neo;
namespace py  = pybind11;

namespace morpheus {

// Define the pybind11 module m, as 'pipeline'.
PYBIND11_MODULE(common, m)
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

    py::class_<neo::TensorObject>(m, "Tensor").def_property_readonly("__cuda_array_interface__", [](neo::TensorObject& self) {
        py::dict array_interface;

        py::list shape_list = py::cast(self.get_shape());

        py::int_ stream_val = 1;

        // See https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
        // if (self.get_stream().is_default())
        // {
        //     stream_val = 1;
        // }
        // else if (self.get_stream().is_per_thread_default())
        // {
        //     stream_val = 2;
        // }
        // else
        // {
        //     // Custom stream. Return value
        //     stream_val = (int64_t)self.get_stream().value();
        // }

        array_interface["shape"]   = py::cast<py::tuple>(shape_list);
        array_interface["typestr"] = self.get_numpy_typestr();
        array_interface["stream"]  = stream_val;
        array_interface["version"] = 3;

        if (self.get_stride().empty())
        {
            array_interface["strides"] = py::none();
        }
        else
        {
            array_interface["strides"] = py::cast<py::tuple>(py::cast(self.get_stride()));
        }
        array_interface["data"] = py::make_tuple((uintptr_t)self.data(), false);

        return array_interface;
    });


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}
