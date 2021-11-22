#include "morpheus/common.hpp"

#include <boost/fiber/buffered_channel.hpp>
#include <boost/fiber/channel_op_status.hpp>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/gil.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pyerrors.h>
#include "glog/logging.h"
#include "morpheus/cudf_helpers.hpp"
#include "pyneo/utils.hpp"

namespace morpheus {

class FiberQueue
{
  public:
    FiberQueue(size_t max_size) : m_queue(max_size) {}

    boost::fibers::channel_op_status put(py::object&& item, bool block = true, float timeout = 0.0)
    {
        if (!block)
        {
            return m_queue.try_push(std::move(item));
        }
        else if (timeout > 0.0)
        {
            return m_queue.push_wait_for(
                std::move(item),
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::duration<float>(timeout)));
        }
        else
        {
            // Blocking no timeout
            return m_queue.push(std::move(item));
        }
    }

    boost::fibers::channel_op_status get(py::object& item, bool block = true, float timeout = 0.0)
    {
        if (!block)
        {
            return m_queue.try_pop(std::ref(item));
        }
        else if (timeout > 0.0)
        {
            return m_queue.pop_wait_for(
                std::ref(item),
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::duration<float>(timeout)));
        }
        else
        {
            // Blocking no timeout
            return m_queue.pop(std::ref(item));
        }
    }

    void close()
    {
        m_queue.close();
    }

    bool is_closed()
    {
        return m_queue.is_closed();
    }

    void join()
    {
        // TODO(MDD): Not sure how to join a buffered channel
    }

  private:
    boost::fibers::buffered_channel<py::object> m_queue;
};

// Define the pybind11 module m, as 'pipeline'.
PYBIND11_MODULE(common, m)
{
    google::InitGoogleLogging("morpheus");

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

    py::class_<neo::TensorObject>(m, "Tensor")
        .def_property_readonly("__cuda_array_interface__", [](neo::TensorObject& self) {
            py::dict array_interface;

            py::list shape_list;

            for (auto& idx : self.get_shape())
            {
                shape_list.append(idx);
            }

            py::list stride_list;

            for (auto& idx : self.get_stride())
            {
                stride_list.append(idx * self.dtype_size());
            }

            // py::list shape_list = py::cast(self.get_shape());

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

            if (self.is_compact() || self.get_stride().empty())
            {
                array_interface["strides"] = py::none();
            }
            else
            {
                array_interface["strides"] = py::cast<py::tuple>(stride_list);
            }
            array_interface["data"] = py::make_tuple((uintptr_t)self.data(), false);

            return array_interface;
        });

    py::class_<FiberQueue, std::shared_ptr<FiberQueue>>(m, "FiberQueue")
        .def(py::init<>([](size_t max_size) {
                 if (max_size < 2 || ((max_size & (max_size - 1)) != 0))
                 {
                     throw std::invalid_argument("max_size must be greater than 1 and a power of 2.");
                 }

                 // Create a new shared_ptr
                 return std::make_shared<FiberQueue>(max_size);
             }),
             py::arg("max_size"))
        .def(
            "put",
            [](FiberQueue& self, py::object item, bool block = true, float timeout = 0.0) {
                boost::fibers::channel_op_status status;

                // Release the GIL and try to move it
                {
                    py::gil_scoped_release nogil;

                    status = self.put(std::move(item), block, timeout);
                }

                switch (status)
                {
                case boost::fibers::channel_op_status::success:
                    return;
                case boost::fibers::channel_op_status::empty: {
                    // Raise queue.Empty
                    py::object exc_class = py::module_::import("queue").attr("Empty");

                    PyErr_SetNone(exc_class.ptr());

                    throw py::error_already_set();
                }
                case boost::fibers::channel_op_status::full:
                case boost::fibers::channel_op_status::timeout: {
                    // Raise queue.Full
                    py::object exc_class = py::module_::import("queue").attr("Empty");

                    PyErr_SetNone(exc_class.ptr());

                    throw py::error_already_set();
                }
                case boost::fibers::channel_op_status::closed: {
                    // Raise queue.Full
                    py::object exc_class = py::module_::import("morpheus.utils.producer_consumer_queue").attr("Closed");

                    PyErr_SetNone(exc_class.ptr());

                    throw py::error_already_set();
                }
                }
            },
            py::arg("item"),
            py::arg("block")   = true,
            py::arg("timeout") = 0.0)
        .def(
            "get",
            [](FiberQueue& self, bool block = true, float timeout = 0.0) {
                boost::fibers::channel_op_status status;

                py::object item;

                // Release the GIL and try to move it
                {
                    py::gil_scoped_release nogil;

                    status = self.get(std::ref(item), block, timeout);
                }

                switch (status)
                {
                case boost::fibers::channel_op_status::success:
                    return item;
                case boost::fibers::channel_op_status::empty: {
                    // Raise queue.Empty
                    py::object exc_class = py::module_::import("queue").attr("Empty");

                    PyErr_SetNone(exc_class.ptr());

                    throw py::error_already_set();
                }
                case boost::fibers::channel_op_status::full:
                case boost::fibers::channel_op_status::timeout: {
                    // Raise queue.Full
                    py::object exc_class = py::module_::import("queue").attr("Empty");

                    PyErr_SetNone(exc_class.ptr());

                    throw py::error_already_set();
                }
                case boost::fibers::channel_op_status::closed: {
                    // Raise queue.Full
                    py::object exc_class = py::module_::import("morpheus.utils.producer_consumer_queue").attr("Closed");

                    PyErr_SetNone(exc_class.ptr());

                    throw py::error_already_set();
                }
                default:
                    throw std::runtime_error("Unknown channel status");
                }
            },
            py::arg("block")   = true,
            py::arg("timeout") = 0.0)
        .def("close", [](FiberQueue& self) {
            // Close
            self.close();
        });

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
}  // namespace morpheus
