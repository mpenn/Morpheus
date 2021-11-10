#pragma once

#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rmm/device_buffer.hpp>
#include <trtlab/neo/core/tensor.hpp>
#include "pyneo/node.hpp"

namespace morpheus {

namespace neo = trtlab::neo;
namespace py  = pybind11;

class RMMTensor : public neo::ITensor
{
  public:
    RMMTensor(std::shared_ptr<rmm::device_buffer> device_buffer,
              size_t offset,
              std::string typestr,
              size_t dtype_size,
              std::vector<neo::TensorIndex> shape,
              std::vector<neo::TensorIndex> stride = {}) :
      m_md(std::move(device_buffer)),
      m_offset(offset),
      m_typestr(std::move(typestr)),
      m_dtype_size(dtype_size),
      m_shape(std::move(shape)),
      m_stride(std::move(stride))
    {
        if (m_stride.empty())
        {
            trtlab::neo::detail::validate_stride(this->m_shape, this->m_stride);
        }
    }
    ~RMMTensor() = default;

    std::shared_ptr<neo::MemoryDescriptor> get_memory() const override
    {
        return nullptr;
    }

    void* data() const override
    {
        return static_cast<uint8_t*>(m_md->data()) + m_offset;
    }

    neo::RankType rank() const final
    {
        return m_shape.size();
    }

    std::size_t dtype_size() const final
    {
        return m_dtype_size;
    }

    std::string typestr() const override
    {
        return m_typestr;
    }

    std::size_t count() const final
    {
        return std::accumulate(m_shape.begin(), m_shape.end(), 1, std::multiplies<>());
    }

    std::size_t bytes() const final
    {
        return count() * dtype_size();
    }

    neo::TensorIndex shape(std::uint32_t idx) const final
    {
        DCHECK_LT(idx, m_shape.size());
        return m_shape.at(idx);
    }

    neo::TensorIndex stride(std::uint32_t idx) const final
    {
        DCHECK_LT(idx, m_stride.size());
        return m_stride.at(idx);
    }

    void get_shape(std::vector<neo::TensorIndex>& s) const final
    {
        s.resize(rank());
        std::copy(m_shape.begin(), m_shape.end(), s.begin());
    }

    void get_stride(std::vector<neo::TensorIndex>& s) const final
    {
        s.resize(rank());
        std::copy(m_stride.begin(), m_stride.end(), s.begin());
    }

    bool is_compact() const final
    {
        neo::TensorIndex ttl = 1;
        for (int i = rank() - 1; i >= 0; i--)
        {
            if (stride(i) != ttl)
            {
                return false;
            }

            ttl *= shape(i);
        }
        return true;
    }

    std::shared_ptr<neo::ITensor> slice(const std::vector<neo::TensorIndex>& min_dims,
                                        const std::vector<neo::TensorIndex>& max_dims) const override
    {
        // Calc new offset
        size_t offset = std::transform_reduce(
            m_stride.begin(), m_stride.end(), min_dims.begin(), m_offset, std::plus<>(), std::multiplies<>());

        // Calc new shape
        std::vector<neo::TensorIndex> shape;
        std::transform(max_dims.begin(), max_dims.end(), min_dims.begin(), std::back_inserter(shape), std::minus<>());

        // Stride remains the same

        return std::make_shared<RMMTensor>(m_md, offset, m_typestr, m_dtype_size, shape, m_stride);
    }

    std::shared_ptr<neo::ITensor> reshape(const std::vector<neo::TensorIndex>& dims) const override
    {
        return std::make_shared<RMMTensor>(m_md, 0, m_typestr, m_dtype_size, dims, m_stride);
    }

    std::shared_ptr<neo::ITensor> deep_copy() const override
    {
        // Deep copy
        std::shared_ptr<rmm::device_buffer> copied_buffer =
            std::make_shared<rmm::device_buffer>(*m_md, m_md->stream(), m_md->memory_resource());

        return std::make_shared<RMMTensor>(copied_buffer, m_offset, m_typestr, m_dtype_size, m_shape, m_stride);
    }

    // Tensor reshape(std::vector<neo::TensorIndex> shape)
    // {
    //     CHECK(is_compact());
    //     return Tensor(descriptor_shared(), dtype_size(), shape);
    // }

  protected:
  private:
    // Memory info
    std::shared_ptr<rmm::device_buffer> m_md;
    size_t m_offset;

    // Type info
    std::string m_typestr;
    std::size_t m_dtype_size;

    // Shape info
    std::vector<neo::TensorIndex> m_shape;
    std::vector<neo::TensorIndex> m_stride;
};

class Tensor
{
  public:
    Tensor(std::shared_ptr<rmm::device_buffer> buffer,
           std::string init_typestr,
           std::vector<int32_t> init_shape,
           std::vector<int32_t> init_strides,
           size_t init_offset = 0) :
      m_device_buffer(std::move(buffer)),
      typestr(std::move(init_typestr)),
      shape(std::move(init_shape)),
      strides(std::move(init_strides)),
      m_offset(init_offset)
    {}

    std::vector<int32_t> shape;
    std::vector<int32_t> strides;
    std::string typestr;

    void* data() const
    {
        return static_cast<uint8_t*>(m_device_buffer->data()) + m_offset;
    }

    size_t bytes_count() const
    {
        // temp just return without shape, size, offset, etc
        return m_device_buffer->size();
    }

    std::vector<uint8_t> get_host_data() const
    {
        std::vector<uint8_t> out_data;

        out_data.resize(this->bytes_count());

        cudaMemcpy(&out_data[0], this->data(), this->bytes_count(), cudaMemcpyDeviceToHost);

        return out_data;
    }

    auto get_stream() const
    {
        return this->m_device_buffer->stream();
    }

    static neo::TensorObject create(std::shared_ptr<rmm::device_buffer> buffer,
                                    std::string init_typestr,
                                    std::vector<neo::TensorIndex> init_shape,
                                    std::vector<neo::TensorIndex> init_strides,
                                    size_t init_offset = 0)
    {
        auto md = nullptr;

        auto tensor = std::make_shared<RMMTensor>(buffer,
                                                  init_offset,
                                                  std::string(1, init_typestr[1]),
                                                  std::stoi(std::string(1, init_typestr[2])),
                                                  init_shape,
                                                  init_strides);

        return neo::TensorObject(md, tensor);
    }

  private:
    size_t m_offset;
    std::shared_ptr<rmm::device_buffer> m_device_buffer;
};

// Before using this, cupy must be loaded into the module with `pyneo::import(m, "cupy")`
py::object tensor_to_cupy(const neo::TensorObject& tensor, const py::module_& mod)
{
    // Get the cupy creation function
    auto array = mod.attr("cupy").attr("array");
    // Cast to a py::object
    auto py_tensor = py::cast(tensor);

    return array(py_tensor);
}

neo::TensorObject cupy_to_tensor(py::object cupy_array)
{
    // Convert inputs from cupy to Tensor
    py::dict arr_interface = cupy_array.attr("__cuda_array_interface__");

    py::tuple shape_tup = arr_interface["shape"];

    auto shape = shape_tup.cast<std::vector<neo::TensorIndex>>();

    std::string typestr = arr_interface["typestr"].cast<std::string>();

    py::tuple data_tup = arr_interface["data"];

    uintptr_t data_ptr = data_tup[0].cast<uintptr_t>();

    std::vector<neo::TensorIndex> strides{};

    if (arr_interface.contains("strides") && !arr_interface["strides"].is_none())
    {
        py::tuple strides_tup = arr_interface["strides"];

        strides = strides_tup.cast<std::vector<neo::TensorIndex>>();
    }

    //  Get the size finally
    auto size = cupy_array.attr("data").attr("mem").attr("size").cast<size_t>();

    auto tensor =
        Tensor::create(std::make_shared<rmm::device_buffer>((void const*)data_ptr, size, rmm::cuda_stream_per_thread),
                       typestr,
                       shape,
                       strides,
                       0);

    return tensor;
}

}  // namespace morpheus
