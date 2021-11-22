#pragma once

#include <memory>
#include <stdexcept>

#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>

#include <trtlab/neo/util/type_utils.hpp>

namespace rmm {
class device_buffer;
}

namespace morpheus {

struct DType : trtlab::neo::DataType  // NOLINT
{
    DType(const trtlab::neo::DataType& dtype);
    DType(trtlab::neo::TypeId tid);

    // Cudf representation
    cudf::type_id cudf_type_id() const;

    // Returns the triton string representation
    std::string triton_str() const;

    // from template
    template <typename T>
    static DType create()
    {
        return DType(trtlab::neo::DataType::create<T>());
    }

    // From cudf
    static DType from_cudf(cudf::type_id tid);

    // From triton
    static DType from_triton(const std::string& type_str);
};

template <typename T>
DType type_to_dtype()
{
    return DType::from_triton(cudf::type_to_id<T>);
}

}  // namespace morpheus
