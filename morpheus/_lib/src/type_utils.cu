#include "morpheus/type_utils.hpp"

#include <memory>
#include <stdexcept>
#include <type_traits>

namespace morpheus {

DType::DType(const trtlab::neo::DataType& dtype) : trtlab::neo::DataType(dtype.type_id()) {}
DType::DType(trtlab::neo::TypeId tid) : trtlab::neo::DataType(tid) {}

// Cudf representation
cudf::type_id DType::cudf_type_id() const
{
    switch (m_type_id)
    {
    case trtlab::neo::TypeId::INT8:
        return cudf::type_id::INT8;
    case trtlab::neo::TypeId::INT16:
        return cudf::type_id::INT16;
    case trtlab::neo::TypeId::INT32:
        return cudf::type_id::INT32;
    case trtlab::neo::TypeId::INT64:
        return cudf::type_id::INT64;
    case trtlab::neo::TypeId::UINT8:
        return cudf::type_id::UINT8;
    case trtlab::neo::TypeId::UINT16:
        return cudf::type_id::UINT16;
    case trtlab::neo::TypeId::UINT32:
        return cudf::type_id::UINT32;
    case trtlab::neo::TypeId::UINT64:
        return cudf::type_id::UINT64;
    case trtlab::neo::TypeId::FLOAT32:
        return cudf::type_id::FLOAT32;
    case trtlab::neo::TypeId::FLOAT64:
        return cudf::type_id::FLOAT64;
    case trtlab::neo::TypeId::BOOL8:
        return cudf::type_id::BOOL8;
    case trtlab::neo::TypeId::EMPTY:
    case trtlab::neo::TypeId::NUM_TYPE_IDS:
    default:
        throw std::runtime_error("Not supported");
    }
}

// Returns the triton string representation
std::string DType::triton_str() const
{
    // Triton doesnt have any definitions or enums. Wow
    switch (m_type_id)
    {
    case trtlab::neo::TypeId::INT8:
        return "INT8";
    case trtlab::neo::TypeId::INT16:
        return "INT16";
    case trtlab::neo::TypeId::INT32:
        return "INT32";
    case trtlab::neo::TypeId::INT64:
        return "INT64";
    case trtlab::neo::TypeId::UINT8:
        return "UINT8";
    case trtlab::neo::TypeId::UINT16:
        return "UINT16";
    case trtlab::neo::TypeId::UINT32:
        return "UINT32";
    case trtlab::neo::TypeId::UINT64:
        return "UINT64";
    case trtlab::neo::TypeId::FLOAT32:
        return "FP32";
    case trtlab::neo::TypeId::FLOAT64:
        return "FP64";
    case trtlab::neo::TypeId::BOOL8:
        return "BOOL";
    case trtlab::neo::TypeId::EMPTY:
    case trtlab::neo::TypeId::NUM_TYPE_IDS:
    default:
        throw std::runtime_error("Not supported");
    }
}

// From cudf
DType DType::from_cudf(cudf::type_id tid)
{
    switch (tid)
    {
    case cudf::type_id::INT8:
        return DType(trtlab::neo::TypeId::INT8);
    case cudf::type_id::INT16:
        return DType(trtlab::neo::TypeId::INT16);
    case cudf::type_id::INT32:
        return DType(trtlab::neo::TypeId::INT32);
    case cudf::type_id::INT64:
        return DType(trtlab::neo::TypeId::INT64);
    case cudf::type_id::UINT8:
        return DType(trtlab::neo::TypeId::UINT8);
    case cudf::type_id::UINT16:
        return DType(trtlab::neo::TypeId::UINT16);
    case cudf::type_id::UINT32:
        return DType(trtlab::neo::TypeId::UINT32);
    case cudf::type_id::UINT64:
        return DType(trtlab::neo::TypeId::UINT64);
    case cudf::type_id::FLOAT32:
        return DType(trtlab::neo::TypeId::FLOAT32);
    case cudf::type_id::FLOAT64:
        return DType(trtlab::neo::TypeId::FLOAT64);
    case cudf::type_id::BOOL8:
        return DType(trtlab::neo::TypeId::BOOL8);
    case cudf::type_id::EMPTY:
    case cudf::type_id::NUM_TYPE_IDS:
    default:
        throw std::runtime_error("Not supported");
    }
}

// From triton
DType DType::from_triton(const std::string& type_str)
{
    if (type_str == "INT8")
    {
        return DType(trtlab::neo::TypeId::INT8);
    }
    else if (type_str == "INT16")
    {
        return DType(trtlab::neo::TypeId::INT16);
    }
    else if (type_str == "INT32")
    {
        return DType(trtlab::neo::TypeId::INT32);
    }
    else if (type_str == "INT64")
    {
        return DType(trtlab::neo::TypeId::INT64);
    }
    else if (type_str == "UINT8")
    {
        return DType(trtlab::neo::TypeId::UINT8);
    }
    else if (type_str == "UINT16")
    {
        return DType(trtlab::neo::TypeId::UINT16);
    }
    else if (type_str == "UINT32")
    {
        return DType(trtlab::neo::TypeId::UINT32);
    }
    else if (type_str == "UINT64")
    {
        return DType(trtlab::neo::TypeId::UINT64);
    }
    else if (type_str == "FP32")
    {
        return DType(trtlab::neo::TypeId::FLOAT32);
    }
    else if (type_str == "FP64")
    {
        return DType(trtlab::neo::TypeId::FLOAT64);
    }
    else if (type_str == "BOOL")
    {
        return DType(trtlab::neo::TypeId::BOOL8);
    }
    else
    {
        throw std::runtime_error("Not supported");
    }
}

}  // namespace morpheus
