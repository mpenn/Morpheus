#pragma once
#include <cudf/io/types.hpp>
#include <cudf/table/table_view.hpp>
#include <utility>

namespace morpheus {

struct TableInfo
{
    TableInfo() = default;
    TableInfo(cudf::table_view view, cudf::io::table_metadata meta) :
      table_view(std::move(view)),
      metadata(std::move(meta))
    {}

    cudf::table_view table_view;
    cudf::io::table_metadata metadata;
};

}  // namespace morpheus
