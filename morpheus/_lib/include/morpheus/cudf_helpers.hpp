#pragma once

#include <glog/logging.h>
#include <pybind11/pytypes.h>
#include <cudf/io/types.hpp>

#include "./table_info.hpp"

// MUST COME LAST!!!
#include "cudf_helpers_api.h"

namespace morpheus {

void load_cudf_helpers()
{
    if (import_morpheus___lib__cudf_helpers() != 0)
    {
        pybind11::error_already_set ex;

        LOG(ERROR) << "Could not load cudf_helpers library: " << ex.what();
        throw ex;
    }
}
}  // namespace morpheus
