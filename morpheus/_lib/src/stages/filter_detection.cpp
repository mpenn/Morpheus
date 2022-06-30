/**
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <morpheus/stages/filter_detection.hpp>

#include <morpheus/utilities/matx_util.hpp>

#include <cstddef>
#include <exception>
#include <memory>
#include <mutex>
#include "morpheus/messages/meta.hpp"

namespace morpheus {
// Component public implementations
// ************ FilterDetectionStage **************************** //
FilterDetectionsStage::FilterDetectionsStage(float threshold) :
  PythonNode(base_t::op_factory_from_sub_fn(build_operator())),
  m_threshold(threshold)
{}

FilterDetectionsStage::subscribe_fn_t FilterDetectionsStage::build_operator()
{
    return [this](rxcpp::observable<sink_type_t> input, rxcpp::subscriber<source_type_t> output) {
        return input.subscribe(rxcpp::make_observer<sink_type_t>(
            [this, &output](sink_type_t x) {
                const auto &probs  = x->get_probs();
                const auto &shape  = probs.get_shape();
                const auto &stride = probs.get_stride();

                CHECK(probs.rank() == 2)
                    << "C++ impl of the FilterDetectionsStage currently only supports two dimensional arrays";

                const std::size_t num_rows    = shape[0];
                const std::size_t num_columns = shape[1];

                // A bit ugly, but we cant get access to the rmm::device_buffer here. So make a copy
                auto tmp_buffer = std::make_shared<rmm::device_buffer>(probs.count() * probs.dtype_size(),
                                                                       rmm::cuda_stream_per_thread);

                SRF_CHECK_CUDA(
                    cudaMemcpy(tmp_buffer->data(), probs.data(), tmp_buffer->size(), cudaMemcpyDeviceToDevice));

                // Depending on the input the stride is given in bytes or elements,
                // divide the stride elements by the smallest item to ensure tensor_stride is defined in
                // terms of elements
                std::vector<TensorIndex> tensor_stride(stride.size());
                auto min_stride = std::min_element(stride.cbegin(), stride.cend());

                std::transform(stride.cbegin(),
                               stride.cend(),
                               tensor_stride.begin(),
                               std::bind(std::divides<>(), std::placeholders::_1, *min_stride));

                // Now call the threshold function
                auto thresh_bool_buffer =
                    MatxUtil::threshold(DevMemInfo{probs.count(), probs.dtype().type_id(), tmp_buffer, 0},
                                        num_rows,
                                        num_columns,
                                        tensor_stride,
                                        m_threshold,
                                        true);

                std::vector<uint8_t> host_bool_values(num_rows);

                // Copy bools back to host
                SRF_CHECK_CUDA(cudaMemcpy(host_bool_values.data(),
                                          thresh_bool_buffer->data(),
                                          thresh_bool_buffer->size(),
                                          cudaMemcpyDeviceToHost));

                std::cerr << "********** 0" << std::endl << std::flush;
                std::size_t selected_rows = 0;
                auto mask =
                    std::make_shared<rmm::device_buffer>(cudf::create_null_mask(num_rows, cudf::mask_state::ALL_NULL));

                std::cerr << "********** 1" << std::endl << std::flush;
                // We are slicing by rows, using num_rows as our marker for undefined
                std::size_t slice_start = num_rows;
                for (std::size_t row = 0; row < num_rows; ++row)
                {
                    bool above_threshold = host_bool_values[row];

                    if (above_threshold && slice_start == num_rows)
                    {
                        slice_start = row;
                    }
                    else if (!above_threshold && slice_start != num_rows)
                    {
                        cudf::set_null_mask(static_cast<cudf::bitmask_type *>(mask->data()), slice_start, row, true);
                        std::cerr << "********** 1.5 [" << slice_start << ":" << row << "]" << std::endl << std::flush;
                        selected_rows += (row - slice_start);
                        slice_start = num_rows;
                    }
                }

                if (slice_start != num_rows)
                {
                    // Last row was above the threshold
                    cudf::set_null_mask(static_cast<cudf::bitmask_type *>(mask->data()), slice_start, num_rows, true);
                    selected_rows += (num_rows - slice_start);
                    std::cerr << "********** 1.6 [" << slice_start << ":" << num_rows << "]" << std::endl << std::flush;
                }

                auto table = x->get_meta();
                std::cerr << "********** 2 selected_rows= " << selected_rows << std::endl << std::flush;
                auto masked_table = table.apply_mask(mask);
                std::cerr << "********** 2.5 table_rows= " << masked_table.tbl->num_rows() << std::endl << std::flush;
                std::cerr << "********** 3" << std::endl << std::flush;
                auto meta = MessageMeta::create_from_cpp(std::move(masked_table), table.num_indices());
                std::cerr << "********** 3.5: meta=" << meta.get() << std::endl << std::flush;
                std::cerr << "\tcount=" << meta->count() << std::endl << std::flush;

                std::cerr << "********** 4" << std::endl << std::flush;
                std::shared_ptr<ResponseMemory> memory = x->memory;
                std::cerr << "********** 4.5: meta=" << meta.get() << std::endl << std::flush;
                std::cerr << "\tcount=" << meta->count() << std::endl << std::flush;
                std::cerr << "\tmem=" << memory.get() << std::endl << std::flush;
                std::cerr << "\toffset=" << x->offset << "\tcount=" << x->count << std::endl << std::flush;
                auto masked_message =
                    std::make_shared<MultiResponseProbsMessage>(meta, 0, meta->count(), memory, x->offset, x->count);
                std::cerr << "********** 5" << std::endl << std::flush;
                output.on_next(masked_message);
            },
            [&](std::exception_ptr error_ptr) { output.on_error(error_ptr); },
            [&]() { output.on_completed(); }));
    };
}

// ************ FilterDetectionStageInterfaceProxy ************* //
std::shared_ptr<srf::segment::Object<FilterDetectionsStage>> FilterDetectionStageInterfaceProxy::init(
    srf::segment::Builder &builder, const std::string &name, float threshold)
{
    auto stage = builder.construct_object<FilterDetectionsStage>(name, threshold);

    return stage;
}
}  // namespace morpheus
