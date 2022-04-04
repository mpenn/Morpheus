/**
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <algorithm>

namespace morpheus {
    /****** Component public free function implementations************/
    /**
     * TODO(Documentation)
     */
    template<typename FuncT, typename SeqT>
    auto foreach_map(const SeqT &seq, FuncT func) {
        using value_t = typename SeqT::const_reference;
        using return_t = decltype(func(std::declval<value_t>()));

        std::vector<return_t> result{};

        std::transform(seq.cbegin(), seq.cend(), std::back_inserter(result), func);

        return result;
    }

    /**
     * TODO(Documentation)
     */
    template<typename FuncT, typename SeqT>
    auto foreach_map2(const SeqT &seq, FuncT func) {
        using value_t = typename SeqT::const_reference;
        using return_t = decltype(func(std::declval<value_t>()));

        std::vector<return_t> result{};

        std::transform(seq.begin(), seq.end(), std::back_inserter(result), func);

        return result;
    }
}  // namespace morpheus
