<!--
SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Multi-Input Multi-Output Morpheus Pipeline Example

## Pipeline Architecture

This example shows how to create a Morpheus pipeline with multiple inputs/outputs and even circular dataflow between stages. The pipeline visualization is:

![MIMO Pipeline](mimo_pipeline.png)

We can see that the `sample-3` stage forks its input into two paths. This is a custom stage that forks message based off whether they have the `data_len` field or not. If the field is missing, the message is routed to `add-data_len-6` which will calculate the `data_len` from the `data` field, reserialize the message, and feed it back into the `deserialize-2` stage.

This has the effect of ensuring the `data_len` field is available. To create messages where some have `data_len` missing, two datasets have been created: `with_data_len.json` and `without_data_len.json`. These datasets both contain 1000 lines and are very similar except for the missing field. Both datasets are loaded and piped into the `deserialze-2` stage where they will be interleaved.

Finally, we illustrate how to use multiple outputs by forking the `monitor-7` stage into two serialization stages. One serialization stage will exclude timestamp information (any field starting with `_ts_*`) and the other will not exclude any fields. This results in two output files that only differ by the fields that were serialized.

## Setup

This example does not require Triton Inference Server. No additional setup is required.




## MIMO Pipeline
Use Morpheus to run the MIMO Pipeline with the following command:

```
python ./examples/multi_in_multi_out/run.py
```

This example does not have any configuration options.
