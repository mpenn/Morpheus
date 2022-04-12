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
## GNN Fraud Detection Pipeline

Use Morpheus to run the GNN fraud detection Pipeline with the transaction data. A pipeline has been configured in `run.py` with several command line options:

```bash
python run.py --help
Usage: run.py [OPTIONS]

Options:
  --num_threads INTEGER RANGE     Number of internal pipeline threads to use
  --pipeline_batch_size INTEGER RANGE
                                  Internal batch size for the pipeline. Can be
                                  much larger than the model batch size. Also
                                  used for Kafka consumers

  --model_max_batch_size INTEGER RANGE
                                  Max batch size to use for the model
  --input_file PATH               Input filepath  [required]
  --output_file TEXT              The path to the file where the inference
                                  output will be saved.
  --training_file PATH            Training data file [required]
  --model_fea_length INTEGER RANGE
                                  Features length to use for the model
  --model-xgb-file PATH           The name of the XGB model that is deployed
  --model-hinsage-file PATH       The name of the trained HinSAGE model file path

  --help                          Show this message and exit.
```

To launch the configured Morpheus pipeline with the sample data that is provided at `<MORPHEUS_ROOT>/models/dataset`, run the following:

```bash

python run.py 
====Building Pipeline====
Added source: <from-file-0; FileSourceStage(filename=validation.csv, iterative=None, file_type=auto, repeat=1, filter_null=False, cudf_kwargs=None)>
  └─> morpheus.MessageMeta
Added stage: <deserialize-1; DeserializeStage()>
  └─ morpheus.MessageMeta -> morpheus.MultiMessage
Added stage: <fraud-graph-construction-2; FraudGraphConstructionStage(training_file=training.csv)>
  └─ morpheus.MultiMessage -> stages.FraudGraphMultiMessage
Added stage: <monitor-3; MonitorStage(description=Graph construction rate, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None)>
  └─ stages.FraudGraphMultiMessage -> stages.FraudGraphMultiMessage
Added stage: <gnn-fraud-sage-4; GraphSAGEStage(model_hinsage_file=model/hinsage-model.pt, batch_size=5, sample_size=[2, 32], record_id=index, target_node=transaction)>
  └─ stages.FraudGraphMultiMessage -> stages.GraphSAGEMultiMessage
Added stage: <monitor-5; MonitorStage(description=Inference rate, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None)>
  └─ stages.GraphSAGEMultiMessage -> stages.GraphSAGEMultiMessage
Added stage: <gnn-fraud-classification-6; ClassificationStage(model_xgb_file=model/xgb-model.pt)>
  └─ stages.GraphSAGEMultiMessage -> morpheus.MultiMessage
Added stage: <monitor-7; MonitorStage(description=Add classification rate, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None)>
  └─ morpheus.MultiMessage -> morpheus.MultiMessage
Added stage: <serialize-8; SerializeStage(include=None, exclude=['^ID$', '^_ts_'], output_type=pandas)>
  └─ morpheus.MultiMessage -> pandas.DataFrame
Added stage: <monitor-9; MonitorStage(description=Serialize rate, smoothing=0.05, unit=messages, delayed_start=False, determine_count_fn=None)>
  └─ pandas.DataFrame -> pandas.DataFrame
Added stage: <to-file-10; WriteToFileStage(filename=result.csv, overwrite=True, file_type=auto)>
  └─ pandas.DataFrame -> pandas.DataFrame
====Building Pipeline Complete!====
====Pipeline Started====
Graph construction rate[Complete]: 265messages [00:00, 1590.22messages/s]
Inference rate[Complete]: 265messages [00:01, 150.23messages/s]
Add classification rate[Complete]: 265messages [00:01, 147.11messages/s]
Serialize rate[Complete]: 265messages [00:01, 142.31messages/s]
```
