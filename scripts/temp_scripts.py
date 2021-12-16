# SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cudf
import json
import pandas as pd
import cupy as cp
import numpy as np

df = cudf.read_csv("models/training-tuning-scripts/sid-models/resources/sid_sample_training_data.csv")

df.rename(columns={"text": "data"}, inplace=True)
df.rename(columns={
    "address": "si_address",
    "bank_acct": "si_bank_acct",
    "credit_card": "si_credit_card",
    "email": "si_email",
    "govt_id": "si_govt_id",
    "name": "si_name",
    "password": "si_password",
    "phone_num": "si_phone_num",
    "secret_keys": "si_secret_keys",
    "user": "si_user"
},
          inplace=True)

df = df.to_pandas()

df.to_csv("data/sid_training_data_truth.csv", index_label="ID")


def remove_bars(y: str):
    return y[1:-1]


df["data"] = df["data"].apply(remove_bars)


def double_serialize(y: str):
    try:
        return json.dumps(json.dumps(json.loads(y)))
    except:
        return y


df["data"] = df["data"].apply(double_serialize)

df["data"] = df["data"].str.replace('\\\\n', '\\\n', regex=False)

output_strs = [json.dumps(y) for y in df.to_dict(orient="records")]

with open("data/pcap_training_data-truth.jsonlines", "w") as f:
    f.writelines("\n".join(output_strs))
    f.write("\n")

# Now just the input
output_strs = [json.dumps(y) for y in df[["data"]].to_dict(orient="records")]

with open("data/pcap_training_data.jsonlines", "w") as f:
    f.writelines("\n".join(output_strs))
    f.write("\n")

cp.set_printoptions(threshold=cp.inf)

# Converting from Rachels data
df = cudf.read_json("data/merged_file-sorted.jsonlines", lines=True)

df = df.to_pandas()

df = df.sort_values(by=['timestamp', "data_len"])

df.data_len.value_counts(dropna=False)

df = df.drop(["any_pii"], axis=1)

si_cols = [
    "si_address",
    "si_bank_acct",
    "si_credit_card",
    "si_email",
    "si_govt_id",
    "si_name",
    "si_password",
    "si_phone_num",
    "si_secret_keys",
    "si_user"
]

df[si_cols] = df[si_cols].fillna(0.0)

df = df.astype({"data_len": "str", "protocol": "str", "src_port": "str", "dest_port": "str", "flags": "str"})
df = df.astype({
    "si_address": "bool",
    "si_bank_acct": "bool",
    "si_credit_card": "bool",
    "si_email": "bool",
    "si_govt_id": "bool",
    "si_name": "bool",
    "si_password": "bool",
    "si_phone_num": "bool",
    "si_secret_keys": "bool",
    "si_user": "bool",
})


def double_serialize(y: str):
    try:
        return json.dumps(json.dumps(json.loads(y[1:-1])))
    except:
        return y


df["data"] = df["data"].str.replace('\\n', '\n', regex=False)
df["data"] = df["data"].str.replace('\/', '/', regex=False)
df["data"] = df["data"].str.replace('0\r\n\r\n', '\"0\"', regex=False)

output_strs = [json.dumps(y) for y in df.to_dict(orient="records")]

with open("data/merged_file-fixed.jsonlines", "w") as f:
    f.writelines("\n".join(output_strs))
    f.write("\n")

with open("data/merged_file-sorted.jsonlines", "w") as f:
    f.writelines("\n".join(output_strs))
    f.write("\n")

pos_detect = {c: 93085 - df[c].value_counts()[False] for c in si_cols}

# Fixing Rachel Truth data
df = cudf.read_json(".tmp/pcap_training_data-truth-bert-base.jsonlines", lines=True)

df = df.to_pandas()

df = df.drop([
    "timestamp",
    "host_ip",
    "data_len",
    "src_mac",
    "dest_mac",
    "protocol",
    "src_ip",
    "dest_ip",
    "src_port",
    "dest_port",
    "flags"
],
             axis=1)


def double_serialize(y: str):
    try:
        return json.dumps(json.dumps(json.loads(y)))
    except:
        return y


# df["data"] = df["data"].apply(double_serialize)

df["data"] = df["data"].str.replace('\\\\n', '\\\n', regex=False)

# Now just the input
output_strs = [json.dumps(y) for y in df.to_dict(orient="records")]

with open(".tmp/pcap_training_data-truth-bert-base-fixed.jsonlines", "w") as f:
    f.writelines("\n".join(output_strs))
    f.write("\n")
