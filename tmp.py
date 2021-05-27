import json
import cudf

full_df = cudf.read_json("/home/mdemoret/Repos/rapids/cyber-dev/.tmp/dataset4/pcap_dump_augmented_0delay.json", lines=True, engine="cudf")

if ("data" in full_df):
    full_df["data"] = full_df["data"].str.replace('\\n', '\n', regex=False)

full_df = full_df.to_pandas()

with_len_ds = full_df[0:1000]
without_len_ds = full_df[1000:2000].drop(columns="data_len")

def write_to_file(df: cudf.DataFrame, filename: str):
    def double_serialize(y: str):
        try:
            return json.dumps(json.dumps(json.loads(y)))
        except:
            return y

    # Special processing for the data column (need to double serialize to match input)
    # if ("data" in df):
    #     df["data"] = df["data"].apply(double_serialize)

    # Convert to list of json string objects
    output_strs = [json.dumps(y) for y in df.to_dict(orient="records")]

    with open(filename, "w") as f:
        f.writelines("\n".join(output_strs))
        f.write("\n")

write_to_file(with_len_ds, "/home/mdemoret/Repos/rapids/cyber-dev/examples/multi_in_multi_out/with_data_len.json")
write_to_file(without_len_ds, "/home/mdemoret/Repos/rapids/cyber-dev/examples/multi_in_multi_out/without_data_len.json")