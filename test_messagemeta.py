import cupy as cp
import numpy

import cudf

import morpheus._lib.messages as neom
import morpheus._lib.stages as neos

# df = neom.MessageMeta.make_df_from_file("data/sid_training_data_truth.csv")

# print(df)

# meta = neom.MessageMeta.make_from_file("data/sid_training_data_truth.csv")
meta = neos.FileSourceStage.make_from_file("data/sid_training_data_truth.csv")

df = meta.df

print(df)

message = neom.MultiMessage(meta, 2, 10)

print(meta.df.loc[2:2 + 10 - 1, "id"])

print(message.get_meta("id"))

# df["times"] = 5.0
message.set_meta("times", 5.0)

print(df)

print(message.get_meta("times"))

message2 = message.get_slice(2, 4)

message2.set_meta("times", 3.0)

print(df)

print(message.get_meta("times"))

# memory = neom.ResponseMemoryProbs(count=10, probs=cp.zeros((10, 8)))

# print(memory.probs.data.ptr)

# tensor = memory.get_output_tensor("probs")

# print(tensor.__cuda_array_interface__)

# data_ptr = tensor.__cuda_array_interface__["data"][0]

# mem = cp.cuda.UnownedMemory(data_ptr, 10 * 8 * 8, tensor, 0)
# memptr = cp.cuda.MemoryPointer(mem, 0)

# test_arr = cp.ndarray((10, 8), numpy.dtype('<f8'), memptr, None)

# test_arr[3, :] = 2

# print(test_arr)
# print(test_arr.data.ptr)

# mem2 = cp.cuda.UnownedMemory(data_ptr, 10 * 8 * 8, tensor, 0)
# memptr2 = cp.cuda.MemoryPointer(mem2, 0)

# test_arr2 = cp.ndarray((10, 8), numpy.dtype('<f8'), memptr2, None)

# print(test_arr2)
# print(test_arr2.data.ptr)

# print(memory.probs)

# a = memory.probs

# print(memory.probs.data.ptr)
# print(a.data.ptr)

# memory.probs[0, :] = 5
# a[1, :] = 4

# print(memory.probs)
# print(a)

# print(memory.probs.data.ptr)
# print(a.data.ptr)
