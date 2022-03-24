# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import typing

import neo
import pandas as pd
import typing_utils

import cudf

import morpheus._lib.stages as neos
from morpheus.config import Config
from morpheus.pipeline.file_types import FileTypes
from morpheus.pipeline.file_types import determine_file_type
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamPair


class WriteToFileStage(SinglePortStage):
    """
    This class writes messages to a file. This class does not buffer or keep the file open between messages.
    It should not be used in production code.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance.
    filename : str
        Name of the file from which the messages will be written
    overwrite : bool
        Overwrite file if exists. Will generate an error otherwise

    """
    def __init__(self, c: Config, filename: str, overwrite: bool, file_type: FileTypes = FileTypes.Auto):
        super().__init__(c)

        self._output_file = filename
        self._overwrite = overwrite

        if (os.path.exists(self._output_file)):
            if (self._overwrite):
                os.remove(self._output_file)
            else:
                raise FileExistsError("Cannot output classifications to '{}'. File exists and overwrite = False".format(
                    self._output_file))

        self._file_type = file_type

        if (self._file_type == FileTypes.Auto):
            self._file_type = determine_file_type(self._output_file)

        self._is_first = True

    @property
    def name(self) -> str:
        return "to-file"

    def accepted_types(self) -> typing.Tuple:
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple[List[str], ]
            Accepted input types

        """
        return (typing.List[str], pd.DataFrame, cudf.DataFrame)

    @classmethod
    def supports_cpp_node(cls):
        return True

    def _convert_to_strings(self, x: typing.Union[pd.DataFrame, cudf.DataFrame]):

        # Convert here to pandas since this will persist after the message is done
        if (isinstance(x, cudf.DataFrame)):
            x_pd = x.to_pandas()
        else:
            x_pd = x

        if (self._file_type == FileTypes.Json):
            output_strs = [json.dumps(y) for y in x_pd.to_dict(orient="records")]
        elif (self._file_type == FileTypes.Csv):
            output_strs = x_pd.to_csv(header=self._is_first).split("\n")
            self._is_first = False
        else:
            raise NotImplementedError("Unknown file type: {}".format(self._file_type))

        # Remove any trailing whitespace
        if (len(output_strs[-1].strip()) == 0):
            output_strs = output_strs[:-1]

        return output_strs

    def _write_str_to_file(self, x: typing.List[str]):
        """
        Messages are written to a file using this function.

        Parameters
        ----------
        x : typing.List[str]
            Messages that should be written to a file.

        """
        with open(self._output_file, "a") as f:
            f.writelines("\n".join(x))
            f.write("\n")

        return x

    def _build_single(self, seg: neo.Segment, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        if (typing_utils.issubtype(input_stream[1], typing.Union[pd.DataFrame, cudf.DataFrame])):
            to_string = seg.make_node(self.unique_name + "-tostr", self._convert_to_strings)
            seg.make_edge(stream, to_string)
            stream = to_string

        # Sink to file
        if (self._build_cpp_node()):
            to_file = neos.WriteToFileStage(seg,
                                            self.unique_name,
                                            self._output_file, ("w+" if self._overwrite else "w"))
        else:
            to_file = seg.make_node(self.unique_name, self._write_str_to_file)

        seg.make_edge(stream, to_file)
        stream = to_file

        # Return input unchanged to allow passthrough
        return input_stream
