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

import enum
import logging
import os
import typing
from functools import reduce
from io import StringIO

import neo
import numpy as np
import pandas as pd
import typing_utils
from tornado import gen
from tornado.ioloop import IOLoop

import cudf

from morpheus.config import Config
from morpheus.pipeline.file_types import FileTypes
from morpheus.pipeline.file_types import determine_file_type
from morpheus.pipeline.pipeline import SingleOutputSource
from morpheus.pipeline.pipeline import StreamFuture
from morpheus.pipeline.pipeline import StreamPair

logger = logging.getLogger(__name__)


def filter_null_data(x: typing.Union[cudf.DataFrame, pd.DataFrame]):

    if ("data" not in x):
        return x

    return x[~x['data'].isna()]


def cudf_json_onread_cleanup(x: typing.Union[cudf.DataFrame, pd.DataFrame]):
    """
    Fixes parsing issues when reading from a file. When loading a JSON file, cuDF converts ``\\n`` to
    ``\\\\n`` for some reason
    """
    if ("data" in x and not x.empty):
        x["data"] = x["data"].str.replace('\\n', '\n', regex=False)

    return x


class FileSourceStage(SingleOutputSource):
    """
    Source stage ised to load messages from a file and dumping the contents into the pipeline immediately. Useful for
    testing performance and accuracy of a pipeline.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance.
    filename : str
        Name of the file from which the messages will be read.
    iterative: boolean
        Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. Iterative mode is
        good for interleaving source stages. Non-iterative is better for dask (uploads entire dataset in one call)
    file_type : FileSourceTypes, default = 'auto'
        Indicates what type of file to read. Specifying 'auto' will determine the file type from the extension.
        Supported extensions: 'json', 'csv'
    repeat: int, default = 1
        Repeats the input dataset multiple times. Useful to extend small datasets for debugging.
    filter_null: bool, default = True
        Whether or not to filter rows with null 'data' column. Null values in the 'data' column can cause issues down
        the line with processing. Setting this to True is recommended
    cudf_kwargs: dict, default=None
        keyword args passed to underlying cuDF I/O function. See the cuDF documentation for `cudf.read_csv()` and
        `cudf.read_json()` for the available options. With `file_type` == 'json', this defaults to ``{ "lines": True }``
        and with `file_type` == 'csv', this defaults to ``{}``
    """
    def __init__(self,
                 c: Config,
                 filename: str,
                 iterative: bool = None,
                 file_type: FileTypes = FileTypes.Auto,
                 repeat: int = 1,
                 filter_null: bool = True,
                 cudf_kwargs: dict = None):
        super().__init__(c)

        self._batch_size = c.pipeline_batch_size
        self._use_dask = c.use_dask

        self._filename = filename
        self._file_type = file_type
        self._filter_null = filter_null
        self._cudf_kwargs = {} if cudf_kwargs is None else cudf_kwargs

        self._input_count = None
        self._max_concurrent = c.num_threads

        # Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. Iterative mode
        # is good for interleaving source stages. Non-iterative is better for dask (uploads entire dataset in one call)
        self._iterative = iterative if iterative is not None else not c.use_dask
        self._repeat_count = repeat

    @property
    def name(self) -> str:
        return "from-file"

    @property
    def input_count(self) -> int:
        """Return None for no max intput count"""
        return self._input_count

    def _read_file(self) -> cudf.DataFrame:

        mode = self._file_type
        parser_args = {}

        if (mode == FileTypes.Auto):
            mode = determine_file_type(self._filename)

        # Special args for JSON
        if (mode == FileTypes.Json):
            parser_args = {"engine": "cudf", "lines": True}
        elif (mode == FileTypes.Csv):
            parser_args = {"index_col": 0}

        # Update with any args set by the user. User values overwrite defaults
        parser_args.update(self._cudf_kwargs)

        # Load all of the data into memory to store as a column
        with open(self._filename) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines if len(line.rstrip()) > 0]

        lines_buffer = StringIO("\n".join(lines))

        if (mode == FileTypes.Json):
            df = cudf.read_json(lines_buffer, **parser_args)

            if (self._filter_null):
                df = filter_null_data(df)

            df = cudf_json_onread_cleanup(df)
            return df
        elif (mode == FileTypes.Csv):
            df = pd.read_csv(lines_buffer, **parser_args)

            if (self._filter_null):
                df = filter_null_data(df)

            return df
        else:
            assert False, "Unsupported file type mode: {}".format(mode)

    def _build_source(self, seg: neo.Segment) -> StreamPair:

        df = self._read_file()

        # out_stream: Source = Stream.from_iterable_done(self._generate_frames(df),
        #                                                max_concurrent=self._max_concurrent,
        #                                                asynchronous=True,
        #                                                loop=IOLoop.current())
        out_stream = seg.make_source(self.unique_name, self._generate_frames(df))
        out_type = cudf.DataFrame

        return out_stream, out_type

    def _post_build_single(self, seg: neo.Segment, out_pair: StreamPair) -> StreamPair:

        out_stream = out_pair[0]
        out_type = out_pair[1]

        # Convert our list of dataframes into the desired type. Either scatter than flatten or just flatten if not using
        # dask
        if (self._use_dask):
            if (typing_utils.issubtype(out_type, typing.List)):
                out_stream = out_stream.scatter_batch().flatten()
                out_type = StreamFuture[typing.get_args(out_type)[0]]
            else:
                out_stream = out_stream.scatter()
                out_type = StreamFuture[out_type]
        else:
            if (typing_utils.issubtype(out_type, typing.List)):
                # out_stream = out_stream.flatten()
                def flatten_fn(input: neo.Observable, output: neo.Subscriber):
                    def obs_on_next(x: typing.List):

                        for y in x:
                            output.on_next(y)

                    def obs_on_error(x):
                        output.on_error(x)

                    def obs_on_completed():
                        output.on_completed()
                        print("Subscribe on_complete in flatten_fn")

                    obs = neo.Observer.make_observer(obs_on_next, obs_on_error, obs_on_completed)

                    input.subscribe(obs)

                    print("Subscribe done in flatten_fn")

                flattened = seg.make_node_full(self.unique_name + "post", flatten_fn)
                seg.make_edge(out_stream, flattened)
                out_stream = flattened
                out_type = typing.get_args(out_type)[0]

        return super()._post_build_single(seg, (out_stream, out_type))

    def _generate_frames(self, df):
        count = 0

        for _ in range(self._repeat_count):

            yield df

            count += 1

            # If we are looping, copy and shift the index
            if (self._repeat_count > 0):
                prev_df = df
                df = prev_df.copy()

                df.index += len(df)

        logger.debug("File input stage complete")
        # Indicate that we are stopping (not the best way of doing this)
        # self._source_stream.stop()
