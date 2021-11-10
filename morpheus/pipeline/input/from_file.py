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

import numpy as np
import pandas as pd
import typing_utils
from streamz import Source
from streamz.core import RefCounter
from streamz.core import Stream
from tornado import gen
from tornado.ioloop import IOLoop

import cudf

from morpheus.config import Config
from morpheus.pipeline.pipeline import SingleOutputSource
from morpheus.pipeline.pipeline import StreamFuture
from morpheus.pipeline.pipeline import StreamPair

logger = logging.getLogger(__name__)


@enum.unique
class FileSourceTypes(str, enum.Enum):
    """The type of files that the `FileSourceStage` can read. Use 'auto' to determine from the file extension."""
    Auto = "auto"
    Json = "json"
    Csv = "csv"


@Stream.register_api(staticmethod)
class from_iterable_done(Source):
    """
    Emits items from an iterable.

    Parameters
    ----------
    iterable : iterable
        An iterable to emit messages from.

    Examples
    --------
    >>> source = Stream.from_iterable(range(3))
    >>> L = source.sink_to_list()
    >>> source.start()
    >>> L
    [0, 1, 2]

    """
    def __init__(self, iterable, **kwargs):
        self._iterable = iterable
        super().__init__(**kwargs)

        self._total_count = 0
        self._counters: typing.List[RefCounter] = []

    async def _source_generator(self):
        async for x in self._iterable:
            yield self._emit(x)

            if (self.stopped):
                break

    @gen.coroutine
    def _ref_callback(self):
        self._total_count -= 1

        sum_count = reduce(lambda count, x: count + min(x.count, 1), self._counters, 0)

        if (sum_count != self._total_count):
            logger.debug("Mismatch. Sum: {}, Count: {}".format(sum_count, self._total_count))

    async def _run(self):
        count = 0
        async for x in self._iterable:
            if self.stopped:
                break

            self._total_count += 1

            count += 1

            await self._emit(x)
        self.stopped = True


def filter_null_data(x: typing.Union[cudf.DataFrame, pd.DataFrame]):
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
                 file_type: FileSourceTypes = FileSourceTypes.Auto,
                 repeat: int = 1,
                 filter_null: bool = True,
                 cudf_kwargs: dict = None):
        super().__init__(c)

        self._batch_size = c.pipeline_batch_size
        self._use_dask = c.use_dask

        self._filename = filename
        self._iterative = iterative if iterative is not None else not c.use_dask
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
        cudf_args = {}

        if (mode == FileSourceTypes.Auto):
            # Determine from the file extension
            ext = os.path.splitext(self._filename)

            # Get the extension without the dot
            ext = ext[1].lower()[1:]

            # Check against supported options
            if (ext == "json" or ext == "jsonlines"):
                mode = FileSourceTypes.Json
                cudf_args = {"engine": "cudf", "lines": True}
            elif (ext == "csv"):
                mode = FileSourceTypes.Csv
            else:
                raise RuntimeError(
                    "Unsupported extension '{}' with 'auto' type. 'auto' only works with: csv, json".format(ext))

        # Update with any args set by the user. User values overwrite defaults
        cudf_args.update(self._cudf_kwargs)

        if (mode == FileSourceTypes.Json):
            df = cudf.read_json(self._filename, **cudf_args)

            if (self._filter_null):
                df = filter_null_data(df)

            df = cudf_json_onread_cleanup(df)
            return df
        elif (mode == FileSourceTypes.Csv):
            df = cudf.read_csv(self._filename, **cudf_args)

            if (self._filter_null):
                df = filter_null_data(df)

            return df
        else:
            assert False, "Unsupported file type mode: {}".format(mode)

    def _build_source(self) -> typing.Tuple[Source, typing.Type]:

        df = self._read_file()

        out_stream: Source = Stream.from_iterable_done(self._generate_frames(df),
                                                       max_concurrent=self._max_concurrent,
                                                       asynchronous=True,
                                                       loop=IOLoop.current())
        out_type = cudf.DataFrame if self._iterative else typing.List[cudf.DataFrame]

        return out_stream, out_type

    def _post_build_single(self, out_pair: StreamPair) -> StreamPair:

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
                out_stream = out_stream.flatten()
                out_type = typing.get_args(out_type)[0]

        return super()._post_build_single((out_stream, out_type))

    async def _generate_frames(self, df):
        count = 0
        out = []

        for _ in range(self._repeat_count):
            for x in df.groupby(np.arange(len(df)) // self._batch_size):
                y = x[1].reset_index(drop=True)

                count += 1

                if (self._iterative):
                    yield y
                else:
                    out.append(y)

            if (not self._iterative):
                yield out
                out = []

        # Indicate that we are stopping (not the best way of doing this)
        self._source_stream.stop()
