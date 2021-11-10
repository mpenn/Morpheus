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

import asyncio
import glob
import logging
import os
import typing

import numpy as np
import pandas as pd
import typing_utils
from streamz import Source
from streamz.core import Stream
from tornado.ioloop import IOLoop

from morpheus.config import Config
from morpheus.pipeline.input.from_file import FileSourceTypes
from morpheus.pipeline.pipeline import SingleOutputSource
from morpheus.pipeline.pipeline import StreamFuture
from morpheus.pipeline.pipeline import StreamPair
from morpheus.utils.producer_consumer_queue import AsyncIOProducerConsumerQueue
from morpheus.utils.producer_consumer_queue import Closed

logger = logging.getLogger(__name__)

LIST_OF_COLUMNS = [
    'eventSource',
    'eventName',
    'sourceIPAddress',
    'userAgent',
    'userIdentitytype',
    'requestParametersroleArn',
    'requestParametersroleSessionName',
    'requestParametersdurationSeconds',
    'responseElementsassumedRoleUserassumedRoleId',
    'responseElementsassumedRoleUserarn',
    'apiVersion',
    'userIdentityprincipalId',
    'userIdentityarn',
    'userIdentityaccountId',
    'userIdentityaccessKeyId',
    'userIdentitysessionContextsessionIssuerprincipalId',
    'userIdentitysessionContextsessionIssuerarn',
    'userIdentitysessionContextsessionIssueruserName',
    'tlsDetailsclientProvidedHostHeader',
    'requestParametersownersSetitems',
    'requestParametersmaxResults',
    'requestParametersinstancesSetitems',
    'errorCode',
    'errorMessage',
    'requestParametersmaxItems',
    'responseElementsrequestId',
    'responseElementsinstancesSetitems',
    'requestParametersgroupSetitems',
    'requestParametersinstanceType',
    'requestParametersmonitoringenabled',
    'requestParametersdisableApiTermination',
    'requestParametersebsOptimized',
    'responseElementsreservationId',
    'requestParametersgroupName',
    'eventTime',
    'recipientAccountId',
    'awsRegion'
]

EXTRA_KEEP_COLS = ["event_dt"]


class CloudTrailSourceStage(SingleOutputSource):
    """
    Source stage ised to load messages from a file and dumping the contents into the pipeline immediately. Useful for
    testing performance and accuracy of a pipeline.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance.
    input_glob : str
        Input glob pattern to match files to read. For example, './input_dir/*.json' would read all files with the
        'json' extension in the directory input_dir.
    watch_directory : bool, default = False
        The watch directory option instructs this stage to not close down once all files have been read. Instead it will
        read all files that match the 'input_glob' pattern, and then continue to watch the directory for additional
        files. Any new files that are added that match the glob will then be processed.
    iterative: boolean
        Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. Iterative mode is
        good for interleaving source stages. Non-iterative is better for dask (uploads entire dataset in one call)
    max_files: int, default = -1
        Max number of files to read. Useful for debugging to limit startup time. Default value of -1 is unlimited.
    file_type : FileSourceTypes, default = 'auto'
        Indicates what type of file to read. Specifying 'auto' will determine the file type from the extension.
        Supported extensions: 'json', 'csv'
    pandas_kwargs: dict, default=None
        keyword args passed to underlying Pandas I/O function. See the Pandas documentation for `pandas.read_csv()` and
        `pandas.read_json()` for the available options. With `file_type` == 'json', this defaults to ``{ "lines": True }``
        and with `file_type` == 'csv', this defaults to ``{}``
    """
    def __init__(self,
                 c: Config,
                 input_glob: str,
                 watch_directory: bool = False,
                 iterative: bool = None,
                 max_files: int = -1,
                 file_type: FileSourceTypes = FileSourceTypes.Auto,
                 repeat: int = 1):

        super().__init__(c)

        self._batch_size = c.pipeline_batch_size
        self._use_dask = c.use_dask

        self._input_glob = input_glob
        self._file_type = file_type
        self._max_files = max_files

        self._input_count = None
        self._max_concurrent = c.num_threads

        # Hold the max index we have seen to ensure sequential and increasing indexes
        self._max_index = 0

        # Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. Iterative mode
        # is good for interleaving source stages. Non-iterative is better for dask (uploads entire dataset in one call)
        self._iterative = iterative if iterative is not None else not c.use_dask
        self._repeat_count = repeat
        self._watch_directory = watch_directory

        # Will be a watchdog observer if enabled
        self._watcher = None

    @property
    def name(self) -> str:
        return "from-cloudtrail"

    @property
    def input_count(self) -> int:
        """Return None for no max intput count"""
        return self._input_count

    async def stop(self):

        if (self._watcher is not None):
            self._watcher.stop()

        return await super().stop()

    async def join(self):

        if (self._watcher is not None):
            self._watcher.join()

        return await super().join()

    def _read_file(self, filename: str) -> pd.DataFrame:

        mode = self._file_type
        kwargs = {}

        if (mode == FileSourceTypes.Auto):
            # Determine from the file extension
            ext = os.path.splitext(filename)

            # Get the extension without the dot
            ext = ext[1].lower()[1:]

            # Check against supported options
            if (ext == "json" or ext == "jsonlines"):
                mode = FileSourceTypes.Json
            elif (ext == "csv"):
                mode = FileSourceTypes.Csv
            else:
                raise RuntimeError(
                    "Unsupported extension '{}' with 'auto' type. 'auto' only works with: csv, json".format(ext))

        if (mode == FileSourceTypes.Json):
            df = pd.read_json(filename, **kwargs)
            df = pd.json_normalize(df['Records'])
            return df
        elif (mode == FileSourceTypes.Csv):
            df = pd.read_csv(filename, **kwargs)
            return df
        else:
            assert False, "Unsupported file type mode: {}".format(mode)

    def _read_files(self, file_list: typing.List[str]) -> pd.DataFrame:

        # Using pandas to parse nested JSON until cuDF adds support
        # https://github.com/rapidsai/cudf/issues/8827
        dfs = []
        for file in file_list:
            df = self._read_file(file)
            dfs.append(df)

        src_df = pd.concat(dfs)

        # Replace all the dots in column names
        src_df.columns = src_df.columns.str.replace('.', '', regex=False)

        # Now sort the dataframe by the event time.
        src_df["event_dt"] = pd.to_datetime(src_df["eventTime"])
        # Sort by time and reset the index to give all unique index values
        src_df = src_df.sort_values(by="event_dt")

        # Set up the sequential index
        src_df = src_df.set_index(pd.RangeIndex(self._max_index, self._max_index + len(src_df), 1))

        self._max_index += len(src_df)

        def remove_null(x):
            """
            Util function that cleans up data
            :param x:
            :return:
            """
            if isinstance(x, list):
                if isinstance(x[0], dict):
                    key = list(x[0].keys())
                    return x[0][key[0]]
            return x

        def filter_columns(cloudtrail_df):
            """
            Filter columns based on a list of columns
            :param cloudtrail_df:
            :return:
            """
            col_not_exists = [col for col in LIST_OF_COLUMNS if col not in cloudtrail_df.columns]
            for col in col_not_exists:
                cloudtrail_df[col] = np.nan
            cloudtrail_df = cloudtrail_df[LIST_OF_COLUMNS + EXTRA_KEEP_COLS]
            return cloudtrail_df

        def clean_column(cloudtrail_df):
            """
            Clean a certain column based on lists inside
            :param cloudtrail_df:
            :return:
            """
            col_name = 'requestParametersownersSetitems'
            cloudtrail_df[col_name] = cloudtrail_df[col_name].apply(lambda x: remove_null(x))
            return cloudtrail_df

        # Clean up the dataframe
        src_df = filter_columns(src_df)
        src_df = clean_column(src_df)

        logger.debug("CloudTrail loading complete. Total rows: %d. Timespan: %s",
                     len(src_df),
                     str(src_df.loc[src_df.index[-1], "event_dt"] - src_df.loc[src_df.index[0], "event_dt"]))

        src_lines = [""] * len(src_df)

        src_df["_orig"] = src_lines

        return src_df

    async def _get_dataframe_queue(self) -> AsyncIOProducerConsumerQueue[pd.DataFrame]:
        # Return an asyncio queue
        queue = AsyncIOProducerConsumerQueue()

        if (self._watch_directory):

            from watchdog.events import FileSystemEvent
            from watchdog.events import PatternMatchingEventHandler
            from watchdog.observers import Observer

            # Create a file watcher
            self._watcher = Observer()
            self._watcher.setDaemon(True)
            self._watcher.setName("DirectoryWatcher")

            glob_split = self._input_glob.split("*", 1)

            if (len(glob_split) == 1):
                raise RuntimeError(("When watching directories, input_glob must have a wildcard. "
                                    "Otherwise no files will be matched."))

            dir_to_watch = os.path.dirname(glob_split[0])
            match_pattern = self._input_glob.replace(dir_to_watch + "/", "", 1)
            dir_to_watch = os.path.abspath(os.path.dirname(glob_split[0]))

            event_handler = PatternMatchingEventHandler(patterns=[match_pattern])

            loop = asyncio.get_running_loop()

            def process_dir_change(event: FileSystemEvent):
                async def process_event(e: FileSystemEvent):
                    df = self._read_files([e.src_path])

                    await queue.put(df)

                asyncio.run_coroutine_threadsafe(process_event(event), loop)

            event_handler.on_created = process_dir_change

            self._watcher.schedule(event_handler, dir_to_watch, recursive=True)

            self._watcher.start()

        # Load the glob once and return
        file_list = glob.glob(self._input_glob)

        if (self._max_files > 0):
            file_list = file_list[:self._max_files]

        logger.info("Found %d CloudTrail files in glob. Loading...", len(file_list))

        df_array = []

        # Only load files if there are any
        if (len(file_list) > 0):
            df = self._read_files(file_list)

            df_array.append(df)

            for _ in range(1, self._repeat_count):
                x = df.copy()

                # Increment the index
                x.index = range(self._max_index, self._max_index + len(df))
                self._max_index += len(df)

                # Now increment the timestamps by the interval in the df
                x["event_dt"] = x["event_dt"] + (x["event_dt"].iloc[-1] - x["event_dt"].iloc[0])
                x["eventTime"] = x["event_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

                df_array.append(x)

                # Set df for next iteration
                df = x

        # Push all to the queue and close it
        for d in df_array:

            await queue.put(d)

        if (not self._watch_directory):
            # Close the queue
            await queue.close()

        return queue

    def _build_source(self) -> typing.Tuple[Source, typing.Type]:

        out_stream: Source = Stream.from_iterable_done(self._generate_frames(),
                                                       max_concurrent=self._max_concurrent,
                                                       asynchronous=True,
                                                       loop=IOLoop.current())

        out_type = pd.DataFrame if self._iterative else typing.List[pd.DataFrame]

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
                out_stream = out_stream.flatten(max_concurrent=self._max_concurrent)
                out_type = typing.get_args(out_type)[0]

        return super()._post_build_single((out_stream, out_type))

    async def _generate_frames(self):

        df_queue: AsyncIOProducerConsumerQueue[pd.DataFrame] = await self._get_dataframe_queue()

        while True:

            try:
                df = await df_queue.get()

                yield df

                df_queue.task_done()

            except Closed:
                break

        # Indicate that we are stopping (not the best way of doing this)
        self._source_stream.stop()
