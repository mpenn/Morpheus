import logging
import typing

import pandas as pd
import srf

import cudf

from morpheus._lib.file_types import FileTypes
from morpheus.config import Config
from morpheus.io.deserializers import read_file_to_df
from morpheus.pipeline.single_output_source import SingleOutputSource
from morpheus.pipeline.stream_pair import StreamPair

from ..utils.column_info import DataFrameInputSchema
from ..utils.column_info import process_dataframe

logger = logging.getLogger("morpheus.{}".format(__name__))


class MultiFileSource(SingleOutputSource):
    """
    Source stage is used to load messages from a file and dumping the contents into the pipeline immediately. Useful for
    testing performance and accuracy of a pipeline.

    Parameters
    ----------
    c : `morpheus.config.Config`
        Pipeline configuration instance.
    filename : str
        Name of the file from which the messages will be read.
    iterative: boolean
        Iterative mode will emit dataframes one at a time. Otherwise a list of dataframes is emitted. Iterative mode is
        good for interleaving source stages.
    file_type : `morpheus._lib.file_types.FileTypes`, default = 'auto'
        Indicates what type of file to read. Specifying 'auto' will determine the file type from the extension.
        Supported extensions: 'json', 'csv'
    repeat: int, default = 1
        Repeats the input dataset multiple times. Useful to extend small datasets for debugging.
    filter_null: bool, default = True
        Whether or not to filter rows with null 'data' column. Null values in the 'data' column can cause issues down
        the line with processing. Setting this to True is recommended.
    cudf_kwargs: dict, default=None
        keyword args passed to underlying cuDF I/O function. See the cuDF documentation for `cudf.read_csv()` and
        `cudf.read_json()` for the available options. With `file_type` == 'json', this defaults to ``{ "lines": True }``
        and with `file_type` == 'csv', this defaults to ``{}``.
    """

    def __init__(
        self,
        c: Config,
        input_schema: DataFrameInputSchema,
        filenames: typing.List[str],
        file_type: FileTypes = FileTypes.Auto,
        cudf_kwargs: dict = None,
    ):

        super().__init__(c)

        self._batch_size = c.pipeline_batch_size

        self._input_schema = input_schema
        self._filenames = filenames
        self._file_type = file_type
        self._cudf_kwargs = {} if cudf_kwargs is None else cudf_kwargs

        self._input_count = None
        self._max_concurrent = c.num_threads

    @property
    def name(self) -> str:
        return "from-multi-file"

    @property
    def input_count(self) -> int:
        """Return None for no max intput count"""
        return self._input_count

    def supports_cpp_node(self):
        return False

    def _generate_frames(self):

        loaded_dfs = []

        for f in self._filenames:

            # Read the dataframe into memory
            df = read_file_to_df(f,
                                 self._file_type,
                                 filter_nulls=True,
                                 df_type="pandas",
                                 parser_kwargs=self._cudf_kwargs)

            df = process_dataframe(df, self._input_schema)

            loaded_dfs.append(df)

        combined_df = pd.concat(loaded_dfs)

        yield combined_df

    def _build_source(self, builder: srf.Builder) -> StreamPair:

        if self._build_cpp_node():
            raise RuntimeError("Does not support C++ nodes")
        else:
            out_stream = builder.make_source(self.unique_name, self._generate_frames())

        out_type = cudf.DataFrame

        return out_stream, out_type
