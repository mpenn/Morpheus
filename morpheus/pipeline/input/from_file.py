from morpheus.pipeline.pipeline import SourceStage
from streamz.core import Stream
from streamz import Source
from tornado.ioloop import IOLoop
from morpheus.pipeline import Stage
from morpheus.config import Config
import cudf
import numpy as np
import typing
import pandas as pd

def df_onread_cleanup(x: typing.Union[cudf.DataFrame, pd.DataFrame]):
    """
    Fixes parsing issues when reading from a file. `\n` gets converted to `\\n` for some reason
    """

    x["data"] = x["data"].str.replace('\\n', '\n', regex=False)

    return x

class FileSourceStage(SourceStage):
    def __init__(self, c: Config, filename: str):
        super().__init__(c)

        self._filename = filename
        self._batch_size = c.pipeline_batch_size
        self._input_count = None

    @property
    def name(self) -> str:
        return "from-file"

    @property
    def input_count(self) -> int:
        # Return None for no max intput count
        return self._input_count

    async def _build(self) -> Stream:

        df = cudf.read_json(self._filename, engine="cudf", lines=True)

        df = df_onread_cleanup(df)

        source: Source = Stream.from_iterable(self._generate_frames(df), asynchronous=True, loop=IOLoop.current())

        def fix_df(x: cudf.DataFrame):
            # Reset the index so they all get a unique index ID
            return x[1].reset_index(drop=True)

        # source = source.map(fix_df)

        return source

    def _generate_frames(self, df):
        for x in df.groupby(np.arange(len(df)) // self._batch_size):
            y = x[1].reset_index(drop=True)

            yield y

        for cb in self._done_callbacks:
            cb()