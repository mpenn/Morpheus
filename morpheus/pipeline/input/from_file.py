import asyncio
import typing
from functools import reduce

import cudf
import numpy as np
import pandas as pd
from streamz import Source
from streamz.core import RefCounter
from streamz.core import Stream
from tornado import gen
from tornado.ioloop import IOLoop
from tqdm import tqdm

from morpheus.config import Config
from morpheus.pipeline.pipeline import SourceStage
from morpheus.pipeline.pipeline import StreamPair


@Stream.register_api(staticmethod)
class from_iterable_done(Source):
    """ Emits items from an iterable.

    Parameters
    ----------
    iterable: iterable
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

    @gen.coroutine
    def _ref_callback(self):
        self._total_count -= 1

        sum_count = reduce(lambda count, x: count + min(x.count, 1), self._counters, 0)

        if (sum_count != self._total_count):
            tqdm.write("Mismatch. Sum: {}, Count: {}".format(sum_count, self._total_count))

    async def _run(self):
        count = 0
        async for x in self._iterable:
            if self.stopped:
                break

            self._total_count += 1

            count += 1

            await self._emit(x)
        self.stopped = True


def df_onread_cleanup(x: typing.Union[cudf.DataFrame, pd.DataFrame]):
    """
    Fixes parsing issues when reading from a file. `\n` gets converted to `\\n` for some reason
    """

    if ("data" in x):
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

    async def _build(self) -> StreamPair:

        df = cudf.read_json(self._filename, engine="cudf", lines=True)

        df = df_onread_cleanup(df)

        source: Source = Stream.from_iterable_done(self._generate_frames(df), asynchronous=True, loop=IOLoop.current())

        return source, typing.List[cudf.DataFrame]

    async def _generate_frames(self, df):
        count = 0
        out = []

        for x in df.groupby(np.arange(len(df)) // self._batch_size):
            y = x[1].reset_index(drop=True)

            out.append(y)
            count += 1

            # yield y

        yield out

        # Perform the callbacks
        for cb in self._done_callbacks:
            cb()
