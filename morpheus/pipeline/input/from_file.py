from functools import reduce
from tornado import gen
from morpheus.pipeline.pipeline import SourceStage
from streamz.core import RefCounter, Stream
from streamz import Source
from tornado.ioloop import IOLoop
from morpheus.pipeline import Stage
from morpheus.config import Config
import cudf
import numpy as np
import typing
import pandas as pd
import asyncio
from tqdm import tqdm


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
        for x in self._iterable:
            if self.stopped:
                break

            counter = RefCounter(initial=0, cb=self._ref_callback, loop=self.loop)

            self._counters.append(counter)
            self._total_count += 1

            meta = [{
                "ref": counter
            }]

            await asyncio.gather(*self._emit(x, metadata=meta))
        self.stopped = True

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

        source: Source = Stream.from_iterable_done(self._generate_frames(df), asynchronous=True, loop=IOLoop.current())

        def fix_df(x: cudf.DataFrame):
            # Reset the index so they all get a unique index ID
            return x[1].reset_index(drop=True)

        # source = source.map(fix_df)

        return source

    def _generate_frames(self, df):
        count = 0

        for x in df.groupby(np.arange(len(df)) // self._batch_size):
            y = x[1].reset_index(drop=True)

            yield y
            count += 1

            # if (count >= 2):
            #     break

        for cb in self._done_callbacks:
            cb()