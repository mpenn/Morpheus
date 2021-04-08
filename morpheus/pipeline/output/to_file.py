from morpheus.pipeline.pipeline import StreamFuture, StreamPair
import typing_utils
from morpheus.pipeline.messages import MultiMessage
from streamz.core import Stream
from streamz import Source
from tornado.ioloop import IOLoop
from morpheus.pipeline import Stage
from morpheus.config import Config
import cudf
import numpy as np
import typing
import re
import json
import os

class WriteToFileStage(Stage):
    def __init__(self, c: Config, filename: str, overwrite: bool):
        super().__init__(c)

        self._output_file = filename
        self._overwrite = overwrite

        if (os.path.exists(self._output_file)):
            if (self._overwrite):
                os.remove(self._output_file)
            else:
                raise FileExistsError("Cannot output classifications to '{}'. File exists and overwrite = False".format(
                    self._output_file))

    @property
    def name(self) -> str:
        return "to-file"

    def accepted_types(self) -> typing.Tuple:
        return (typing.List[str],)

    def write_to_file(self, x: typing.List[str]):
        with open(self._output_file, "a") as f:
            f.writelines("\n".join(x))
            f.write("\n")

    async def _build(self, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        # Wrap single strings into lists
        if (typing_utils.issubtype(input_stream[1], StreamFuture[str]) or typing_utils.issubtype(input_stream[1], str)):
            stream = stream.map(lambda x: [x])

        # Do a gather just in case we are using dask
        stream = stream.gather()

        # Sink to file
        stream.sink(self.write_to_file)

        # Return input unchanged
        return input_stream