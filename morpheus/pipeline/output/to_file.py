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
    def __init__(self, c: Config, output_file: str, overwrite: bool):
        super().__init__(c)

        self._output_file = output_file
        self._overwrite = overwrite
        self._ignore_columns = [r'^ID$', r'^ts_']

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
        return (MultiMessage, )

    def _convert_to_json(self, x: MultiMessage):

        # Get list of columns that pass ignore regex
        columns = list(x.meta.df.columns)

        for test in self._ignore_columns:
            columns = [y for y in columns if not re.match(test, y)]

        # Get metadata from columns
        df = x.get_meta(columns)

        def double_serialize(y: str):
            try:
                return json.dumps(json.dumps(json.loads(y)))
            except:
                return y

        # Special processing for the data column (need to double serialize to match input)
        if ("data" in df):
            df["data"] = df["data"].apply(double_serialize)

        # Convert to list of json string objects
        output_strs = [json.dumps(y) + "\n" for y in df.to_dict(orient="records")]

        # Return list of strs to write out
        return output_strs

    def write_to_file(self, x: typing.List[str]):
        with open(self._output_file, "a") as f:
            f.writelines(x)

    async def _build(self, input_stream: typing.Tuple[Stream, typing.Type]) -> typing.Tuple[Stream, typing.Type]:

        # Convert the messages to rows of strings
        stream = input_stream[0].async_map(self._convert_to_json, executor=self._pipeline.thread_pool)

        # Sink to file
        stream.sink(self.write_to_file)

        # Return input unchanged
        return input_stream