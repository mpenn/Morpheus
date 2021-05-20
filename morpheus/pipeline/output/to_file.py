import os
import typing

import typing_utils

from morpheus.config import Config
from morpheus.pipeline import Stage
from morpheus.pipeline.pipeline import StreamFuture
from morpheus.pipeline.pipeline import StreamPair


class WriteToFileStage(Stage):
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
        """
        Returns accepted input types for this stage.

        Returns
        -------
        typing.Tuple[List[str], ]
            Accepted input types

        """
        return (typing.List[str], )

    def write_to_file(self, x: typing.List[str]):
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
