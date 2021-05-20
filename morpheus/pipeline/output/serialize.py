import json
import re
import typing

from morpheus.config import Config
from morpheus.pipeline import Stage
from morpheus.pipeline.messages import MultiMessage
from morpheus.pipeline.pipeline import StreamPair


class SerializeStage(Stage):
    """
    This class converts a `MultiMessage` object into a list of strings for writing out to file or Kafka.

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance
    include : list[str]
        Attributes that are required send to downstream stage. 
    exclude : typing.List[str]
        Attributes that are not required send to downstream stage.

    """
    def __init__(self, c: Config, include: typing.List[str] = None, exclude: typing.List[str] = [r'^ID$', r'^ts_']):
        super().__init__(c)

        self._include_columns = include
        self._exclude_columns = exclude

    @property
    def name(self) -> str:
        return "serialize"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple(morpheus.pipeline.messages.MultiMessage, )
            Accepted input types

        """
        return (MultiMessage, )

    @staticmethod
    def convert_to_json(x: MultiMessage, include_columns: typing.Pattern, exclude_columns: typing.List[typing.Pattern]):
        """
        Converts dataframe to entries to JSON lines.

        Parameters
        ----------
        x : morpheus.pipeline.messages.MultiMessage
            MultiMessage instance that contains data.
        include_columns : typing.Pattern
            Columns that are required send to downstream stage.
        exclude_columns : typing.List[typing.Pattern])
            Columns that are not required send to downstream stage.

        """
        columns: typing.List[str] = []

        # First build up list of included. If no include regex is specified, select all
        if (include_columns is None):
            columns = list(x.meta.df.columns)
        else:
            columns = [y for y in list(x.meta.df.columns) if include_columns.match(y)]

        # Now remove by the ignore
        for test in exclude_columns:
            columns = [y for y in columns if not test.match(y)]

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
        output_strs = [json.dumps(y) for y in df.to_dict(orient="records")]

        # Return list of strs to write out
        return output_strs

    async def _build(self, input_stream: StreamPair) -> StreamPair:

        include_columns = None

        if (self._include_columns is not None and len(self._include_columns) > 0):
            include_columns = re.compile("({})".format("|".join(self._include_columns)))

        exclude_columns = [re.compile(x) for x in self._exclude_columns]

        # Convert the messages to rows of strings
        stream = input_stream[0].async_map(SerializeStage.convert_to_json,
                                           executor=self._pipeline.thread_pool,
                                           include_columns=include_columns,
                                           exclude_columns=exclude_columns)

        # Return input unchanged
        return stream, typing.List[str]
