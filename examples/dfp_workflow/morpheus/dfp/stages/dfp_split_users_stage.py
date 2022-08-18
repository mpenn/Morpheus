import logging
import time
import typing
from contextlib import contextmanager

import numpy as np
import srf
from srf.core import operators as ops

import cudf

from morpheus.config import Config
from morpheus.messages.message_meta import UserMessageMeta
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

# Setup conda environment
conda_env = {
    'channels': ['defaults', 'conda-forge'],
    'dependencies': ['python={}'.format('3.8'), 'pip'],
    'pip': ['mlflow', 'dfencoder'],
    'name': 'mlflow-env'
}

logger = logging.getLogger("morpheus.{}".format(__name__))


@contextmanager
def log_time(log_fn, msg: str, *args, **kwargs):

    import time

    start_time = time.time()

    yield

    duration = (time.time() - start_time) * 1000.0

    # Call the log function
    log_fn(msg.format(**{"duration": duration}), *args, **kwargs)


class DFPSplitUsersStage(SinglePortStage):

    def __init__(self, c: Config, include_generic: bool, include_individual: bool, skip_users: typing.List[str]):
        super().__init__(c)

        self._include_generic = include_generic
        self._include_individual = include_individual
        self._skip_users = skip_users

        # Map of user ids to total number of messages. Keeps indexes monotonic and increasing per user
        self._user_index_map: typing.Dict[str, int] = {}

    @property
    def name(self) -> str:
        return "dfp-split-users"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (cudf.DataFrame, )

    def extract_users(self, message: cudf.DataFrame):
        if (message is None):
            return []

        start_time = time.time()

        if (isinstance(message, cudf.DataFrame)):
            # Convert to pandas because cudf is slow at this
            message = message.to_pandas()

        split_dataframes: typing.List[typing.Tuple[str, cudf.DataFrame]] = []

        # If we are skipping users, do that here
        if (len(self._skip_users) > 0):
            message = message[~message[self._config.ae.userid_column_name].isin(self._skip_users)]

        # Split up the dataframes
        if (self._include_generic):
            split_dataframes.append(("generic_user", message))

        if (self._include_individual):
            with log_time(logger.debug, "unique() call took {duration} ms"):
                usernames = message[message["username"].notnull()]["username"].unique()

            with log_time(logger.debug, 'set_index call took {duration} ms'):
                df2 = message.set_index("username", append=True)
                [x for i, x in df2.groupby(level=1, sort=False)]

            with log_time(logger.debug,
                          '[x for i, x in message.groupby("username", sort=False)] call took {duration} ms'):
                [x for i, x in message.groupby("username", sort=False)]

            with log_time(logger.debug, "message[message['username'] == x] call took {duration} ms"):
                split_dataframes.extend([(x, message[message['username'] == x]) for x in usernames])

        output_messages: typing.List[UserMessageMeta] = []

        for user_id, user_df in split_dataframes:

            if (user_id in self._skip_users):
                continue

            current_user_count = self._user_index_map.get(user_id, 0)

            # Reset the index so that users see monotonically increasing indexes
            user_df.index = range(current_user_count, current_user_count + len(user_df))
            self._user_index_map[user_id] = current_user_count + len(user_df)

            output_messages.append(UserMessageMeta(df=user_df, user_id=user_id))

            # logger.debug("Emitting dataframe for user '%s'. Start: %s, End: %s, Count: %s",
            #              user,
            #              df_user["timestamp"].min(),
            #              df_user["timestamp"].max(),
            #              df_user["timestamp"].count())

        rows_per_user = [len(x.df) for x in output_messages]

        duration = (time.time() - start_time) * 1000.0

        logger.debug(
            "Batch split users complete. Input: %s rows from %s to %s. Output: %s users, rows/user min: %s, max: %s, avg: %s. Duration: %s ms",
            len(message),
            message["timestamp"].min(),
            message["timestamp"].max(),
            len(rows_per_user),
            np.min(rows_per_user),
            np.max(rows_per_user),
            np.mean(rows_per_user),
            duration,
        )

        return output_messages

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(self.extract_users), ops.flatten()).subscribe(sub)

        stream = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], stream)

        return stream, UserMessageMeta
