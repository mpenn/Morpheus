import logging
import time
import typing
from datetime import date

import numpy as np
import srf
from srf.core import operators as ops

import cudf

from morpheus.config import Config
from morpheus.messages.message_meta import UserMessageMeta
from morpheus.messages.multi_ae_message import MultiAEMessage
from morpheus.pipeline.single_port_stage import SinglePortStage
from morpheus.pipeline.stream_pair import StreamPair

from .dfp_autoencoder import DFPAutoEncoder

logger = logging.getLogger("morpheus.{}".format(__name__))


class DFPPostprocessingStage(SinglePortStage):

    def __init__(self, c: Config):
        super().__init__(c)

    @property
    def name(self) -> str:
        return "dfp-postproc"

    def supports_cpp_node(self):
        return False

    def accepted_types(self) -> typing.Tuple:
        return (MultiAEMessage, )

    def _extract_events(self, message: MultiAEMessage):

        # Return the message for the next stage
        # user = message.meta.user_id
        # df_user = message.get_meta()
        model: DFPAutoEncoder = message.model

        z_scores = (message.get_meta('anomaly_score') - model.val_loss_mean) / model.val_loss_std

        message.set_meta("z-score", z_scores)

        above_threshold_df = message.get_meta()[z_scores > 2.0]

        if (not above_threshold_df.empty):
            above_threshold_df['event_time'] = date.today().strftime('%Y-%m-%dT%H:%M:%SZ')
            above_threshold_df = above_threshold_df.replace(np.nan, 'NaN', regex=True)

            return above_threshold_df

        return None

    def on_data(self, message: MultiAEMessage):
        if (not message):
            return None

        start_time = time.time()

        extracted_events = self._extract_events(message)

        duration = (time.time() - start_time) * 1000.0

        logger.debug("Completed postprocessing for user %s in %s ms. Event count: %s. Start: %s, End: %s",
                     message.meta.user_id,
                     duration,
                     0 if extracted_events is None else len(extracted_events),
                     message.get_meta("timestamp").min(),
                     message.get_meta("timestamp").max())

        return UserMessageMeta(extracted_events, user_id=typing.cast(UserMessageMeta, message.meta).user_id)

    def _build_single(self, builder: srf.Builder, input_stream: StreamPair) -> StreamPair:

        def node_fn(obs: srf.Observable, sub: srf.Subscriber):
            obs.pipe(ops.map(self.on_data), ops.filter(lambda x: x is not None)).subscribe(sub)

        stream = builder.make_node_full(self.unique_name, node_fn)
        builder.make_edge(input_stream[0], stream)

        return stream, UserMessageMeta
