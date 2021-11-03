import logging
import typing

import cupy as cp
import mlflow

from morpheus.config import Config
from morpheus.pipeline.messages import MultiResponseMessage
from morpheus.pipeline.messages import MultiResponseProbsMessage
from morpheus.pipeline.pipeline import SinglePortStage
from morpheus.pipeline.pipeline import StreamPair

logger = logging.getLogger(__name__)


class MLFlowDriftStage(SinglePortStage):
    """
    Caculates model drift over time and reports the information to ML Flow

    Parameters
    ----------
    c : morpheus.config.Config
        Pipeline configuration instance

    """
    def __init__(self,
                 c: Config,
                 tracking_uri: str = None,
                 experiment_name: str = "Morpheus",
                 run_id: str = None,
                 labels: typing.List[str] = None,
                 batch_size: int = -1,
                 force_new_run: bool = False):
        super().__init__(c)

        self._feature_length = c.feature_length
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._run_id = run_id
        self._labels = c.class_labels if labels is None or len(labels) == 0 else labels

        if (batch_size == -1):
            self._batch_size = c.pipeline_batch_size
        else:
            self._batch_size = batch_size

        if (self._batch_size > c.pipeline_batch_size):
            logger.warning(("Warning: MLFlowDriftStage batch_size (%d) is greater than pipeline_batch_size (%d). "
                            "Reducing stage batch_size to pipeline_batch_size"),
                           self._batch_size,
                           c.pipeline_batch_size)
            self._batch_size = c.pipeline_batch_size

        # Set the active run up
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        if (run_id is None and not force_new_run):
            # Get the current experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)

            # Find all active runs
            active_runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                                             order_by=["attribute.start_time"])

            if (len(active_runs) > 0 and "tags.morpheus.type" in active_runs):
                morpheus_runs = active_runs[active_runs["tags.morpheus.type"] == "drift"]

                if (len(morpheus_runs) > 0):
                    run_id = morpheus_runs.head(n=1)["run_id"].iloc[0]

        mlflow.start_run(run_id, run_name="Model Drift", tags={"morpheus.type": "drift"})

    @property
    def name(self) -> str:
        return "mlflow_drift"

    def accepted_types(self) -> typing.Tuple:
        """
        Accepted input types for this stage are returned.

        Returns
        -------
        typing.Tuple[MultiResponseMessage, ]
            Accepted input types

        """
        return (MultiResponseProbsMessage, )

    def _calc_drift(self, x: MultiResponseProbsMessage):

        # All probs in a batch will be calculated
        shifted = cp.abs(x.probs - 0.5) + 0.5

        # Make sure the labels list is long enough
        for x in range(len(self._labels), shifted.shape[1]):
            self._labels.append(str(x))

        for i in list(range(0, x.count, self._batch_size)):
            start = i
            end = min(start + self._batch_size, x.count)
            mean = cp.mean(shifted[start:end, :], axis=0, keepdims=True)

            # For each column, report the metric
            metrics = {self._labels[y]: mean[0, y].item() for y in range(mean.shape[1])}

            metrics["total"] = cp.mean(mean).item()

            mlflow.log_metrics(metrics)

        return x

    def _build_single(self, input_stream: StreamPair) -> StreamPair:

        stream = input_stream[0]

        # Convert the messages to rows of strings
        stream = stream.async_map(self._calc_drift, executor=self._pipeline.thread_pool)

        # Return input unchanged
        return stream, MultiResponseMessage
