import cudf
from morpheus.pipeline.pipeline import StreamPair
from streamz.core import Stream
from tornado.ioloop import IOLoop
from morpheus.pipeline import SourceStage
from morpheus.config import Config

class KafkaSourceStage(SourceStage):
    def __init__(self,
                 c: Config,
                 bootstrap_servers: str,
                 input_topic: str = "test_pcap",
                 group_id: str = "custreamz",
                 use_dask: bool = False,
                 poll_interval: str = "10millis"):
        super().__init__(c)

        self._consumer_conf = {'bootstrap.servers': bootstrap_servers, 'group.id': group_id, 'session.timeout.ms': "60000"}

        self._input_topic = input_topic
        self._use_dask = use_dask
        self._poll_interval = poll_interval
        self._max_batch_size = c.pipeline_batch_size

    @property
    def name(self) -> str:
        return "from-kafka"

    async def _build(self) -> StreamPair:

        if (self._use_dask):
            from dask.distributed import Client
            client = Client()

            source: Stream = Stream.from_kafka_batched(self._input_topic,
                                                       self._consumer_conf,
                                                       npartitions=None,
                                                       start=False,
                                                       asynchronous=True,
                                                       dask=True,
                                                       engine="cudf",
                                                       poll_interval=self._poll_interval,
                                                       loop=IOLoop.current(),
                                                       max_batch_size=self._max_batch_size)
        else:
            source: Stream = Stream.from_kafka_batched(self._input_topic,
                                                       self._consumer_conf,
                                                       npartitions=None,
                                                       start=False,
                                                       asynchronous=True,
                                                       dask=False,
                                                       engine="cudf",
                                                       poll_interval=self._poll_interval,
                                                       loop=IOLoop.current(),
                                                       max_batch_size=self._max_batch_size)

        # Always gather here (no-op if not using dask)
        return source.gather(), cudf.DataFrame