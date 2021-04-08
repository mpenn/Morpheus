from time import time
from streamz import Stream
from streamz.core import convert_interval
from tornado import gen
import asyncio

from tornado.queues import Queue

@Stream.register_api()
class async_map(Stream):
    """ Apply a function to every element in the stream

    Parameters
    ----------
    func: callable
    *args :
        The arguments to pass to the function.
    **kwargs:
        Keyword arguments to pass to func

    Examples
    --------
    >>> source = Stream()
    >>> source.map(lambda x: 2*x).sink(print)
    >>> for i in range(5):
    ...     source.emit(i)
    0
    2
    4
    6
    8
    """
    def __init__(self, upstream, func, *args, **kwargs):
        self.func = func
        # this is one of a few stream specific kwargs
        stream_name = kwargs.pop('stream_name', None)
        self.executor = kwargs.pop('executor', None)
        self.kwargs = kwargs
        self.args = args

        Stream.__init__(self, upstream, stream_name=stream_name, ensure_io_loop=True)

        if (self.executor is not None):

            loop = self.loop
            executor = self.executor

            
            async def get_value_with_executor(x, *inner_args, **inner_kwargs):
                return await loop.run_in_executor(executor, func, x, *inner_args, **inner_kwargs)

            self.func = get_value_with_executor

    @gen.coroutine
    def update(self, x, who=None, metadata=None):
        try:
            self._retain_refs(metadata)
            r = self.func(x, *self.args, **self.kwargs)
            result = yield r
        except Exception as e:
            # logger.exception(e)
            print(e)
            raise
        else:
            emit = yield self._emit(result, metadata=metadata)

            # return emit
        finally:
            self._release_refs(metadata)


@Stream.register_api()
class time_delay(Stream):
    """ Add a time delay to results """
    _graphviz_shape = 'octagon'

    def __init__(self, upstream, interval, **kwargs):
        self.interval = convert_interval(interval)
        self.queue = Queue()

        kwargs["ensure_io_loop"] = True
        Stream.__init__(self, upstream,**kwargs)

        self.loop.add_callback(self.cb)

    @gen.coroutine
    def cb(self):
        while True:
            q_time, x, metadata = yield self.queue.get()
            
            duration = self.interval - (time() - q_time)

            if duration > 0:
                yield gen.sleep(duration)

            yield self._emit(x, metadata=metadata)
            
            self._release_refs(metadata)

    def update(self, x, who=None, metadata=None):
        self._retain_refs(metadata)
        return self.queue.put((time(), x, metadata))
