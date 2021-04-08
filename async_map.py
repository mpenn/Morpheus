from streamz import Stream
from tornado import gen


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
            r = self.func(x, *self.args, **self.kwargs)
            result = yield r
        except Exception as e:
            # logger.exception(e)
            print(e)
            raise
        else:
            return self._emit(result, metadata=metadata)