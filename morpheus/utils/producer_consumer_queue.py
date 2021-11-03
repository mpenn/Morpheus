import queue
import threading
import typing
from time import time

_T = typing.TypeVar("_T")


class Closed(Exception):
    'Exception raised when the queue is closed'
    pass


class ProducerConsumerQueue(queue.Queue, typing.Generic[_T]):
    """
    Custom queue.Queue implementation which supports closing and uses recursive locks
    """
    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)

        # Use a recursive lock here to prevent reentrant deadlocks
        self.mutex = threading.RLock()

        self.not_empty = threading.Condition(self.mutex)
        self.not_full = threading.Condition(self.mutex)
        self.all_tasks_done = threading.Condition(self.mutex)

        self._is_closed = False

    def join(self):
        """
        Blocks until the queue has been closed and all tasks are completed
        """
        with self.all_tasks_done:
            while not self._is_closed and self.unfinished_tasks:
                self.all_tasks_done.wait()

    def put(self, item: _T, block: bool = True, timeout: typing.Optional[float] = None) -> None:
        with self.not_full:
            if self.maxsize > 0:
                if not block:
                    if self._qsize() >= self.maxsize and not self._is_closed:
                        raise queue.Full  # @IgnoreException
                elif timeout is None:
                    while self._qsize() >= self.maxsize and not self._is_closed:
                        self.not_full.wait()
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    endtime = time() + timeout
                    while self._qsize() >= self.maxsize and not self._is_closed:
                        remaining = endtime - time()
                        if remaining <= 0.0:
                            raise queue.Full  # @IgnoreException
                        self.not_full.wait(remaining)

            if (self._is_closed):
                raise Closed  # @IgnoreException

            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()

    def get(self, block: bool = True, timeout: typing.Optional[float] = None) -> _T:
        with self.not_empty:
            if not block:
                if not self._qsize() and not self._is_closed:
                    raise queue.Empty  # @IgnoreException
            elif timeout is None:
                while not self._qsize() and not self._is_closed:
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time() + timeout
                while not self._qsize() and not self._is_closed:
                    remaining = endtime - time()
                    if remaining <= 0.0:
                        raise queue.Empty  # @IgnoreException
                    self.not_empty.wait(remaining)

            if (self._is_closed and not self._qsize()):
                raise Closed  # @IgnoreException

            item = self._get()
            self.not_full.notify()
            return item

    def close(self):
        with self.mutex:
            if (not self._is_closed):
                self._is_closed = True
                self.not_full.notify_all()
                self.not_empty.notify_all()
                self.all_tasks_done.notify_all()

    def is_closed(self):
        with self.mutex:
            return self._is_closed
