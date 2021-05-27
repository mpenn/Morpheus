import threading


# Simple atomic integer from https://stackoverflow.com/a/48433648/634820
class AtomicInteger():
    def __init__(self, value=0):
        self._value = int(value)
        self._lock = threading.Lock()

    def inc(self, d=1):
        with self._lock:
            self._value += int(d)
            return self._value

    def dec(self, d=1):
        return self.inc(-d)

    def get_and_inc(self, d=1):
        """
        Gets the current value, returns it, and increments. Different from `inc()` which increments, then returns
        """
        with self._lock:
            tmp_val = self._value
            self._value += int(d)
            return tmp_val

    @property
    def value(self):
        with self._lock:
            return self._value

    @value.setter
    def value(self, v):
        with self._lock:
            self._value = int(v)
            return self._value