import concurrent.futures 
import dataclasses 
import itertools

import constants


_PREFETCH_EXEC = concurrent.futures.ThreadPoolExecutor(
    max_workers=constants.prefetch_workers
)


@dataclasses.dataclass
class PrefetchHandle:
    _future: concurrent.futures.Future

    def get(self, timeout=None):
        return self._future.result(timeout=timeout)

    def done(self):
        return self._future.done()

    def cancel(self):
        return self._future.cancel()


def submit(task):
    fut = _PREFETCH_EXEC.submit(task)
    return PrefetchHandle(fut)


def shutdown_prefetch():
    _PREFETCH_EXEC.shutdown(wait=False, cancel_futures=True)
