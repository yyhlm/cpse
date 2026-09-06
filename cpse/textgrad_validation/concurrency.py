"""Cancellation-aware daemon worker utilities for long PDF API calls."""

from __future__ import annotations

import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.thread import _threads_queues, _worker


class RunCancelled(RuntimeError):
    """Raised by cooperative workers after the active run receives Ctrl+C."""


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """Thread pool whose stuck network workers never delay interpreter exit."""

    def _adjust_thread_count(self) -> None:
        if not (
            hasattr(self, "_work_queue")
            and hasattr(self, "_idle_semaphore")
            and hasattr(self, "_max_workers")
            and hasattr(self, "_threads")
            and hasattr(self, "_initializer")
            and hasattr(self, "_initargs")
            and hasattr(self, "_thread_name_prefix")
        ):
            super()._adjust_thread_count()
            return
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            thread = threading.Thread(
                name=thread_name,
                target=_worker,
                args=(weakref.ref(self, weakref_cb), self._work_queue, self._initializer, self._initargs),
                daemon=True,
            )
            thread.start()
            self._threads.add(thread)
            _threads_queues[thread] = self._work_queue
