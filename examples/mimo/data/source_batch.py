# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Source-global-batch iteration for replicated multimodal example datasets."""

from concurrent.futures import ThreadPoolExecutor
from weakref import finalize


class SourceBatchIterator:
    """Resume a deterministic dataset at a complete source-global-batch boundary.

    The dataset implements ``build_global_batch(step, global_batch_size)`` and
    returns ``(samples, media)``. This is an example dataset contract, independent
    of the core runtime and the number of decoder rounds selected by packing.
    Optional CPU prefetch keeps at most one unconsumed batch, including work in
    progress. ``step`` advances only when a batch is returned; checkpoint resume
    still uses the host's consumed sample count, never prefetched progress.
    """

    def __init__(self, dataset, global_batch_size, consumed_samples=0, *, prefetch=False) -> None:
        if global_batch_size <= 0 or consumed_samples < 0 or consumed_samples % global_batch_size:
            raise ValueError("MIMO resume requires a complete source global batch")
        self.dataset = dataset
        self.global_batch_size = global_batch_size
        self.step = consumed_samples // global_batch_size
        self._closed = False
        self._future = None
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="mimo-source")
            if prefetch
            else None
        )
        self._finalizer = (
            finalize(self, self._executor.shutdown, wait=False, cancel_futures=True)
            if self._executor is not None
            else None
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise RuntimeError("Source batch iterator is closed")
        try:
            if self._executor is None:
                samples, media = self.dataset.build_global_batch(self.step, self.global_batch_size)
            else:
                if self._future is None:
                    self._future = self._executor.submit(
                        self.dataset.build_global_batch, self.step, self.global_batch_size
                    )
                samples, media = self._future.result()
                self._future = self._executor.submit(
                    self.dataset.build_global_batch, self.step + 1, self.global_batch_size
                )
            result = dict(step=self.step, samples=samples, media=media)
            self.step += 1
            return result
        except BaseException:
            self.close()
            raise

    def close(self):
        """Discard unread lookahead and finish any in-flight CPU preparation."""
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._finalizer.detach()
        self._future = None
