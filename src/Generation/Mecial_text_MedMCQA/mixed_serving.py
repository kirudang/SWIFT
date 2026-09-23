from __future__ import annotations

import asyncio
import collections
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Set, Tuple

RequestType = Literal["generation", "watermark"]


@dataclass(frozen=True)
class WatermarkQueueItem:
    """One completed sentence segment waiting for watermarking."""

    prompt_id: int
    sent_idx: int
    sentence_text: str
    enqueue_time: float


class AdaptiveRatioController:
    """Maintains watermark admission ratio ``r_t``, updated every ``K`` intervals."""

    def __init__(
        self,
        *,
        r_min: float = 0.0,
        r_max: float = 1.0,
        b_max: float = 64.0,
        w_max: float = 10.0,
        k: int = 10,
    ) -> None:
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.b_max = max(1.0, float(b_max))
        self.w_max = max(1e-6, float(w_max))
        self.k = max(1, int(k))
        self.r_t = self.r_min
        self._interval_count = 0

    def compute_pressure(self, queue_len: int, oldest_wait_s: float) -> float:
        if queue_len <= 0:
            return 0.0
        b_t = int(queue_len)
        w_t = float(oldest_wait_s)
        pressure = max(b_t / self.b_max, w_t / self.w_max)
        return min(max(pressure, 0.0), 1.0)

    def maybe_update(self, queue_len: int, oldest_wait_s: float) -> float:
        """Persist ``r_t`` every ``K`` scheduling intervals (for monitoring)."""
        self._interval_count += 1
        if queue_len <= 0:
            self.r_t = self.r_min
            return self.r_t
        if self._interval_count % self.k == 0:
            pressure = self.compute_pressure(queue_len, oldest_wait_s)
            self.r_t = self.r_min + (self.r_max - self.r_min) * pressure
        return self.r_t

    def admission_ratio(self, queue_len: int, oldest_wait_s: float) -> float:
        """
        Ratio used for slot allocation on every schedule call.
        Reacts immediately when the queue is non-empty; ``r_t`` itself updates every K.
        """
        if queue_len <= 0:
            return self.r_min
        pressure = self.compute_pressure(queue_len, oldest_wait_s)
        return self.r_min + (self.r_max - self.r_min) * pressure


class _GlobalWatermarkQueueBase:
    def __init__(self) -> None:
        self._deque: collections.deque[WatermarkQueueItem] = collections.deque()

    def _enqueue_locked(self, prompt_id: int, sent_idx: int, text: str) -> WatermarkQueueItem:
        item = WatermarkQueueItem(
            prompt_id=int(prompt_id),
            sent_idx=int(sent_idx),
            sentence_text=text,
            enqueue_time=time.time(),
        )
        self._deque.append(item)
        return item

    def _len_locked(self) -> int:
        return len(self._deque)

    def _oldest_wait_locked(self) -> float:
        if not self._deque:
            return 0.0
        return time.time() - self._deque[0].enqueue_time

    def _dequeue_locked(self) -> Optional[WatermarkQueueItem]:
        if self._deque:
            return self._deque.popleft()
        return None


class GlobalWatermarkQueue(_GlobalWatermarkQueueBase):
    """Thread-safe async FIFO watermark queue."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = asyncio.Lock()

    async def enqueue(self, prompt_id: int, sent_idx: int, text: str) -> WatermarkQueueItem:
        async with self._lock:
            return self._enqueue_locked(prompt_id, sent_idx, text)

    async def dequeue(self) -> Optional[WatermarkQueueItem]:
        async with self._lock:
            return self._dequeue_locked()

    async def __len__(self) -> int:
        async with self._lock:
            return self._len_locked()

    async def oldest_wait_s(self) -> float:
        async with self._lock:
            return self._oldest_wait_locked()


class SyncGlobalWatermarkQueue(_GlobalWatermarkQueueBase):
    """Thread-safe sync FIFO watermark queue."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def enqueue(self, prompt_id: int, sent_idx: int, text: str) -> WatermarkQueueItem:
        with self._lock:
            return self._enqueue_locked(prompt_id, sent_idx, text)

    def dequeue(self) -> Optional[WatermarkQueueItem]:
        with self._lock:
            return self._dequeue_locked()

    def __len__(self) -> int:
        with self._lock:
            return self._len_locked()

    def oldest_wait_s(self) -> float:
        with self._lock:
            return self._oldest_wait_locked()


@dataclass
class ServingRequestMeta:
    request_id: str
    request_type: RequestType
    prompt_id: int
    sent_idx: Optional[int] = None


def compute_watermark_admission(
    *,
    r_t: float,
    batch_size: int,
    queue_len: int,
    active_watermark: int,
) -> int:
    """Return how many new watermark sequences to admit this scheduling step."""
    if queue_len <= 0:
        return 0
    target_wm = int(r_t * batch_size)
    # int() truncates small ratios to 0; always admit at least one waiter.
    if target_wm == 0:
        target_wm = 1
    target_wm = min(target_wm, queue_len)
    return max(0, target_wm - active_watermark)


def compute_available_gen_slots(
    *,
    batch_size: int,
    active_generation: int,
    active_watermark: int,
) -> int:
    return max(0, batch_size - active_generation - active_watermark)
