"""
Backward-compatible re-exports.

The two-GPU sentence batcher was replaced by ``mixed_serving.GlobalWatermarkQueue``
(FIFO per-sentence queue + adaptive mixed batch admission in Generation.py).
"""

from mixed_serving import (
    AdaptiveRatioController,
    GlobalWatermarkQueue,
    SyncGlobalWatermarkQueue,
    WatermarkQueueItem,
)

__all__ = [
    "AdaptiveRatioController",
    "GlobalWatermarkQueue",
    "SyncGlobalWatermarkQueue",
    "WatermarkQueueItem",
]
