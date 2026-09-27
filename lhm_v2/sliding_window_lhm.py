"""Thin sliding-window adapter around the shared timestamp-based engine."""
from lhm_v2.features import FeatureEngine, WINDOW_SECONDS


class LhmSlidingWindow(FeatureEngine):
    def __init__(self, window_seconds=WINDOW_SECONDS):
        super().__init__(window_seconds=window_seconds)

    def add_sample(self, timestamp, canonical_sensors):
        return super().add_sample(timestamp, canonical_sensors)
