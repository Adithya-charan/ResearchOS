"""Retry policy: bounded attempts with exponential back-off."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int  # retries after the first attempt
    base_delay_s: float

    def can_retry(self, retries_used: int) -> bool:
        return retries_used < self.max_retries

    def delay_s(self, retries_used: int) -> float:
        """Wait before retry number `retries_used + 1`: base, 2*base, 4*base, ..."""
        return self.base_delay_s * (2**retries_used)
