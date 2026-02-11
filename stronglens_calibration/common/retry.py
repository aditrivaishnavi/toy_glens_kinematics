from __future__ import annotations
import time
import random
from typing import Callable, TypeVar, Tuple

T = TypeVar("T")

class RetryError(RuntimeError):
    pass

def with_retries(
    fn: Callable[[], T],
    *,
    max_tries: int = 6,
    base_sleep_s: float = 0.5,
    max_sleep_s: float = 10.0,
    jitter: float = 0.2,
    retry_on: Tuple[type, ...] = (Exception,),
) -> T:
    last_exc: Exception | None = None
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except retry_on as e:
            last_exc = e
            if attempt == max_tries:
                break
            sleep = min(max_sleep_s, base_sleep_s * (2 ** (attempt - 1)))
            sleep *= (1.0 + random.uniform(-jitter, jitter))
            time.sleep(max(0.0, sleep))
    raise RetryError(f"Failed after {max_tries} tries") from last_exc
