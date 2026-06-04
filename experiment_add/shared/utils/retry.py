from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar


T = TypeVar("T")


def retry_call(
    fn: Callable[[], T],
    max_retries: int = 2,
    sleep_seconds: float = 1.0,
    backoff: bool = False,
    logger: Any = None,
) -> T:
    """Call `fn` with retries and re-raise the final exception on failure.

    `max_retries` is the number of retries after the first attempt. With the
    default value of 2, the function may be called up to 3 times. Exceptions are
    logged when `logger` is provided. Waiting is fixed unless `backoff=True`,
    in which case the delay doubles after each failed attempt.
    """
    attempts = max(0, int(max_retries)) + 1
    delay = max(0.0, float(sleep_seconds))
    last_exc: BaseException | None = None

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if logger is not None:
                logger.warning("Attempt %d/%d failed: %s", attempt, attempts, exc)
            if attempt >= attempts:
                break
            if delay > 0:
                time.sleep(delay)
            if backoff:
                delay *= 2

    if logger is not None:
        logger.error("All %d attempts failed; re-raising last exception.", attempts)
    assert last_exc is not None
    raise last_exc
