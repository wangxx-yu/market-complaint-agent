from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Any, Iterator

from app.core.schemas import AgentStep
from app.core.text import mask_pii


@contextmanager
def agent_step(
    steps: list[AgentStep],
    name: str,
    input_summary: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    started = perf_counter()
    state: dict[str, Any] = {"output_summary": {}, "confidence": None, "error": None, "degraded": False}
    try:
        yield state
    except Exception as exc:
        state["error"] = str(exc)
        state["degraded"] = True
        raise
    finally:
        duration_ms = (perf_counter() - started) * 1000
        safe_input = {key: mask_pii(str(value)) for key, value in (input_summary or {}).items()}
        steps.append(
            AgentStep(
                name=name,
                input_summary=safe_input,
                output_summary=state["output_summary"],
                confidence=state["confidence"],
                duration_ms=round(duration_ms, 2),
                error=state["error"],
                degraded=state["degraded"],
            )
        )

