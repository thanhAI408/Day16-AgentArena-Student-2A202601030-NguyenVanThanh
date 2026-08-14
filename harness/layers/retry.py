"""Tool retry layer for Agent Arena."""

from __future__ import annotations

from arena.model import is_degraded

from harness.middleware import Middleware

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RESERVE = 1


class Retry(Middleware):
    """Retry failed/degraded tool results below the model."""

    name = "retry"

    def __init__(
        self,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        reserve: int = DEFAULT_RESERVE,
    ) -> None:
        self.max_attempts = max(1, int(max_attempts))
        self.reserve = max(0, int(reserve))

    def wrap_tool_call(self, ctx, call, name, args):
        attempts = 1
        result = call(name, args)

        while (
            attempts < self.max_attempts
            and ((not result.ok) or is_degraded(result.content))
        ):
            limit = ctx.max_tool_calls
            if (
                limit is not None
                and ctx.tools.calls >= limit - self.reserve
            ):
                break
            result = call(name, args)
            attempts += 1

        ctx.state["retry_attempts"] = ctx.state.get("retry_attempts", 0) + attempts
        ctx.state["retry_extra_calls"] = (
            ctx.state.get("retry_extra_calls", 0) + attempts - 1
        )
        return result
