"""Prompt-injection defence layer for Agent Arena."""

from __future__ import annotations

from arena.corpus import INJECTION_CANARY
from arena.tools import ToolResult

from harness.middleware import Middleware

BLOCK_START = "--- LƯU Ý HỆ THỐNG"
BLOCK_END = "--- HẾT LƯU Ý HỆ THỐNG ---"
PLACEHOLDER = "[nội dung không đáng tin cậy đã bị loại bỏ]"


class InjectionGuard(Middleware):
    """Quarantine untrusted document instructions and sweep the final answer."""

    name = "injection_guard"

    def wrap_tool_call(self, ctx, call, name, args):
        result = call(name, args)
        content = result.content

        if BLOCK_START not in content:
            return result

        while BLOCK_START in content:
            start = content.find(BLOCK_START)
            end = content.find(BLOCK_END, start + len(BLOCK_START))
            if end < 0:
                content = content[:start] + PLACEHOLDER
                break
            content = (
                content[:start]
                + PLACEHOLDER
                + content[end + len(BLOCK_END):]
            )

        return ToolResult(ok=result.ok, content=content, error=result.error)

    def after_agent(self, ctx, report):
        answer = report.get("answer")
        if isinstance(answer, str) and INJECTION_CANARY in answer:
            report["answer"] = answer.replace(INJECTION_CANARY, PLACEHOLDER)
        return report
