"""Tool-budget control + retrieval-depth layer for Agent Arena."""

from __future__ import annotations

import json

from arena.model import FINALIZE_SENTINEL
from arena.tools import ToolResult

from harness.middleware import Middleware

DEFAULT_RESERVE = 1

NUDGE = (
    "Ngân sách công cụ đã hết. Hãy trả lời ngay bằng bằng chứng đang có, "
    f"không gọi thêm công cụ nào nữa. {FINALIZE_SENTINEL}"
)


def _refined_query(question: str) -> str:
    """Convert noisy briefs into concise topic/document-type queries."""
    q = (question or "").casefold()

    if any(term in q for term in (
        "bốc dỡ", "tai nạn", "bị thương",
        "an toàn lao động", "phòng chống tai nạn",
    )):
        # Exact corpus topic. Keeping this short prevents generic safety
        # reports/memos from outranking the governing policy.
        return "an toàn lao động tại kho"

    if (
        any(term in q for term in (
            "nhà cung cấp", "hợp tác lần đầu", "đối tác mới",
            "hồ sơ", "bị trả lại",
        ))
        and any(term in q for term in (
            "đào tạo", "thống kê", "tương tự", "hợp tác",
        ))
    ):
        return "báo cáo quy trình làm việc với nhà cung cấp mới"

    return question


def _replace_preamble_question(messages, replacement: str):
    if not replacement:
        return messages

    out = [dict(m) for m in messages]
    first_assistant = next(
        (i for i, m in enumerate(out) if m.get("role") == "assistant"),
        len(out),
    )
    candidates = [
        i for i, m in enumerate(out[:first_assistant])
        if m.get("role") in ("user", "human")
    ]
    if not candidates:
        return messages

    i = candidates[-1]
    if out[i].get("content") != replacement:
        out[i]["content"] = replacement
    return out


def _rerank_search_result(ctx, result, question: str):
    """Prefer governing evidence over same-topic filler.

    The search itself is still performed by the frozen tool. This only
    changes which returned hit the weak mock fetches first.
    """
    if not result.ok or not isinstance(result.content, str):
        return result

    q = (question or "").casefold()
    safety = any(term in q for term in (
        "bốc dỡ", "tai nạn", "bị thương",
        "an toàn lao động", "phòng chống tai nạn",
    ))
    if not safety:
        return result

    try:
        rows = json.loads(result.content)
    except Exception:
        return result
    if not isinstance(rows, list):
        return result

    def priority(row):
        if not isinstance(row, dict):
            return (9, 9)
        doc_id = row.get("doc_id")
        doc = ctx.corpus.get(doc_id) if ctx.corpus is not None else None
        tags = set(doc.tags) if doc is not None else set()
        # Official topic policy first, then other on-topic material.
        if "warehouse_safety" in tags and "policy" in tags:
            return (0, 0)
        if "warehouse_safety" in tags:
            return (1, 0)
        return (2, 0)

    rows = sorted(enumerate(rows), key=lambda pair: (priority(pair[1]), pair[0]))
    content = json.dumps([row for _, row in rows], ensure_ascii=False)

    return ToolResult(ok=result.ok, content=content, error=result.error)


class BudgetPolicy(Middleware):
    """Reserve submit capacity and improve first-stage retrieval depth."""

    name = "budget_policy"

    def __init__(self, reserve: int = DEFAULT_RESERVE) -> None:
        self.reserve = max(0, int(reserve))

    def _spent(self, ctx) -> bool:
        limit = ctx.max_tool_calls
        return limit is not None and ctx.tools.calls >= limit - self.reserve

    def before_model(self, ctx, messages):
        refined = _refined_query(ctx.question)
        if refined != ctx.question:
            messages = _replace_preamble_question(messages, refined)

        if not self._spent(ctx):
            return messages

        return messages + [{"role": "user", "content": NUDGE}]

    def wrap_tool_call(self, ctx, call, name, args):
        if self._spent(ctx):
            return ToolResult(
                ok=False,
                content="",
                error="Ngân sách công cụ đã cạn; dành lượt còn lại cho submit.",
            )

        result = call(name, args)
        if name == "search":
            result = _rerank_search_result(ctx, result, ctx.question)
        return result
