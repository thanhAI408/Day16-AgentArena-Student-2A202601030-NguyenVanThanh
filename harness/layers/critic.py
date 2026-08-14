"""Evidence critic layer for Agent Arena."""

from __future__ import annotations

from harness.middleware import Middleware


def _matches_one_line(doc, text: str) -> bool:
    return bool(text) and any(text in line for line in doc.body.splitlines())


def _observed_sources(ctx, text: str):
    if ctx.corpus is None or not ctx.saw(text):
        return []
    observed = ctx.observed_text
    return [
        doc
        for doc in ctx.corpus.docs
        if doc.body in observed and _matches_one_line(doc, text)
    ]


def _split_contradiction(ctx, text: str):
    marker = " và "
    start_at = 0

    while True:
        index = text.find(marker, start_at)
        if index < 0:
            return None

        left = text[:index].strip()
        right = text[index + len(marker):].strip()
        if left and right and ctx.saw(left) and ctx.saw(right):
            left_docs = _observed_sources(ctx, left)
            right_docs = _observed_sources(ctx, right)
            for left_doc in left_docs:
                for right_doc in right_docs:
                    if left_doc.doc_id != right_doc.doc_id:
                        return (
                            {"text": left, "doc_id": left_doc.doc_id},
                            {"text": right, "doc_id": right_doc.doc_id},
                        )

        start_at = index + 1


class Critic(Middleware):
    """Delete unsupported claims and abstain when evidence is insufficient."""

    name = "critic"

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims:
            return report

        kept = []
        split_conflict = False

        for claim in claims:
            if not isinstance(claim, dict):
                continue
            text = claim.get("text")
            if not isinstance(text, str) or not text:
                continue

            if ctx.saw(text):
                kept.append(claim)
                continue

            split = _split_contradiction(ctx, text)
            if split is not None:
                kept.extend(split)
                split_conflict = True

        report["claims"] = kept

        if split_conflict:
            report["abstain"] = True

        if not kept:
            report["abstain"] = True
            report["claims"] = []
            report["citations"] = []
            report["answer"] = (
                "Không đủ căn cứ trong các tài liệu đã quan sát "
                "để đưa ra kết luận đáng tin cậy."
            )
            return report

        report["citations"] = sorted(
            {
                claim.get("doc_id")
                for claim in kept
                if isinstance(claim.get("doc_id"), str)
                and claim.get("doc_id")
            }
        )
        return report
