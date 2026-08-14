"""Citation re-attribution layer for Agent Arena."""

from __future__ import annotations

from harness.middleware import Middleware


def _matches_one_line(doc, text: str) -> bool:
    return bool(text) and any(text in line for line in doc.body.splitlines())


class CitationChecker(Middleware):
    """Point every observed claim at the observed document that contains it."""

    name = "citation_checker"

    def after_agent(self, ctx, report):
        claims = report.get("claims")
        if not isinstance(claims, list) or not claims or ctx.corpus is None:
            return report

        observed = ctx.observed_text

        for claim in claims:
            if not isinstance(claim, dict):
                continue
            text = claim.get("text")
            if not isinstance(text, str) or not text:
                continue

            current = ctx.corpus.get(claim.get("doc_id"))
            if current is not None and _matches_one_line(current, text):
                continue

            if text not in observed:
                continue

            for doc in ctx.corpus.docs:
                if doc.body in observed and _matches_one_line(doc, text):
                    claim["doc_id"] = doc.doc_id
                    break

        report["citations"] = sorted(
            {
                claim.get("doc_id")
                for claim in claims
                if isinstance(claim, dict)
                and isinstance(claim.get("doc_id"), str)
                and claim.get("doc_id")
            }
        )
        return report
