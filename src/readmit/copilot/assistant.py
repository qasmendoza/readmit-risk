"""Care-transition copilot: a grounded briefing for care managers.

Pipeline: model risk + key drivers -> retrieve relevant guidance -> generate a short brief that
cites patient facts (``[Condition/...]``) and guidance (``[guide:heart-failure-transition]``).

Responsible-AI guardrails:
* the prompt restricts the model to the supplied facts and guidance and forbids diagnoses or
  medication changes (those remain clinical decisions);
* every citation in generated text is validated against the ids actually supplied; unknown
  citations are removed and reported as warnings, so fabricated evidence never reaches users;
* without an API key, or if the LLM call fails, a deterministic template brief is returned built
  from the same facts, so the service never hard-fails on an external dependency;
* only minimum-necessary, de-identified context is sent (see ``context.py``). With real PHI, use an
  LLM endpoint covered by a Business Associate Agreement and your organization's policies.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from readmit.copilot.context import PatientContext
from readmit.copilot.retriever import Hit, TfidfRetriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a care-transition assistant that helps hospital care managers prepare post-discharge "
    "outreach. Use only the patient facts and guidance documents provided. Cite patient facts with "
    "their ids in square brackets, e.g. [Condition/abc], and guidance with its id, e.g. "
    "[guide:follow-up-visit]. Do not diagnose, and do not recommend starting, stopping, or changing "
    "medications or doses; instead suggest topics for the care team to review. Write four short "
    "sections: Risk summary, Key drivers, Suggested transition-of-care actions, Caveats. Stay under "
    "250 words."
)
DISCLAIMER = (
    "Decision support only, not a diagnosis or treatment recommendation. Developed and evaluated on "
    "synthetic data; not validated for clinical use."
)
_CITATION = re.compile(r"\[([^\[\]]+)\]")


@dataclass
class Brief:
    text: str
    mode: Literal["llm", "template"]
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    guidance: list[dict[str, Any]] = field(default_factory=list)


def make_llm_client() -> Any | None:
    """Anthropic client when ``ANTHROPIC_API_KEY`` is set and the SDK is installed."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        logger.warning("ANTHROPIC_API_KEY is set but the 'anthropic' package is not installed")
        return None
    return anthropic.Anthropic()


def validate_citations(text: str, allowed: set[str]) -> tuple[str, list[str], list[str]]:
    """Remove citations that do not refer to supplied evidence."""
    valid: list[str] = []
    invalid: list[str] = []

    def replace(match: re.Match[str]) -> str:
        kept = []
        for token in re.split(r"[;,]\s*", match.group(1)):
            token = token.strip()
            (kept if token in allowed else invalid).append(token)
        valid.extend(kept)
        return f"[{', '.join(kept)}]" if kept else ""

    cleaned = _CITATION.sub(replace, text)
    return re.sub(r" {2,}", " ", cleaned).strip(), list(dict.fromkeys(valid)), invalid


class CareCopilot:
    def __init__(
        self,
        guidance_docs: list[dict[str, Any]],
        llm_client: Any | None = None,
        model: str | None = None,
        max_tokens: int = 700,
        top_k: int = 3,
    ) -> None:
        self.retriever = TfidfRetriever(guidance_docs)
        self.llm_client = llm_client
        self.model = model or os.environ.get("READMIT_LLM_MODEL", "claude-haiku-5-5")
        self.max_tokens = max_tokens
        self.top_k = top_k

    def guidance_for(self, context: PatientContext, factors: list[dict[str, Any]]) -> list[Hit]:
        terms = [f["description"] for f in factors]
        terms += [
            f.text.split(" (documented")[0]
            for f in context.facts
            if f.kind in ("condition", "social_need")
        ]
        query = " ".join(terms) or "follow-up appointment after discharge"
        return self.retriever.search(query, top_k=self.top_k)

    def _prompt(
        self,
        context: PatientContext,
        risk: float,
        tier: str,
        factors: list[dict[str, Any]],
        hits: list[Hit],
    ) -> str:
        drivers = (
            "\n".join(
                f"- {f['text']} (adds ~{f['contribution'] * 100:.1f} percentage points)"
                for f in factors
            )
            or "- none above typical"
        )
        guidance = "\n\n".join(
            f"[guide:{h.doc['id']}] {h.doc['title']}: {h.doc['text']}" for h in hits
        )
        return (
            f"Model output: 30-day readmission risk {risk:.0%} ({tier} tier).\n"
            f"Key drivers from the model:\n{drivers}\n\n"
            f"Patient facts:\n{context.render()}\n\nGuidance documents:\n{guidance}"
        )

    def _template(
        self,
        context: PatientContext,
        risk: float,
        tier: str,
        factors: list[dict[str, Any]],
        hits: list[Hit],
    ) -> str:
        index_fact = next((f for f in context.facts if f.kind == "encounter"), None)
        lines = [
            "Risk summary: estimated 30-day readmission risk "
            f"{risk:.0%} ({tier} tier){f' [{index_fact.id}]' if index_fact else ''}.",
            "",
            "Key drivers:",
        ]
        for factor in factors:
            evidence = context.facts_for(factor["feature"])
            cite = f" [{', '.join(f.id for f in evidence)}]" if evidence else ""
            lines.append(f"- {factor['text']}{cite}")
        if not factors:
            lines.append("- No individual factor stands out above a typical patient.")
        lines += ["", "Suggested transition-of-care actions (for care-team review):"]
        for hit in hits:
            first_sentence = hit.doc["text"].split(". ")[0].rstrip(".")
            lines.append(f"- {hit.doc['title']}: {first_sentence}. [guide:{hit.doc['id']}]")
        lines += ["", f"Caveats: {DISCLAIMER}"]
        return "\n".join(lines)

    def brief(
        self, context: PatientContext, risk: float, tier: str, factors: list[dict[str, Any]]
    ) -> Brief:
        hits = self.guidance_for(context, factors)
        allowed = context.ids | {f"guide:{h.doc['id']}" for h in hits}
        guidance = [
            {
                "id": h.doc["id"],
                "title": h.doc["title"],
                "score": round(h.score, 4),
                "references": h.doc.get("references", ""),
            }
            for h in hits
        ]
        if self.llm_client is not None:
            try:
                response = self.llm_client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": self._prompt(context, risk, tier, factors, hits),
                        }
                    ],
                )
                raw = "".join(
                    getattr(b, "text", "")
                    for b in response.content
                    if getattr(b, "type", "text") == "text"
                ).strip()
                if not raw:
                    raise ValueError("empty LLM response")
                text, valid, invalid = validate_citations(raw, allowed)
                if invalid:
                    logger.warning(
                        "removed %d unsupported citation(s) from LLM output", len(invalid)
                    )
                return Brief(f"{text}\n\n{DISCLAIMER}", "llm", valid, invalid, guidance)
            except Exception:
                logger.exception("LLM generation failed; using template brief")
        text = self._template(context, risk, tier, factors, hits)
        _, valid, _ = validate_citations(text, allowed)
        return Brief(text, "template", valid, [], guidance)
