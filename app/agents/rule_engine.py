"""规则引擎 — 基于 YAML 规则进行关键词匹配。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.rule_loader import RuleLoader


@dataclass
class RuleMatch:
    rule_id: str
    rule_name: str
    decision: str  # ACCEPT / REJECT / REVIEW
    reason_type: str
    priority: int
    matched_keywords: list[str]
    review_required: bool
    source: str
    suggest_department: str | None = None
    note: str | None = None


@dataclass
class RuleEngineResult:
    matches: list[RuleMatch] = field(default_factory=list)
    highest_priority_match: RuleMatch | None = None

    @property
    def has_match(self) -> bool:
        return len(self.matches) > 0


class RuleEngine:
    def __init__(self, loader: RuleLoader | None = None) -> None:
        self.loader = loader or RuleLoader()
        self._rules: dict[str, list[dict[str, Any]]] = {}
        self.reload()

    def reload(self) -> None:
        self._rules = self.loader.load_all()

    def match_accept(self, text: str) -> RuleEngineResult:
        return self._match(text, self._rules.get("accept", []))

    def match_reject(self, text: str) -> RuleEngineResult:
        return self._match(text, self._rules.get("reject", []))

    def match_sensitive(self, text: str) -> RuleEngineResult:
        return self._match(text, self._rules.get("sensitive", []))

    def match_dispatch(self, text: str) -> RuleEngineResult:
        return self._match(text, self._rules.get("dispatch", []))

    def _match(self, text: str, rules: list[dict[str, Any]]) -> RuleEngineResult:
        matches: list[RuleMatch] = []
        for rule in rules:
            keywords = rule.get("keywords", [])
            exclude_keywords = rule.get("exclude_keywords", [])
            if exclude_keywords and any(kw in text for kw in exclude_keywords):
                continue
            matched = [kw for kw in keywords if kw in text]
            if matched:
                matches.append(RuleMatch(
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    decision=rule.get("decision", "REVIEW"),
                    reason_type=rule.get("reason_type", "UNKNOWN"),
                    priority=rule.get("priority", 50),
                    matched_keywords=matched,
                    review_required=rule.get("review_required", True),
                    source=rule.get("source", ""),
                    suggest_department=rule.get("suggest_department"),
                    note=rule.get("note"),
                ))
        matches.sort(key=lambda m: m.priority, reverse=True)
        return RuleEngineResult(
            matches=matches,
            highest_priority_match=matches[0] if matches else None,
        )
