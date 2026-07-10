"""从 YAML 加载规则配置。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class RuleLoader:
    def __init__(self, rules_dir: Path | None = None) -> None:
        self.rules_dir = rules_dir or Path("data/rules")

    def load(self, name: str) -> list[dict[str, Any]]:
        path = self.rules_dir / f"{name}_rules.yaml"
        if not path.exists():
            return []
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return []
        rules = data.get("rules", [])
        rules.sort(key=lambda r: r.get("priority", 0), reverse=True)
        return rules

    def load_all(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "accept": self.load("accept"),
            "reject": self.load("reject"),
            "dispatch": self.load("dispatch"),
            "sensitive": self.load("sensitive"),
        }
