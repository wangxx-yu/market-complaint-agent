"""Prompt 模板加载器 — 从 prompts/ 目录读取 Jinja2 风格的模板。

支持变量替换: {variable_name}
"""

from __future__ import annotations

from pathlib import Path


PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

_cache: dict[str, str] = {}


def load_prompt(name: str) -> str:
    """加载 prompt 模板（不含扩展名）。"""
    if name not in _cache:
        path = PROMPTS_DIR / f"{name}.txt"
        if path.exists():
            _cache[name] = path.read_text(encoding="utf-8")
        else:
            _cache[name] = ""
    return _cache[name]


def render_prompt(name: str, **kwargs: str) -> str:
    """加载并渲染 prompt 模板。"""
    template = load_prompt(name)
    for key, value in kwargs.items():
        template = template.replace(f"{{{key}}}", value)
    return template


def list_prompts() -> list[str]:
    """列出所有可用的 prompt 模板名称。"""
    if not PROMPTS_DIR.exists():
        return []
    return [p.stem for p in PROMPTS_DIR.glob("*.txt")]
