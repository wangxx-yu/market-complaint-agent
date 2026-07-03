from __future__ import annotations

import re
from hashlib import sha256


PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")
ID_RE = re.compile(r"(?<!\d)(\d{17}[\dXx])(?!\d)")


def normalize_text(value: str | None, max_chars: int = 2000) -> str:
    if not value:
        return ""
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value[:max_chars]


def mask_pii(value: str | None) -> str:
    text = normalize_text(value, max_chars=4000)
    text = PHONE_RE.sub(lambda m: m.group(1)[:3] + "****" + m.group(1)[7:], text)
    text = ID_RE.sub(lambda m: m.group(1)[:6] + "********" + m.group(1)[14:], text)
    return text


def text_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:16]

