from __future__ import annotations

import re
from typing import Any


_AI_CLICHES = (
    "作为你的队友",
    "作为你的伙伴",
    "作为AI",
    "作为人工智能",
    "很高兴为你",
    "需要我帮忙吗",
    "有什么我可以帮",
    "请问还有什么",
    "希望我的回答",
    "如果还有问题",
    "让我们一起",
    "请注意安全",
    "建议您",
    "根据当前情况分析",
    "综合来看",
    "总而言之",
    "首先，",
    "其次，",
    "最后，",
)

_NUMERIC_REPORT_RE = re.compile(
    r"(玩家|你|乌枭|我)(的)?(HP|血量|生命)[为是:=：]?\s*\d+",
    re.I,
)


def strip_emotion_tag(text: str) -> tuple[str, str | None]:
    m = re.search(r"<emotion>([\w]+)</emotion>\s*$", text.strip(), re.I)
    if not m:
        return text.strip(), None
    emotion = m.group(1).lower()
    cleaned = re.sub(r"<emotion>[\w]+</emotion>\s*$", "", text.strip(), flags=re.I).strip()
    return cleaned, emotion


def anti_ai_postprocess(
    text: str,
    *,
    combat: bool = False,
    max_chars: int | None = None,
    recent_texts: list[str] | None = None,
) -> str:
    """去掉客服腔/报告腔，战斗中强制短句。"""
    out = (text or "").strip()
    if not out:
        return out

    for phrase in _AI_CLICHES:
        out = out.replace(phrase, "")

    # 去掉常见列表符号开头
    lines = []
    for line in out.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^[\-\*\d]+[\.\)、]\s*", "", stripped)
        lines.append(stripped)
    out = "".join(lines) if combat else "\n".join(lines)
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(r"^[\s,，、;；:：]+", "", out)
    out = re.sub(r"[，,]{2,}", "，", out)
    out = re.sub(r"^[「\"'“]+|[」\"'”]+$", "", out).strip()

    if combat:
        # 战斗：最多两句
        parts = re.split(r"(?<=[。！？!?])", out)
        kept = [p for p in parts if p.strip()][:2]
        out = "".join(kept).strip() or out
        limit = max_chars if max_chars is not None else 48
        if len(out) > limit:
            cut = out[:limit]
            # 尽量在标点处截断
            for i in range(len(cut) - 1, max(8, len(cut) - 20), -1):
                if cut[i] in "。！？!?，,":
                    cut = cut[: i + 1]
                    break
            out = cut

    # 弱化纯数值播报感
    if _NUMERIC_REPORT_RE.search(out) and "见底" not in out and "残" not in out:
        out = _NUMERIC_REPORT_RE.sub(
            lambda m: f"{m.group(1)}血线",
            out,
        )

    return out.strip()


def clip_for_combat(text: str, max_sentences: int = 2, max_chars: int = 48) -> str:
    return anti_ai_postprocess(text, combat=True, max_chars=max_chars)


def is_combat_scene(scene_info: dict[str, Any] | None) -> bool:
    scene_info = scene_info or {}
    if int(scene_info.get("enemy_count", 0) or 0) > 0:
        return True
    if scene_info.get("boss_alive"):
        return True
    if scene_info.get("player_in_danger") or scene_info.get("ally_in_danger"):
        return True
    if scene_info.get("hazard_near_player"):
        return True
    return False
