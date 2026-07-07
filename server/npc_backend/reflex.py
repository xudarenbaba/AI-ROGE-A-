from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ReflexDecision:
    type: Literal["dialogue", "command", "noop"]
    stance: str | None = None
    reply: str | None = None


def reflex_decide(
    scene_info: dict[str, Any],
    *,
    reflex_enabled: bool = True,
) -> ReflexDecision | None:
    """规则快路径：危急局面零 LLM 响应（<1ms）。"""
    if not reflex_enabled:
        return None

    ally_hp = int(scene_info.get("ally_hp", 100))
    player_hp = int(scene_info.get("player_hp", 100))
    player_max = float(scene_info.get("player_max_hp", 160) or 160)
    ally_stance = str(scene_info.get("ally_stance", "guard"))
    enemy_count = int(scene_info.get("enemy_count", 0))
    floor_state = str(scene_info.get("floor_state", "playing"))

    if floor_state != "playing":
        return ReflexDecision(type="noop")

    if ally_hp <= 0:
        if ally_stance == "assault":
            return ReflexDecision(
                type="command",
                stance="guard",
                reply="灵核快散了，我先收回来护着你。",
            )
        return ReflexDecision(type="noop")

    if enemy_count <= 0:
        return ReflexDecision(type="noop")

    player_hp_pct = player_hp / max(player_max, 1.0)
    if player_hp_pct <= 0.25 and ally_stance == "assault":
        return ReflexDecision(
            type="command",
            stance="guard",
            reply="你都快倒了，我回来贴着你。",
        )

    return None