from __future__ import annotations

from typing import Any

from server.npc_backend.llm import chat_completion
from server.npc_backend.narrative import is_duplicate_reply


def polish_line(
    *,
    template: str,
    npc_name: str,
    scene_info: dict[str, Any],
    recent_texts: list[str],
    intent: str,
) -> str | None:
    """异步润色模板句；失败或重复则返回 None，保留原句。"""
    stance = str(scene_info.get("ally_stance", ""))
    near = bool(scene_info.get("ally_near_player") or scene_info.get("ally_already_guarding"))
    dist = scene_info.get("ally_player_distance", "?")
    spatial = (
        f"当前姿态={stance}，与玩家距离={dist}px。"
        + (
            "已在玩家身边守护，台词禁止喊「跟紧我/过来/靠近/别愣着」让玩家过来。"
            if stance == "guard" and near
            else ""
        )
    )
    system = (
        f"你是 NPC「{npc_name}」，嘴臭话痨。"
        f"把下面这句战斗台词润色得更自然，保持原意，1句话，不超过25字。"
        f"intent={intent}。{spatial}不要加引号，不要解释。"
    )
    user = f"原句：{template}\n局面：{scene_info}"
    try:
        polished = chat_completion([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]).strip()
        if not polished or len(polished) > 60:
            return None
        if is_duplicate_reply(polished, recent_texts):
            return None
        return polished
    except Exception:
        return None