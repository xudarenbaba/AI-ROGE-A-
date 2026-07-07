from __future__ import annotations

from typing import Any


def _join_lines(lines: list[str]) -> str:
    if not lines:
        return "无"
    return "\n".join(f"- {line}" for line in lines)


def _events_block(scene_info: dict[str, Any]) -> str:
    events = scene_info.get("recent_events") or []
    if not events:
        return "无"
    return "、".join(str(e) for e in events)


def _tactical_block(scene_info: dict[str, Any]) -> str:
    keys = (
        ("nearest_enemy_distance", "最近敌人距离"),
        ("ally_nearest_enemy_distance", "乌枭最近敌人距离"),
        ("player_has_los", "玩家视线"),
        ("ally_has_los", "乌枭视线"),
        ("los_blocked", "视线被挡"),
        ("incoming_bullets_player", "飞向玩家弹幕"),
        ("incoming_bullets_ally", "飞向乌枭弹幕"),
        ("enemies_in_player_range", "玩家射程内敌数"),
        ("enemies_in_ally_range", "乌枭射程内敌数"),
        ("ally_nav_stuck", "乌枭寻路受阻"),
        ("player_under_fire", "玩家刚挨打"),
        ("ally_already_guarding", "已在守护玩家"),
        ("ally_guard_duration_s", "守护姿态持续秒"),
        ("player_is_active", "玩家近期活跃"),
        ("assault_recommended", "建议切突击"),
    )
    parts = [
        f"{label}={scene_info.get(key)}"
        for key, label in keys
        if key in scene_info
    ]
    return "；".join(parts) if parts else "无"


def _coherence_for_trigger(trigger: str, scene_info: dict[str, Any] | None = None) -> str:
    scene_info = scene_info or {}
    assault_hint = ""
    if scene_info.get("assault_recommended"):
        assault_hint = (
            "6. assault_recommended=true 且 ally_stance=guard 时，"
            "优先 command→assault，不要只吐槽不动。\n"
        )
    if trigger in ("periodic", "social"):
        return (
            "【连贯性与去重约束】\n"
            "1. 不要重复[最近自主发言]中已出现过的措辞。\n"
            "2. guard→assault 在 assault_recommended=true 时不算「轻易反转」。\n"
            "3. periodic/social 且 since_last_npc_speech≥45s 时可 dialogue 碎嘴；"
            "但若 assault_recommended=true 应优先 assault。\n"
            "4. 可利用[战术态势]给实用提醒。\n"
            "5. 一句话只表达一个意图。\n"
            + assault_hint
        )
    if trigger == "scene_change":
        return (
            "【连贯性与去重约束】\n"
            "1. 不要重复[最近自主发言]中已出现过的措辞。\n"
            "2. 换层/开战时若 assault_recommended=true，优先 command→assault。\n"
            "3. 一句话只表达一个意图。\n"
            + assault_hint
        )
    return _COHERENCE_RULES


def build_messages(
    *,
    npc_name: str,
    player_message: str,
    scene_info: dict[str, Any],
    world_chunks: list[str],
    persona_chunks: list[str],
    dialogue_daily_chunks: list[str],
    dialogue_important_chunks: list[str],
    short_term_history: list[dict[str, str]],
    narrative_context: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    short_term_lines = [
        f"{item.get('role', 'unknown')}: {item.get('content', '')}"
        for item in short_term_history
        if item.get("content")
    ]
    system_prompt = (
        f"你是 NPC“{npc_name}”，现在进行单 NPC 对话演示。\n"
        "要求：\n"
        "1) 回复使用中文，简洁自然。\n"
        "2) 战斗场景优先给出可执行建议，通常控制在 1-3 句。\n"
        "3) 非战斗场景可适度话痨，允许 3-6 句碎嘴吐槽。\n"
        "4) 严格基于记忆，不要编造长期事实。\n"
        "5) 注意保持与最近对话的一致性；承接你刚才自主说过的话，不要装作没说过。\n"
        "5b) 不要重复[最近自主发言]里已说过的句子或战术宣告。\n"
        "5c) 若你刚在战斗中自主求援或催促玩家，玩家回应时要承接该语境（如「来了」「收到」）。\n"
        "6) 可以毒舌和贱嗖嗖，但不能恶意辱骂或人身攻击。\n"
        "7) 不输出工具调用、不输出 JSON。\n"
        "8) 在回复正文末尾另起一行，输出一个情绪标签，格式严格为 <emotion>单词</emotion>，"
        "从以下词中选一个最符合当前语气的：neutral focused annoyed worried happy tense sarcastic。"
        "不要解释标签，不要省略。"
    )
    narrative = narrative_context or {}
    user_prompt = (
        f"[场景]\n{scene_info}\n\n"
        f"[最近战斗事件]\n{_events_block(scene_info)}\n\n"
        f"[战术态势]\n{_tactical_block(scene_info)}\n\n"
        f"[战斗情绪]\n{narrative.get('mood_block', '无')}\n\n"
        f"[当前战术意图]\n{narrative.get('intent_block', '无')}\n\n"
        f"[最近自主发言]\n{narrative.get('recent_autonomous', '无')}\n\n"
        f"[世界观设定]\n{_join_lines(world_chunks)}\n\n"
        f"[角色设定]\n{_join_lines(persona_chunks)}\n\n"
        f"[对话记忆-重要]\n{_join_lines(dialogue_important_chunks)}\n\n"
        f"[对话记忆-日常]\n{_join_lines(dialogue_daily_chunks)}\n\n"
        f"[短期记忆-最近轮次]\n{_join_lines(short_term_lines)}\n\n"
        f"[玩家输入]\n{player_message}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


_COHERENCE_RULES = (
    "【连贯性与去重约束】\n"
    "1. 不要重复[最近自主发言]中已出现过的措辞、句式或战术宣告。\n"
    "2. 若战术意图仍在有效期内，禁止输出相反姿态的 command。\n"
    "3. 刚切过姿态或刚说过话，优先 noop；无新信息不说。\n"
    "4. 一句话只表达一个意图，不要混搭吐槽+切姿态+战术教学。\n"
    "5. 前后立场一致；不能上一句守着，下一句毫无理由要冲。\n"
    "6. 局面无剧变时（见 scene_dramatic_change），倾向 noop。\n"
)

_AUTONOMOUS_DECIDE_RULES = (
    "你是战斗中的 NPC 自主决策器，只输出 JSON，格式三选一：\n"
    '{"type":"noop"}\n'
    '{"type":"command","stance":"guard|assault","reply":"NPC一句确认话"}\n'
    '{"type":"dialogue","reply":"NPC主动说的一句话"}\n\n'
    "判断规则：\n"
    "1. noop：局面平稳、刚说过话、无显著变化、没什么好说的、或会重复旧话时。\n"
    "2. command：玩家危急且正在突击→guard；"
    "assault_recommended=true 时→assault（尤其换层/玩家在打/敌数适中）。\n"
    "3. dialogue：有战术提醒但暂不切姿态时用。\n"
    "4. ally_hp≤0 时禁止 assault。\n"
    "4b. assault_recommended=true 时不要 noop，应 command→assault。\n"
    "4c. guard 贴身时 dialogue 禁止喊「过来」；但 assault 不受此限。\n"
    "5. reply 符合嘴臭话痨风格，简短，1句话，且必须新颖不重复。\n"
    "6. dialogue 回复末尾不要加 emotion 标签。\n"
    "7. 当前姿态与目标 stance 相同时，必须输出 noop，不要 command。\n"
)


def build_autonomous_decide_messages(
    *,
    npc_name: str,
    scene_info: dict[str, Any],
    world_chunks: list[str],
    persona_chunks: list[str],
    dialogue_daily_chunks: list[str],
    dialogue_important_chunks: list[str],
    short_term_history: list[dict[str, str]],
    narrative_context: dict[str, str] | None = None,
    allowed_intents: list[str] | None = None,
    trigger: str = "periodic",
) -> list[dict[str, str]]:
    short_term_lines = [
        f"{item.get('role', 'unknown')}: {item.get('content', '')}"
        for item in short_term_history
        if item.get("content")
    ]
    system_prompt = (
        f"你是 NPC「{npc_name}」的自主决策模块。\n"
        "玩家当前没有说话。你需要根据局面判断：保持沉默(noop)、"
        "切换战斗姿态(command)、或主动说一句话(dialogue)。\n"
        "只输出 JSON，不要输出其他文字。"
    )
    narrative = narrative_context or {}
    intent_hint = "、".join(allowed_intents) if allowed_intents else "noop、command、dialogue"
    user_prompt = (
        f"[场景]\n{scene_info}\n\n"
        f"[最近战斗事件]\n{_events_block(scene_info)}\n\n"
        f"[战术态势]\n{_tactical_block(scene_info)}\n\n"
        f"[战斗情绪]\n{narrative.get('mood_block', '无')}\n\n"
        f"[当前战术意图]\n{narrative.get('intent_block', '无')}\n\n"
        f"[最近自主发言]（禁止重复以下措辞）\n{narrative.get('recent_autonomous', '无')}\n\n"
        f"[本轮建议intent]\n{intent_hint}\n\n"
        f"[触发类型]\n{trigger}\n\n"
        f"[世界观设定]\n{_join_lines(world_chunks)}\n\n"
        f"[角色设定]\n{_join_lines(persona_chunks)}\n\n"
        f"[对话记忆-重要]\n{_join_lines(dialogue_important_chunks)}\n\n"
        f"[对话记忆-日常]\n{_join_lines(dialogue_daily_chunks)}\n\n"
        f"[短期记忆-最近轮次]\n{_join_lines(short_term_lines)}\n\n"
        f"[触发]\n自主思考：玩家暂时无输入，trigger={trigger}，请评估局面并决策。"
    )
    return [
        {
            "role": "system",
            "content": _AUTONOMOUS_DECIDE_RULES + "\n" + _coherence_for_trigger(trigger, scene_info) + "\n" + system_prompt,
        },
        {"role": "user", "content": user_prompt},
    ]


def build_memory_classify_messages(
    *,
    player_message: str,
    npc_reply: str,
    scene_info: dict[str, Any],
) -> list[dict[str, str]]:
    system_prompt = (
        "你是对话记忆分级器，只输出 JSON："
        '{"dialogue_tier":"daily|important","processed_text":"..."}。\n'
        "规则：important 保留原文；daily 压缩成 1-2 句摘要。"
    )
    user_prompt = (
        f"scene_info={scene_info}\n"
        f"player_message={player_message}\n"
        f"npc_reply={npc_reply}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

