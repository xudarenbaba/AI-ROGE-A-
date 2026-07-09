from __future__ import annotations

from typing import Any

from server.npc_backend.postprocess import is_combat_scene
from server.npc_backend.salient import build_salient_snapshot


# 固定人设铁律：不依赖 RAG 是否抽中
_PERSONA_CORE = (
    "你是地狱巡差「乌枭」（黑签），嘴臭、贱、话痨，但极其专业，从不弃队。\n"
    "说话公式：毒舌吐槽 + 事实判断 + 可执行落点（位置/目标/时机）。\n"
    "把玩家当临时搭档：嘴上嫌弃，行动兜底。短句，口语，可阴阳怪气，禁止人身攻击与脏话升级。\n"
    "你在战场上，不是客服、不是旁白、不是系统提示。"
)

_WAR_BUDDY_PROTOCOL = (
    "【战友对话协议】\n"
    "1) 先回应玩家这句话的语义，再带场面；禁止答非所问。\n"
    "2) 只引用[此刻重点]/[本局迄今]里出现的事实，禁止编造未给出的剧情、数值细节。\n"
    "3) 必须承接[短期对话]与[我刚说过]；玩家说「来了/收到/好」时要接你自己的求援或指令语境。\n"
    "4) 不要重复你最近刚说过的句子或战术宣告。\n"
    "5) 战斗/危急：1-2句；安全/走廊：可3-5句碎嘴。\n"
    "6) 禁止念血条报表、禁止清单式分析、禁止「作为你的队友」等AI腔。\n"
    "7) 姿态语义：guard=你贴玩家护；assault=你前压输出。已在贴身守护时，不要喊玩家「过来/跟紧我」。\n"
    "8) 不输出工具调用、不输出JSON；正文末尾另起一行输出 <emotion>标签</emotion>，"
    "从 neutral focused annoyed worried happy tense sarcastic 中选一个。"
)


def _join_lines(lines: list[str]) -> str:
    if not lines:
        return "无"
    return "\n".join(f"- {line}" for line in lines)


def _events_block(scene_info: dict[str, Any]) -> str:
    events = scene_info.get("recent_events") or []
    if not events:
        return "无"
    return "、".join(str(e) for e in events)


def _enemy_breakdown_block(scene_info: dict[str, Any]) -> str:
    bd = scene_info.get("enemy_breakdown")
    if not bd:
        return "无"
    return (
        f"mob={bd.get('mob', 0)} elite={bd.get('elite', 0)} "
        f"boss={bd.get('boss', 0)} shade={bd.get('shade', 0)}"
    )


def _rule_hints_block(scene_info: dict[str, Any]) -> str:
    hints = scene_info.get("rule_hints") or []
    if not hints:
        return "无"
    lines: list[str] = []
    for item in hints:
        if not isinstance(item, dict):
            continue
        intent = str(item.get("intent", "")).strip()
        hint = str(item.get("hint", "")).strip()
        reason = str(item.get("reason", "")).strip()
        if not hint:
            continue
        parts = [f"intent={intent}" if intent else "", f"参考语气：{hint}"]
        if reason:
            parts.append(f"原因={reason}")
        lines.append("；".join(p for p in parts if p))
    return "\n".join(f"- {line}" for line in lines) if lines else "无"


def _blessings_block(scene_info: dict[str, Any]) -> str:
    summary = scene_info.get("blessings_summary") or {}
    tags = summary.get("tags") or scene_info.get("build_tags") or []
    total = scene_info.get("blessings_total", summary.get("total", 0))
    if not total:
        return "无"
    tag_str = "、".join(str(t) for t in tags[:6]) if tags else "无"
    return f"总数={total}；标签={tag_str}"


def scene_summary_block(scene_info: dict[str, Any]) -> str:
    parts = [
        f"第{scene_info.get('floor', '?')}层 {scene_info.get('floor_name', '')}",
        f"房间={scene_info.get('room_label', '')}({scene_info.get('room_type', '')})",
        f"进度={scene_info.get('dungeon_progress', '')}",
        f"状态={scene_info.get('floor_state', '')}",
        f"门={'已开' if scene_info.get('door_open') else '未开'}",
        f"姿态={scene_info.get('ally_stance', '')}",
        f"阶段={scene_info.get('ally_combat_phase', '')}",
        f"玩家HP={scene_info.get('player_hp', '?')}/{scene_info.get('player_max_hp', '?')}",
        f"乌枭HP={scene_info.get('ally_hp', '?')}/{scene_info.get('ally_max_hp', '?')}",
        f"敌数={scene_info.get('enemy_count', 0)}",
        f"Boss={'存活' if scene_info.get('boss_alive') else '无'}",
    ]
    if scene_info.get("boss_name"):
        parts.append(f"Boss名={scene_info.get('boss_name')}")
    if scene_info.get("elite_present"):
        parts.append(f"精英={scene_info.get('elite_name', '')}")
    return "；".join(parts)


def player_state_block(scene_info: dict[str, Any]) -> str:
    keys = (
        ("player_last_action_s", "距上次操作秒"),
        ("player_is_active", "玩家活跃"),
        ("player_is_idle", "玩家发呆"),
        ("idle_seconds", "玩家聊天空闲秒"),
        ("since_last_npc_speech", "距上次NPC发言秒"),
        ("player_silenced", "玩家缄言"),
        ("player_pulled", "玩家被磁引"),
        ("screen_fog", "屏幕迷雾"),
        ("player_in_danger", "玩家危急"),
        ("ally_in_danger", "乌枭危急"),
    )
    parts = [f"{label}={scene_info.get(key)}" for key, label in keys if key in scene_info]
    return "；".join(parts) if parts else "无"


def tactical_block(scene_info: dict[str, Any]) -> str:
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
        ("ally_near_player", "乌枭贴身"),
        ("ally_guard_duration_s", "守护持续秒"),
        ("assault_recommended", "建议切突击"),
        ("scene_dramatic_change", "局面剧变"),
        ("hazard_count", "地面圈数"),
        ("hazard_near_player", "玩家在圈内"),
        ("hazard_near_ally", "乌枭在圈内"),
        ("combat_threat_mul", "敌强倍率"),
        ("can_autonomy_speak", "可自主发言"),
        ("trigger", "触发类型"),
        ("trigger_reason", "触发原因"),
    )
    parts = [f"{label}={scene_info.get(key)}" for key, label in keys if key in scene_info]
    return "；".join(parts) if parts else "无"


def build_scene_context_blocks(
    scene_info: dict[str, Any],
    narrative_context: dict[str, str] | None = None,
    *,
    include_player_state: bool = True,
    use_salient: bool = True,
    last_npc_line: str = "",
    player_message: str = "",
    mode: str = "think",
) -> str:
    narrative = narrative_context or {}
    blocks: list[str] = []

    if use_salient:
        snap = scene_info.get("salient_snapshot") or build_salient_snapshot(
            scene_info,
            last_npc_line=last_npc_line or narrative.get("last_autonomous_line", ""),
            player_message=player_message,
            mode=mode,
        )
        blocks.append(f"[此刻重点]\n{snap}")
        journal = narrative.get("journal_block", "无")
        blocks.append(f"[本局迄今]\n{journal}")
        relation = narrative.get("relation_block", "")
        if relation:
            blocks.append(f"[搭档关系]\n{relation}")
        last_auto = narrative.get("last_autonomous_line", "无")
        blocks.append(f"[我刚说过]\n{last_auto}")
        # 精简后备，避免全量 key 堆砌
        blocks.append(f"[场景摘要]\n{scene_summary_block(scene_info)}")
        blocks.append(f"[战斗情绪]\n{narrative.get('mood_block', '无')}")
        blocks.append(f"[当前战术意图]\n{narrative.get('intent_block', '无')}")
        return "\n\n".join(blocks)

    blocks = [
        f"[场景摘要]\n{scene_summary_block(scene_info)}",
        f"[敌人构成]\n{_enemy_breakdown_block(scene_info)}",
        f"[狱印摘要]\n{_blessings_block(scene_info)}",
        f"[最近战斗事件]\n{_events_block(scene_info)}",
        f"[战术态势]\n{tactical_block(scene_info)}",
    ]
    if include_player_state:
        blocks.append(f"[玩家状态]\n{player_state_block(scene_info)}")
    blocks.extend([
        f"[战斗情绪]\n{narrative.get('mood_block', '无')}",
        f"[当前战术意图]\n{narrative.get('intent_block', '无')}",
        f"[最近自主发言]\n{narrative.get('recent_autonomous', '无')}",
    ])
    return "\n\n".join(blocks)


def build_intent_classify_prompt(
    *,
    npc_name: str,
    message: str,
    scene_info: dict[str, Any],
) -> str:
    snap = scene_info.get("salient_snapshot") or build_salient_snapshot(
        scene_info, player_message=message, mode="chat", max_points=4
    )
    return (
        f"npc_name={npc_name}\n"
        f"[此刻重点]\n{snap}\n"
        f"ally_stance={scene_info.get('ally_stance', '')}\n"
        f"assault_recommended={scene_info.get('assault_recommended', '')}\n"
        f"player_message={message}"
    )


def _tactical_block(scene_info: dict[str, Any]) -> str:
    return tactical_block(scene_info)


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
            "1. 不要重复[最近自主发言]/[我刚说过]中已出现过的措辞。\n"
            "2. guard→assault 在 assault_recommended=true 时不算「轻易反转」。\n"
            "3. periodic/social 且 since_last_npc_speech≥38s 时可 dialogue 碎嘴；"
            "但若 assault_recommended=true 应优先 assault。\n"
            "4. 可利用[此刻重点]给实用提醒。\n"
            "5. 一句话只表达一个意图。\n"
            + assault_hint
        )
    if trigger == "scene_change":
        return (
            "【连贯性与去重约束】\n"
            "1. 不要重复已说过的措辞。\n"
            "2. 换层/开战时若 assault_recommended=true，优先 command→assault。\n"
            "3. 一句话只表达一个意图。\n"
            + assault_hint
        )
    return _COHERENCE_RULES


def _chat_mode_hint(chat_mode: str) -> str:
    hints = {
        "combat_ack": "模式=战斗短应：1句承接+可选半句战术，别展开。",
        "combat_question": "模式=战术问答：先答问题，再给一句可执行建议。",
        "rest_banter": "模式=休整碎嘴：可话痨3-5句，仍要像乌枭。",
        "meta_lore": "模式=设定问答：答设定，少念血条；一句点明当前是否安全。",
        "emotional": "模式=情绪承接：先接情绪，毒舌收着点，别升级嘲讽。",
    }
    return hints.get(chat_mode, "模式=自然对话：像战场搭档接话。")


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
    chat_thread: list[dict[str, str]] | None = None,
    chat_mode: str = "rest_banter",
    last_npc_line: str = "",
) -> list[dict[str, str]]:
    """玩家对话：system + 上下文 user + 真多轮 history + 当前 user。"""
    narrative = narrative_context or {}
    combat = is_combat_scene(scene_info)
    length_rule = (
        "当前偏战斗/危急：最多2句，短、狠、有用。"
        if combat or chat_mode in ("combat_ack", "combat_question")
        else "当前偏安全：可3-5句碎嘴，仍口语。"
    )
    system_prompt = (
        f"{_PERSONA_CORE}\n"
        f"你当前扮演「{npc_name}」。\n"
        f"{_WAR_BUDDY_PROTOCOL}\n"
        f"{_chat_mode_hint(chat_mode)}\n"
        f"{length_rule}"
    )

    context = build_scene_context_blocks(
        scene_info,
        narrative,
        use_salient=True,
        last_npc_line=last_npc_line or narrative.get("last_autonomous_line", ""),
        player_message=player_message,
        mode="chat",
    )

    # 人设：固定核心已在 system；RAG 仅作补充，限量
    persona_extra = _join_lines((persona_chunks or [])[:2])
    world_extra = _join_lines((world_chunks or [])[:2])
    mem_imp = _join_lines((dialogue_important_chunks or [])[:4])
    mem_daily = _join_lines((dialogue_daily_chunks or [])[:3])

    context_user = (
        f"{context}\n\n"
        f"[角色补充]\n{persona_extra}\n\n"
        f"[世界补充]\n{world_extra}\n\n"
        f"[对话记忆-重要]\n{mem_imp}\n\n"
        f"[对话记忆-日常]\n{mem_daily}\n\n"
        f"[最近自主发言日志]\n{narrative.get('recent_autonomous', '无')}\n\n"
        "以上是你的处境与记忆。下面是你与玩家的近期对话；请以乌枭身份自然接话。"
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context_user},
        {
            "role": "assistant",
            "content": "知道了。场面我盯着，你说话我就接——别废话太多。",
        },
    ]

    # 真多轮对话线程
    thread = chat_thread
    if thread is None:
        # 兼容旧 short_term：过滤 system，映射 role
        thread = []
        for item in short_term_history or []:
            role = str(item.get("role", ""))
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            if role == "system":
                continue
            if role not in {"user", "assistant"}:
                role = "user" if role == "user" else "assistant"
            thread.append({"role": role, "content": content})

    # 最近 8 轮（16 条）
    for item in thread[-16:]:
        role = item.get("role", "user")
        content = str(item.get("content", "")).strip()
        if not content or role not in {"user", "assistant"}:
            continue
        messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": player_message})
    return messages


_COHERENCE_RULES = (
    "【连贯性与去重约束】\n"
    "1. 不要重复[最近自主发言]/[我刚说过]中已出现过的措辞、句式或战术宣告。\n"
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
    "4d. can_autonomy_speak=false（选印/过门）时只输出 noop。\n"
    "4e. player_silenced=true 时不要催玩家射击。\n"
    "4f. hazard_near_player=true 时提醒躲圈，优先 dialogue。\n"
    "4c. guard 且 ally_near_player/ally_already_guarding=true 时，"
    "dialogue 禁止喊「跟紧我/过来/靠近/别愣着」让玩家过来；可说「我贴着你/别冲太前」。"
    "assault 不受此限。\n"
    "4g. 参考[姿态语义]与[规则参考]组织台词，勿颠倒跟随关系。\n"
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
    chat_thread: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    narrative = narrative_context or {}
    short_term_lines = [
        f"{item.get('role', 'unknown')}: {item.get('content', '')}"
        for item in (chat_thread or short_term_history or [])
        if item.get("content")
    ][-12:]
    system_prompt = (
        f"{_PERSONA_CORE}\n"
        f"你是「{npc_name}」的自主决策模块。\n"
        "玩家当前没有说话。根据局面判断：保持沉默(noop)、"
        "切换战斗姿态(command)、或主动说一句话(dialogue)。\n"
        "只输出 JSON，不要输出其他文字。dialogue/command 的 reply 要像战友随口一句。"
    )
    intent_hint = "、".join(allowed_intents) if allowed_intents else "noop、command、dialogue"
    context = build_scene_context_blocks(
        scene_info,
        narrative,
        use_salient=True,
        last_npc_line=narrative.get("last_autonomous_line", ""),
        mode="think",
    )
    stance_sem = str(scene_info.get("stance_semantics", "")).strip() or "无"
    user_prompt = (
        f"{context}\n\n"
        f"[姿态语义]\n{stance_sem}\n\n"
        f"[规则参考]\n{_rule_hints_block(scene_info)}\n\n"
        f"[本轮建议intent]\n{intent_hint}\n\n"
        f"[触发类型]\n{trigger}\n\n"
        f"[角色补充]\n{_join_lines((persona_chunks or [])[:2])}\n\n"
        f"[对话记忆-重要]\n{_join_lines((dialogue_important_chunks or [])[:3])}\n\n"
        f"[近期对话]\n{_join_lines(short_term_lines) if short_term_lines else '无'}\n\n"
        f"[最近自主发言]\n{narrative.get('recent_autonomous', '无')}\n\n"
        f"[触发]\n自主思考：玩家暂时无输入，trigger={trigger}，请评估局面并决策。"
    )
    return [
        {
            "role": "system",
            "content": _AUTONOMOUS_DECIDE_RULES
            + "\n"
            + _coherence_for_trigger(trigger, scene_info)
            + "\n"
            + system_prompt,
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
        "玩家与乌枭的约定、情绪转折、关键战术承诺 → important；闲聊碎嘴 → daily。"
    )
    user_prompt = (
        f"scene_summary={scene_summary_block(scene_info)}\n"
        f"player_message={player_message}\n"
        f"npc_reply={npc_reply}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
