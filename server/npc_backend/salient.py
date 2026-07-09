from __future__ import annotations

from typing import Any


def _pct_label(pct: float | None) -> str:
    if pct is None:
        return "未知"
    try:
        p = float(pct)
    except (TypeError, ValueError):
        return "未知"
    if p <= 0.15:
        return "濒死"
    if p <= 0.30:
        return "危急"
    if p <= 0.55:
        return "半血"
    if p <= 0.80:
        return "还行"
    return "健康"


def _stance_label(stance: str) -> str:
    return {"guard": "守护", "assault": "突击"}.get(stance, stance or "未知")


def _boolish(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.lower() in {"1", "true", "yes", "y"}
    return bool(v)


def build_salient_snapshot(
    scene_info: dict[str, Any],
    *,
    last_npc_line: str = "",
    player_message: str = "",
    mode: str = "chat",
    max_points: int = 6,
) -> str:
    """规则生成「此刻重点」自然语言摘要，供对话/决策 Prompt 使用。"""
    points: list[tuple[int, str]] = []

    player_pct = scene_info.get("player_hp_pct")
    ally_pct = scene_info.get("ally_hp_pct")
    try:
        p_pct = float(player_pct) if player_pct is not None else None
    except (TypeError, ValueError):
        p_pct = None
    try:
        a_pct = float(ally_pct) if ally_pct is not None else None
    except (TypeError, ValueError):
        a_pct = None

    if _boolish(scene_info.get("ally_down")) or (a_pct is not None and a_pct <= 0):
        points.append((100, "我灵核失稳，暂时没法出力"))
    elif a_pct is not None and a_pct <= 0.25:
        points.append((95, f"我自己血量{_pct_label(a_pct)}"))

    if p_pct is not None and p_pct <= 0.30:
        points.append((98, f"你血量{_pct_label(p_pct)}，优先保命"))
    elif _boolish(scene_info.get("player_under_fire")):
        points.append((88, "你刚挨打"))
    elif p_pct is not None and p_pct <= 0.55:
        points.append((70, f"你血量{_pct_label(p_pct)}"))

    if _boolish(scene_info.get("hazard_near_player")):
        points.append((92, "你踩在危险圈附近，该躲"))
    if _boolish(scene_info.get("hazard_near_ally")):
        points.append((75, "我也挨着危险圈"))

    bullets_p = int(scene_info.get("incoming_bullets_player", 0) or 0)
    bullets_a = int(scene_info.get("incoming_bullets_ally", 0) or 0)
    if bullets_p >= 2:
        points.append((90, f"有弹幕朝你飞（约{bullets_p}）"))
    if bullets_a >= 2:
        points.append((72, f"有弹幕朝我飞（约{bullets_a}）"))

    if _boolish(scene_info.get("player_silenced")):
        points.append((85, "你被缄言，别催你开火"))
    if _boolish(scene_info.get("player_pulled")):
        points.append((80, "你被磁引扯住"))

    enemy_count = int(scene_info.get("enemy_count", 0) or 0)
    boss_alive = _boolish(scene_info.get("boss_alive"))
    boss_name = str(scene_info.get("boss_name") or "").strip()
    elite_name = str(scene_info.get("elite_name") or "").strip()
    bd = scene_info.get("enemy_breakdown") or {}

    if boss_alive or (isinstance(bd, dict) and int(bd.get("boss", 0) or 0) > 0):
        name = boss_name or "Boss"
        points.append((93, f"Boss「{name}」在场"))
    elif _boolish(scene_info.get("elite_present")) or (
        isinstance(bd, dict) and int(bd.get("elite", 0) or 0) > 0
    ):
        points.append((82, f"精英「{elite_name or '未知'}」在场，敌约{enemy_count}"))
    elif enemy_count > 0:
        points.append((65, f"交战中，敌约{enemy_count}"))
    elif _boolish(scene_info.get("room_cleared")) or _boolish(scene_info.get("door_open")):
        door = "门已开" if _boolish(scene_info.get("door_open")) else "房已清"
        points.append((60, f"暂时安全，{door}"))
    else:
        points.append((40, "周围暂无明确交火"))

    stance = str(scene_info.get("ally_stance") or "guard")
    near = _boolish(scene_info.get("ally_near_player") or scene_info.get("ally_already_guarding"))
    dist = scene_info.get("ally_player_distance")
    if stance == "guard" and near:
        points.append((78, "我在守护，贴着你"))
    elif stance == "guard":
        points.append((68, f"我在守护，但离你还有一段（约{dist}）" if dist is not None else "我在守护"))
    elif stance == "assault":
        phase = str(scene_info.get("ally_combat_phase") or "")
        phase_hint = f"，阶段{phase}" if phase else ""
        points.append((68, f"我在突击前线{phase_hint}"))

    if _boolish(scene_info.get("los_blocked")):
        points.append((74, "视线被挡，不好直接对射"))
    if _boolish(scene_info.get("ally_nav_stuck")):
        points.append((73, "我寻路受阻，绕不开"))

    floor = scene_info.get("floor", "?")
    floor_name = str(scene_info.get("floor_name") or "").strip()
    room_label = str(scene_info.get("room_label") or "").strip()
    room_type = str(scene_info.get("room_type") or "").strip()
    progress = str(scene_info.get("dungeon_progress") or "").strip()
    place_bits = [f"第{floor}层"]
    if floor_name:
        place_bits.append(floor_name)
    if room_label:
        place_bits.append(room_label)
    if room_type:
        place_bits.append(f"({room_type})")
    if progress:
        place_bits.append(f"进度{progress}")
    points.append((35, " ".join(place_bits)))

    floor_state = str(scene_info.get("floor_state") or "")
    if floor_state == "blessing_pick" or scene_info.get("mode") == "blessing_pick":
        bp = scene_info.get("blessing_just_picked") or {}
        if isinstance(bp, dict) and bp.get("name"):
            points.append((86, f"你刚选了狱印「{bp.get('name')}」"))
        else:
            points.append((66, "正在选狱印"))
    elif floor_state == "clear":
        points.append((62, "本层肃清，可喘口气"))

    events = scene_info.get("recent_events") or []
    if events:
        ev = "、".join(str(e) for e in events[:4])
        points.append((55, f"最近事件：{ev}"))

    mood = str(scene_info.get("combat_mood") or "").strip()
    if mood:
        points.append((30, f"我当前情绪偏向：{mood}"))

    if last_npc_line:
        clipped = last_npc_line.strip()
        if len(clipped) > 40:
            clipped = clipped[:40] + "…"
        points.append((84, f"我刚说过：「{clipped}」"))

    if player_message and mode == "chat":
        msg = player_message.strip()
        if len(msg) > 36:
            msg = msg[:36] + "…"
        points.append((50, f"你在跟我说：{msg}"))

    if _boolish(scene_info.get("player_is_idle")) and mode != "chat":
        points.append((77, "你有点发呆，可能需要提醒"))

    # 协同房
    coop_mode = str(scene_info.get("coop_mode") or "").strip()
    if coop_mode and coop_mode != "none":
        tag = str(scene_info.get("coop_room_tag") or coop_mode)
        points.append((96, f"协同房：{tag}"))
        if _boolish(scene_info.get("split_active")):
            pp = scene_info.get("split_player_progress", 0)
            ap = scene_info.get("split_ally_progress", 0)
            points.append((97, f"裂狱进度 你{int(float(pp or 0)*100)}% / 我{int(float(ap or 0)*100)}%"))
            if _boolish(scene_info.get("split_player_done")) and not _boolish(
                scene_info.get("split_ally_done")
            ):
                points.append((94, "你侧已清完，我还在打"))
            if _boolish(scene_info.get("split_ally_done")) and not _boolish(
                scene_info.get("split_player_done")
            ):
                points.append((98, "我左边已肃清，正在等你清右边；可嘲讽催促一句"))
            if _boolish(scene_info.get("split_ally_done")) and _boolish(
                scene_info.get("split_player_done")
            ):
                points.append((98, "裂狱两边都清完，催玩家走右边门"))
        if _boolish(scene_info.get("info_active")) and not _boolish(
            scene_info.get("info_solved")
        ):
            points.append((96, "判词分卷未解，顺序只有我清楚"))
            rep = str(scene_info.get("info_report") or "").strip()
            if rep:
                points.append((91, f"正确顺序：{rep}"))
        if _boolish(scene_info.get("proxy_active")):
            points.append((95, "代行房：你缄言，我靠你指挥输出"))
        slots = scene_info.get("command_slots_left")
        if slots is not None:
            points.append((58, f"口令剩余{slots}/{scene_info.get('command_slots_max', '?')}"))
        if _boolish(scene_info.get("combo_window")):
            points.append((80, "契印连携窗口已开"))

    points.sort(key=lambda x: x[0], reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for _, text in points:
        if text in seen:
            continue
        seen.add(text)
        selected.append(text)
        if len(selected) >= max_points:
            break

    if not selected:
        return "场面平稳，无特别紧急事项。"
    return "\n".join(f"- {line}" for line in selected)


def build_memory_search_query(
    *,
    player_message: str = "",
    scene_info: dict[str, Any] | None = None,
    npc_name: str = "乌枭",
    trigger: str = "",
) -> str:
    """精炼向量检索 query，避免塞入整包 scene_info。"""
    scene_info = scene_info or {}
    bits: list[str] = []
    msg = (player_message or "").strip()
    if msg:
        bits.append(msg)

    keywords: list[str] = []
    if _boolish(scene_info.get("boss_alive")) or scene_info.get("boss_name"):
        keywords.append(f"Boss {scene_info.get('boss_name', '')}".strip())
    if _boolish(scene_info.get("elite_present")) or scene_info.get("elite_name"):
        keywords.append(f"精英 {scene_info.get('elite_name', '')}".strip())
    if _boolish(scene_info.get("player_in_danger")) or _boolish(scene_info.get("ally_in_danger")):
        keywords.append("危急 救援")
    if _boolish(scene_info.get("hazard_near_player")):
        keywords.append("危险圈")
    if scene_info.get("blessing_just_picked"):
        bp = scene_info.get("blessing_just_picked") or {}
        if isinstance(bp, dict) and bp.get("name"):
            keywords.append(f"狱印 {bp.get('name')}")
    floor_name = str(scene_info.get("floor_name") or "").strip()
    if floor_name:
        keywords.append(floor_name)
    stance = str(scene_info.get("ally_stance") or "")
    if stance:
        keywords.append(_stance_label(stance))
    if trigger:
        keywords.append(trigger)

    if keywords:
        bits.append(" ".join(keywords[:6]))
    bits.append(f"{npc_name} 战友对话")
    return "\n".join(b for b in bits if b)


def infer_chat_mode(
    message: str,
    scene_info: dict[str, Any] | None = None,
) -> str:
    """轻量分流：影响 Prompt 语气与检索权重。"""
    scene_info = scene_info or {}
    text = (message or "").strip()
    enemy_count = int(scene_info.get("enemy_count", 0) or 0)
    in_fight = enemy_count > 0 or _boolish(scene_info.get("boss_alive"))
    danger = (
        _boolish(scene_info.get("player_in_danger"))
        or _boolish(scene_info.get("ally_in_danger"))
        or _boolish(scene_info.get("hazard_near_player"))
    )

    lore_keys = ("为什么", "什么是", "阴司", "地狱", "狱", "鬼差", "判词", "阎", "传说", "设定")
    emotion_keys = ("谢谢", "抱歉", "对不起", "害怕", "怕", "加油", "靠谱", "骂", "滚")
    tactics_keys = ("怎么打", "怎么办", "策略", "注意", "要不要", "该不该", "先打谁")

    if any(k in text for k in emotion_keys):
        return "emotional"
    if any(k in text for k in lore_keys):
        return "meta_lore"
    if any(k in text for k in tactics_keys):
        return "combat_question"
    if in_fight or danger:
        if len(text) <= 12:
            return "combat_ack"
        return "combat_question" if ("?" in text or "？" in text or "吗" in text) else "combat_ack"
    return "rest_banter"


def command_reply_from_scene(
    stance: str,
    scene_info: dict[str, Any] | None = None,
    *,
    same_stance: bool = False,
) -> str:
    """战术指令确认语：模板槽位 + 场面，0-LLM。"""
    scene_info = scene_info or {}
    player_pct = scene_info.get("player_hp_pct")
    try:
        p_pct = float(player_pct) if player_pct is not None else None
    except (TypeError, ValueError):
        p_pct = None
    near = _boolish(scene_info.get("ally_near_player") or scene_info.get("ally_already_guarding"))
    enemy_count = int(scene_info.get("enemy_count", 0) or 0)
    boss = _boolish(scene_info.get("boss_alive"))

    if stance == "guard":
        if same_stance and near:
            if p_pct is not None and p_pct <= 0.30:
                return "还护着呢，你血见底了，往后半步，我挡。"
            return "本来就贴着你，别瞎指挥。盯前边。"
        if p_pct is not None and p_pct <= 0.30:
            return "收了，我贴着你。血见底就别硬刚。"
        if enemy_count >= 4 or boss:
            return "行，收拢护着你。这波乱，别脱节。"
        return "行，收拢了，你别乱跑，我贴着你。"

    if stance == "assault":
        if same_stance:
            if boss:
                return "我已经在前头撕了，你盯 Boss 窗口，别送。"
            return "还在突呢，别催，你跟好输出位。"
        if p_pct is not None and p_pct <= 0.30:
            return "我去压一波，你血薄，别跟太前。"
        if boss:
            return "好嘞，我顶上去撕，你抓空隙打。"
        if enemy_count >= 5:
            return "怪多，我先撕开路，你别站中间。"
        return "好嘞，我去前面撕，你别拖后腿。"

    return "收到。"
