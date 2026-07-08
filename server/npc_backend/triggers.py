from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from server.npc_backend.narrative import NarrativeState


@dataclass(frozen=True)
class TriggerSpeech:
    type: str  # dialogue | command
    intent: str
    reply: str
    stance: str | None = None
    reason: str = ""
    allow_polish: bool = True
    skip_dedup: bool = False
    mode: str = "hard"  # hard | hint | fallback


_TEMPLATES: dict[str, list[str]] = {
    "call_for_help": [
        "快帮我挡一下，我快撑不住了！",
        "别愣着，过来帮我分担一下火力！",
        "我血量见底了，你再不动我就先撤了！",
    ],
    "call_for_help_guarding": [
        "我已经贴着你了，帮我把最近的杂碎清了！",
        "我快扛不住了，你也集火一下，别让我白顶。",
        "就在你身边守着呢，别乱跑，先把火力卸掉！",
    ],
    "call_for_help_dire": [
        "我要退了，你再不来我就真躺了！",
        "快没命了，你倒是过来啊！",
    ],
    "call_for_help_guarding_dire": [
        "我就剩一口气了，贴着你也得帮我打啊！",
        "再不打我真要退了，别光让我一个人扛！",
    ],
    "warn_player_idle": [
        "你倒是动啊，站着挨打很有意思？",
        "别发呆，快没命了还愣着！",
        "快来帮我，别光看着！",
    ],
    "warn_player_guard": [
        "你都快倒了，我回来贴着你，别送了。",
        "别硬撑了，我先护着你。",
    ],
    "tactical_surge": [
        "敌一下子涌上来了，别散开！",
        "这波怪有点多，稳住别贪。",
    ],
    "tactical_thin": [
        "就剩这几个了，别松懈。",
        "收尾了，集中火力。",
    ],
    "floor_enter_guard": [
        "新一层，我贴着你走，你别冲太前。",
        "往下探了，跟在我这边，别乱跑。",
    ],
    "floor_enter_assault": [
        "新一层，我先探路，你跟上。",
        "往下探了，我先去清场，别掉队。",
    ],
    "boss_spawn": [
        "大家伙来了，别盯着小怪。",
        "Boss 现身了，注意走位。",
    ],
    "boss_enrage": [
        "它进二阶段了，技能会更密！",
        "Boss 暴怒了，别贪刀。",
    ],
    "elite_enrage": [
        "精英厉怒了，技能加快！",
        "这精英半血了，当心连招。",
    ],
    "banter_guard": [
        "还行，我贴着你看，别冲太前。",
        "啧，这局面还行，稳住别乱冲。",
    ],
    "banter_assault": [
        "你别光看着，想好怎么打没？",
        "还行，继续，别掉链子。",
    ],
    "relieved": [
        "行了，这波算稳住了。",
        "还好没翻车，继续。",
    ],
    "enemy_clear": [
        "清干净了，喘口气再走。",
        "这层杂鱼收拾完了，别松懈。",
    ],
    "los_blocked": [
        "墙挡着线，绕一下再打。",
        "障碍物卡视野了，别硬穿。",
    ],
    "dodge_bullets": [
        "弹幕来了，别站桩！",
        "躲开这波弹道，再继续。",
    ],
    "ally_nav_stuck": [
        "路被堵死了，我先绕过去。",
        "绕不过去，换条路线。",
    ],
    "under_fire": [
        "你刚挨了几下，先躲一躲！",
        "别硬顶火力，走位啊！",
    ],
    "stance_assault": [
        "行，我去前面开路，你跟紧。",
        "局面稳了，我切突击上去拉仇恨。",
        "守够久了，我去冲，你别掉队。",
    ],
    "stance_assault_opening": [
        "新一层开了，我先去探路，跟上。",
        "别磨蹭，我上去清场，你跟紧。",
    ],
    "blessing_picked_guard": [
        "印烙好了，我护着你继续往下。",
        "行，这印能用，贴着我走别冲太前。",
    ],
    "blessing_picked_assault": [
        "印烙好了，我先上，你跟上。",
        "行，这印能用，继续往下探。",
    ],
    "room_cleared": [
        "门开了，贴右边走。",
        "清完了，往前门那边靠。",
    ],
    "room_enter_elite": [
        "精英房，留神它的技能。",
        "这间有硬茬，别贪刀。",
    ],
    "hazard_dodge": [
        "圈要炸了，先出圈！",
        "地面落印了，别踩圈里！",
    ],
    "player_silenced": [
        "缄言封着你呢，先走位别硬射。",
        "你嘴被封了，躲圈等我打。",
    ],
}


def _pick(intent: str) -> str:
    pool = _TEMPLATES.get(intent) or _TEMPLATES["banter_guard"]
    return random.choice(pool)


def _ally_stance(scene: dict[str, Any]) -> str:
    return str(scene.get("ally_stance", "guard"))


def _pick_stance(scene: dict[str, Any], guard_key: str, assault_key: str) -> str:
    if _ally_stance(scene) == "assault":
        return _pick(assault_key)
    return _pick(guard_key)


def _as_hint(speech: TriggerSpeech) -> TriggerSpeech:
    if speech.mode == "hint":
        return speech
    return TriggerSpeech(
        type=speech.type,
        intent=speech.intent,
        reply=speech.reply,
        stance=speech.stance,
        reason=speech.reason,
        allow_polish=speech.allow_polish,
        skip_dedup=speech.skip_dedup,
        mode="hint",
    )


def _as_hard(speech: TriggerSpeech) -> TriggerSpeech:
    if speech.mode == "hard":
        return speech
    return TriggerSpeech(
        type=speech.type,
        intent=speech.intent,
        reply=speech.reply,
        stance=speech.stance,
        reason=speech.reason,
        allow_polish=speech.allow_polish,
        skip_dedup=speech.skip_dedup,
        mode="hard",
    )


def _f(scene: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(scene.get(key, default))
    except (TypeError, ValueError):
        return default


def _i(scene: dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(scene.get(key, default))
    except (TypeError, ValueError):
        return default


def _b(scene: dict[str, Any], key: str, default: bool = False) -> bool:
    val = scene.get(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("1", "true", "yes")
    return bool(val)


def _room_clear_still_actionable(scene_info: dict[str, Any], events: list[Any]) -> bool:
    """清房台词仅在「当前仍站在已清未走的房间」时成立。"""
    if "room_enter" in events or "floor_enter" in events:
        return False
    if str(scene_info.get("floor_state", "playing")) != "playing":
        return False
    if not _b(scene_info, "room_cleared"):
        return False
    if not _b(scene_info, "door_open"):
        return False
    if _i(scene_info, "enemy_count") > 0:
        return False
    return True


def _ally_near_player(scene: dict[str, Any], *, threshold: float = 80) -> bool:
    if "ally_near_player" in scene:
        return _b(scene, "ally_near_player")
    return _f(scene, "ally_player_distance", 999) <= threshold


def _ally_guarding_player(scene: dict[str, Any]) -> bool:
    if "ally_already_guarding" in scene:
        return _b(scene, "ally_already_guarding")
    return str(scene.get("ally_stance", "")) == "guard" and _ally_near_player(scene)


def assault_recommended(
    scene_info: dict[str, Any],
    cfg: dict[str, Any],
    *,
    opening: bool = False,
) -> bool:
    """当前局面是否适合从 guard 切到 assault。"""
    triggers = cfg.get("triggers", {})
    if str(scene_info.get("ally_stance", "guard")) != "guard":
        return False
    if _i(scene_info, "enemy_count") <= 0:
        return False
    if _b(scene_info, "ally_in_danger") or _b(scene_info, "player_in_danger"):
        return False

    ally_pct = _f(scene_info, "ally_hp_pct", 1)
    player_pct = _f(scene_info, "player_hp_pct", 1)
    if ally_pct < float(triggers.get("assault_min_ally_hp_pct", 0.35)):
        return False
    if player_pct < float(triggers.get("assault_min_player_hp_pct", 0.35)):
        return False

    max_enemies = int(triggers.get("assault_max_enemies", 10))
    if _i(scene_info, "enemy_count") > max_enemies:
        return False

    nearest = _f(scene_info, "nearest_enemy_distance", 999)
    if nearest < 0 or nearest > float(triggers.get("assault_max_nearest_dist", 420)):
        return False

    if opening:
        return True

    if _b(scene_info, "player_is_idle"):
        return False
    active_s = float(triggers.get("assault_player_active_s", 6))
    if _f(scene_info, "player_last_action_s", 99) > active_s:
        return False

    guard_min_s = float(triggers.get("assault_guard_min_s", 10))
    if _f(scene_info, "ally_guard_duration_s", 0) < guard_min_s:
        return False

    return True


def _evaluate_assault(
    scene_info: dict[str, Any],
    cfg: dict[str, Any],
    *,
    opening: bool = False,
) -> TriggerSpeech | None:
    if not assault_recommended(scene_info, cfg, opening=opening):
        return None
    intent_key = "stance_assault_opening" if opening else "stance_assault"
    return TriggerSpeech(
        type="command",
        intent="stance_assault",
        stance="assault",
        reply=_pick(intent_key),
        reason="assault_opening" if opening else "assault_opportunity",
        mode="hard",
    )


def evaluate_p0(scene_info: dict[str, Any], cfg: dict[str, Any]) -> TriggerSpeech | None:
    triggers = cfg.get("triggers", {})
    ally_pct = _f(scene_info, "ally_hp_pct", 1.0)
    player_pct = _f(scene_info, "player_hp_pct", 1.0)
    player_idle = bool(scene_info.get("player_is_idle"))
    player_idle_s = _f(scene_info, "player_last_action_s")
    ally_stance = str(scene_info.get("ally_stance", "guard"))
    enemy_count = _i(scene_info, "enemy_count")
    guarding = _ally_guarding_player(scene_info)
    near_player = _ally_near_player(scene_info)

    ally_crit = float(triggers.get("ally_critical_hp_pct", 0.25))
    ally_dire = float(triggers.get("ally_dire_hp_pct", 0.10))
    player_danger = float(triggers.get("player_danger_hp_pct", 0.30))
    idle_for_help = float(triggers.get("player_idle_for_help_s", 8))

    if ally_pct <= ally_dire:
        if guarding:
            return TriggerSpeech(
                type="dialogue",
                intent="call_for_help",
                reply=_pick("call_for_help_guarding_dire"),
                reason="ally_dire_guarding",
                skip_dedup=True,
            )
        return TriggerSpeech(
            type="command",
            intent="call_for_help",
            stance="guard",
            reply=_pick("call_for_help_dire"),
            reason="ally_dire",
            skip_dedup=True,
        )

    if ally_pct <= ally_crit:
        if guarding:
            return TriggerSpeech(
                type="dialogue",
                intent="call_for_help",
                reply=_pick("call_for_help_guarding"),
                reason="ally_critical_guarding",
            )
        if ally_stance == "assault" and not near_player:
            return TriggerSpeech(
                type="command",
                intent="call_for_help",
                stance="guard",
                reply=_pick("call_for_help"),
                reason="ally_critical_retreat",
            )
        return TriggerSpeech(
            type="dialogue",
            intent="call_for_help",
            reply=_pick("call_for_help"),
            reason="ally_critical",
        )

    if player_pct <= player_danger and ally_stance == "assault":
        return TriggerSpeech(
            type="command",
            intent="warn_player",
            stance="guard",
            reply=_pick("warn_player_guard"),
            reason="player_danger_assault",
            skip_dedup=True,
        )

    if player_pct <= player_danger and player_idle and player_idle_s >= idle_for_help:
        return TriggerSpeech(
            type="dialogue",
            intent="warn_player",
            reply=_pick("warn_player_idle"),
            reason="player_idle_danger",
            skip_dedup=True,
        )

    if ally_pct <= 0.40 and enemy_count >= 3:
        return TriggerSpeech(
            type="dialogue",
            intent="tactical_bark",
            reply="敌太多了，先集火一个，别散开！",
            reason="ally_stressed",
        )

    return None


def evaluate_p1(
    scene_info: dict[str, Any],
    st: NarrativeState,
    cfg: dict[str, Any],
) -> TriggerSpeech | None:
    kind = str(scene_info.get("scene_change_kind", "")).strip()
    floor = _i(scene_info, "floor")
    events = scene_info.get("recent_events") or []

    if (
        (kind == "blessing_picked" or "blessing_picked" in events)
        and "floor_enter" not in events
        and (scene_info.get("blessing_just_picked") or str(scene_info.get("floor_state", "")) == "blessing_pick")
    ):
        picked = scene_info.get("blessing_just_picked") or {}
        name = str(picked.get("name", "")).strip()
        reply = _pick_stance(scene_info, "blessing_picked_guard", "blessing_picked_assault")
        if name:
            reply = f"「{name}」烙好了。{reply}"
        return _as_hint(TriggerSpeech(
            type="dialogue",
            intent="blessing_picked",
            reply=reply,
            reason="blessing_picked",
        ))

    if ("room_enter" in events or kind == "room_enter") and str(scene_info.get("room_type", "")) == "elite":
        return _as_hint(TriggerSpeech(
            type="dialogue",
            intent="room_enter_elite",
            reply=_pick("room_enter_elite"),
            reason="room_enter_elite",
        ))

    if kind in ("floor_enter", "opening") or "floor_enter" in events:
        assault = _evaluate_assault(scene_info, cfg, opening=True)
        if assault is not None:
            return assault
        if kind == "floor_enter" or "floor_enter" in events:
            if st.last_floor_commented != floor:
                return _as_hint(TriggerSpeech(
                    type="dialogue",
                    intent="floor_react",
                    reply=_pick_stance(scene_info, "floor_enter_guard", "floor_enter_assault"),
                    reason="floor_enter",
                ))

    if ("room_cleared" in events or kind == "room_cleared") and _room_clear_still_actionable(scene_info, events):
        return _as_hint(TriggerSpeech(
            type="dialogue",
            intent="room_cleared",
            reply=_pick("room_cleared"),
            reason="room_cleared",
        ))

    if "boss_spawn" in events and st.boss_warned_floor != floor:
        bname = str(scene_info.get("boss_name", "")).strip()
        reply = _pick("boss_spawn")
        if bname:
            reply = f"{bname}来了。{reply}"
        return _as_hint(TriggerSpeech(
            type="dialogue",
            intent="floor_react",
            reply=reply,
            reason="boss_spawn",
        ))

    if "boss_enrage" in events:
        bname = str(scene_info.get("boss_name", "")).strip()
        reply = _pick("boss_enrage")
        if bname:
            reply = f"{bname}{reply}"
        return _as_hint(TriggerSpeech(
            type="dialogue",
            intent="tactical_bark",
            reply=reply,
            reason="boss_enrage",
        ))

    if "elite_enrage" in events:
        ename = str(scene_info.get("elite_name", "")).strip()
        reply = _pick("elite_enrage")
        if ename:
            reply = f"{ename}：{reply}"
        return _as_hint(TriggerSpeech(
            type="dialogue",
            intent="tactical_bark",
            reply=reply,
            reason="elite_enrage",
        ))

    prev_ec = _i(scene_info, "prev_enemy_count", -1)
    ec = _i(scene_info, "enemy_count")
    if prev_ec >= 0:
        if ec - prev_ec >= 3:
            return _as_hint(TriggerSpeech(
                type="dialogue",
                intent="tactical_bark",
                reply=_pick("tactical_surge"),
                reason="enemy_surge",
            ))
        if prev_ec >= 3 and ec <= 1:
            thin_assault = _evaluate_assault(scene_info, cfg)
            if thin_assault is not None:
                return thin_assault
            return _as_hint(TriggerSpeech(
                type="dialogue",
                intent="tactical_bark",
                reply=_pick("tactical_thin"),
                reason="enemy_thin",
            ))

    assault = _evaluate_assault(scene_info, cfg)
    if assault is not None:
        return assault

    if (
        "floor_clear" in events
        and st.combat_mood in ("engaged", "critical")
        and _i(scene_info, "enemy_count") <= 0
        and _b(scene_info, "room_cleared")
        and "room_enter" not in events
        and "floor_enter" not in events
    ):
        return _as_hint(TriggerSpeech(
            type="dialogue",
            intent="relieved",
            reply=_pick("enemy_clear"),
            reason="floor_clear",
        ))

    return None


def evaluate_p2(
    scene_info: dict[str, Any],
    st: NarrativeState,
    cfg: dict[str, Any],
) -> TriggerSpeech | None:
    triggers = cfg.get("triggers", {})
    social_s = float(triggers.get("social_beat_s", 45))
    since_speech = _f(scene_info, "since_last_npc_speech", 999)

    if st.combat_mood == "critical":
        return None

    tactical = _evaluate_tactical(scene_info)
    if tactical is not None:
        return _as_hint(tactical)

    assault = _evaluate_assault(scene_info, cfg)
    if assault is not None:
        return assault

    return None


def _evaluate_tactical(scene_info: dict[str, Any]) -> TriggerSpeech | None:
    if _b(scene_info, "hazard_near_player"):
        return TriggerSpeech(
            type="dialogue",
            intent="hazard_dodge",
            reply=_pick("hazard_dodge"),
            reason="hazard_near_player",
        )
    if _b(scene_info, "player_silenced"):
        return TriggerSpeech(
            type="dialogue",
            intent="player_silenced",
            reply=_pick("player_silenced"),
            reason="player_silenced",
        )
    nearest = _f(scene_info, "nearest_enemy_distance", 999)
    incoming_p = _i(scene_info, "incoming_bullets_player")
    incoming_a = _i(scene_info, "incoming_bullets_ally")

    if _b(scene_info, "player_under_fire") and _b(scene_info, "player_is_idle"):
        return TriggerSpeech(
            type="dialogue",
            intent="tactical_bark",
            reply=_pick("under_fire"),
            reason="player_under_fire",
        )

    if incoming_p >= 2 or incoming_a >= 2:
        return TriggerSpeech(
            type="dialogue",
            intent="tactical_bark",
            reply=_pick("dodge_bullets"),
            reason="incoming_bullets",
        )

    if _b(scene_info, "los_blocked") and nearest >= 0 and nearest < 250:
        return TriggerSpeech(
            type="dialogue",
            intent="tactical_bark",
            reply=_pick("los_blocked"),
            reason="los_blocked",
        )

    if _b(scene_info, "ally_nav_stuck"):
        return TriggerSpeech(
            type="dialogue",
            intent="tactical_bark",
            reply=_pick("ally_nav_stuck"),
            reason="ally_nav_stuck",
        )

    return None


def evaluate_p3(
    scene_info: dict[str, Any],
    st: NarrativeState,
    cfg: dict[str, Any],
    *,
    force_banter: bool = False,
) -> TriggerSpeech | None:
    triggers = cfg.get("triggers", {})
    social_s = float(triggers.get("social_beat_s", 45))
    since_speech = _f(scene_info, "since_last_npc_speech", 999)

    if not force_banter:
        return None

    if st.combat_mood == "critical":
        return None

    tactical = _evaluate_tactical(scene_info)
    if tactical is not None:
        return TriggerSpeech(
            type=tactical.type,
            intent=tactical.intent,
            reply=tactical.reply,
            stance=tactical.stance,
            reason=tactical.reason,
            mode="fallback",
        )

    assault = _evaluate_assault(scene_info, cfg)
    if assault is not None:
        return assault

    if st.combat_mood == "relieved":
        return TriggerSpeech(
            type="dialogue",
            intent="relieved",
            reply=_pick("relieved"),
            reason="mood_relieved",
            mode="fallback",
        )

    return TriggerSpeech(
        type="dialogue",
        intent="banter",
        reply=_pick_stance(scene_info, "banter_guard", "banter_assault"),
        reason="periodic_banter",
        mode="fallback",
    )


def hint_for_scene(speech: TriggerSpeech | None) -> list[dict[str, str]]:
    if speech is None or speech.mode != "hint":
        return []
    return [{
        "intent": speech.intent,
        "hint": speech.reply,
        "reason": speech.reason,
    }]


def stance_semantics(scene_info: dict[str, Any]) -> str:
    stance = _ally_stance(scene_info)
    if stance == "assault":
        return "突击：乌枭前压开路，玩家在后支援，可说「跟上/我先上」。"
    near = _ally_guarding_player(scene_info)
    if near:
        return "守护贴身：乌枭跟随玩家，禁喊「跟紧我/过来/别愣着」指玩家过来。"
    return "守护：乌枭趋向玩家，禁把玩家当成需要被喊过来的一方。"


def trigger_to_decision(speech: TriggerSpeech) -> dict[str, Any]:
    if speech.type == "command":
        return {
            "type": "command",
            "stance": speech.stance,
            "reply": speech.reply,
            "intent": speech.intent,
        }
    return {"type": "dialogue", "reply": speech.reply, "intent": speech.intent}