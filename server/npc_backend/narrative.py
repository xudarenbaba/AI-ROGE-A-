from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any


def scene_hash(scene_info: dict[str, Any]) -> str:
    return "|".join(
        str(scene_info.get(k, ""))
        for k in (
            "floor",
            "room_index",
            "room_type",
            "door_open",
            "floor_state",
            "player_hp",
            "ally_hp",
            "enemy_count",
            "ally_stance",
            "boss_alive",
            "blessings_total",
        )
    )


def _normalize_text(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text.lower())


def _bigrams(text: str) -> set[str]:
    norm = _normalize_text(text)
    if len(norm) < 2:
        return {norm} if norm else set()
    return {norm[i : i + 2] for i in range(len(norm) - 1)}


def text_similarity(a: str, b: str) -> float:
    sa, sb = _bigrams(a), _bigrams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_duplicate_reply(
    reply: str,
    recent_texts: list[str],
    *,
    similarity_threshold: float = 0.55,
) -> bool:
    reply = reply.strip()
    if not reply:
        return True
    norm_reply = _normalize_text(reply)
    for prev in recent_texts:
        prev = prev.strip()
        if not prev:
            continue
        norm_prev = _normalize_text(prev)
        if norm_reply == norm_prev:
            return True
        if len(norm_reply) >= 6 and len(norm_prev) >= 6:
            if norm_reply in norm_prev or norm_prev in norm_reply:
                return True
        if text_similarity(reply, prev) >= similarity_threshold:
            return True
    return False


@dataclass
class AutonomousLine:
    at: float
    type: str
    text: str = ""
    stance: str | None = None
    intent: str = ""


@dataclass
class NarrativeState:
    tactical_intent: str | None = None
    intent_reason: str = ""
    intent_set_at: float = 0.0
    combat_mood: str = "observing"
    recent_lines: list[AutonomousLine] = field(default_factory=list)
    last_stance_change_at: float = 0.0
    last_speech_at: float = 0.0
    consecutive_noop: int = 0
    last_scene_hash: str = ""
    last_think_at: float = 0.0
    recent_reply_texts: list[str] = field(default_factory=list)
    speech_timestamps: list[float] = field(default_factory=list)
    last_floor_commented: int = 0
    boss_warned_floor: int = -1
    prev_enemy_count: int = -1


class NarrativeStore:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._lock = threading.Lock()
        self._states: dict[str, NarrativeState] = {}

    @staticmethod
    def _key(player_id: str, npc_id: str) -> str:
        return f"{player_id}:{npc_id}"

    def get(self, player_id: str, npc_id: str) -> NarrativeState:
        with self._lock:
            return self._states.setdefault(self._key(player_id, npc_id), NarrativeState())

    def on_player_message(self, player_id: str, npc_id: str) -> None:
        with self._lock:
            st = self._states.setdefault(self._key(player_id, npc_id), NarrativeState())
            st.consecutive_noop = 0

    def update_mood(self, st: NarrativeState, scene_info: dict[str, Any]) -> None:
        ally_pct = float(scene_info.get("ally_hp_pct", 1) or 1)
        player_pct = float(scene_info.get("player_hp_pct", 1) or 1)
        events = scene_info.get("recent_events") or []
        enemy_count = int(scene_info.get("enemy_count", 0) or 0)

        if ally_pct <= 0.25 or player_pct <= 0.30:
            st.combat_mood = "critical"
        elif ally_pct > 0.55 and player_pct > 0.50:
            if st.combat_mood == "critical":
                st.combat_mood = "relieved"
            elif st.combat_mood == "relieved":
                pass
            elif enemy_count > 0:
                st.combat_mood = "engaged"
        elif st.combat_mood == "observing" and (
            enemy_count > 0
            or "floor_enter" in events
            or "boss_spawn" in events
            or "hp_drop" in events
        ):
            st.combat_mood = "engaged"

        if st.combat_mood == "relieved" and st.last_speech_at:
            if time.time() - st.last_speech_at > 8:
                st.combat_mood = "observing"

    def mood_block(self, st: NarrativeState) -> str:
        hints = {
            "observing": "观察中，可偶尔碎嘴，少切姿态。",
            "engaged": "交战活跃，可战术点评或轻量吐槽。",
            "critical": "危急，优先求援/护主/催促玩家，姿态偏保守。",
            "relieved": "刚脱险，可一句碎嘴后恢复平静。",
        }
        return f"战斗情绪：{st.combat_mood}。{hints.get(st.combat_mood, '')}"

    def recent_lines_for_prompt(self, st: NarrativeState, max_items: int = 5) -> str:
        if not st.recent_lines:
            return "无"
        lines: list[str] = []
        now = time.time()
        for i, item in enumerate(reversed(st.recent_lines[-max_items:]), start=1):
            ago = max(0, int(now - item.at))
            if item.type == "noop":
                lines.append(f"{i}. [{ago}s前] noop")
            elif item.type == "command":
                lines.append(f"{i}. [{ago}s前] command→{item.stance}：「{item.text}」")
            else:
                lines.append(f"{i}. [{ago}s前] {item.intent or 'dialogue'}：「{item.text}」")
        return "\n".join(lines)

    def intent_block(self, st: NarrativeState) -> str:
        intent_ttl = float(self._cfg.get("intent_ttl_s", 30))
        if not st.tactical_intent:
            return "当前战术意图：未设定"
        age = time.time() - st.intent_set_at
        if age > intent_ttl:
            return f"当前战术意图：{st.tactical_intent}（已过期，可调整）"
        return f"当前战术意图：{st.tactical_intent}（{int(age)}s前，勿轻易反转）"

    def scene_dramatic_change(self, st: NarrativeState, scene_info: dict[str, Any]) -> bool:
        new_hash = scene_hash(scene_info)
        if not st.last_scene_hash:
            return True
        if new_hash != st.last_scene_hash:
            old_parts = st.last_scene_hash.split("|")
            new_parts = new_hash.split("|")
            if old_parts and new_parts and old_parts[0] != new_parts[0]:
                return True
            try:
                if len(old_parts) >= 5 and len(new_parts) >= 5:
                    old_ph, old_ah, old_ec = int(old_parts[2]), int(old_parts[3]), int(old_parts[4])
                    new_ph, new_ah, new_ec = int(new_parts[2]), int(new_parts[3]), int(new_parts[4])
                    if abs(new_ph - old_ph) >= 20 or abs(new_ah - old_ah) >= 25:
                        return True
                    if abs(new_ec - old_ec) >= 2:
                        return True
            except ValueError:
                return True
            return True
        return False

    def can_speak(
        self,
        st: NarrativeState,
        *,
        priority: int,
        trigger: str = "",
    ) -> str | None:
        now = time.time()
        max_per_min = int(self._cfg.get("max_speech_per_minute", 4))
        st.speech_timestamps = [t for t in st.speech_timestamps if now - t < 60]
        if len(st.speech_timestamps) >= max_per_min:
            return "speech_rate_limit"

        if priority == 0:
            speech_cd = float(self._cfg.get("critical_speech_cooldown_s", 6))
        elif priority == 1 and trigger in ("scene_change", "critical"):
            speech_cd = float(self._cfg.get("scene_speech_cooldown_s", 4))
        else:
            speech_cd = float(self._cfg.get("speech_cooldown_s", 12))
        if st.last_speech_at and now - st.last_speech_at < speech_cd:
            return f"speech_cooldown({speech_cd}s)"
        return None

    @staticmethod
    def _violates_guard_dialogue(reply: str, scene_info: dict[str, Any]) -> bool:
        stance = str(scene_info.get("ally_stance", "guard"))
        if stance != "guard":
            return False
        near = bool(
            scene_info.get("ally_near_player")
            or scene_info.get("ally_already_guarding")
        )
        if not near:
            return False
        norm = _normalize_text(reply)
        banned = (
            "跟紧我", "跟紧", "过来", "靠近", "别愣着", "别发呆", "别乱跑",
            "跟我", "跟上我", "快过来", "你倒是过来", "别掉队",
        )
        return any(_normalize_text(phrase) in norm for phrase in banned)

    def should_rule_noop(
        self,
        st: NarrativeState,
        scene_info: dict[str, Any],
        *,
        priority: int = 3,
        now: float | None = None,
    ) -> str | None:
        now = now or time.time()
        autonomy = self._cfg

        trigger = str(scene_info.get("trigger", ""))
        if priority <= 0:
            return self.can_speak(st, priority=priority, trigger=trigger)

        if trigger in ("social", "periodic"):
            since_speech = float(scene_info.get("since_last_npc_speech", 0) or 0)
            social_s = float(autonomy.get("triggers", {}).get("social_beat_s", 45))
            if since_speech >= social_s:
                return None

        rate_reason = self.can_speak(st, priority=priority, trigger=trigger)
        if rate_reason and priority >= 2 and trigger not in ("social", "periodic"):
            return rate_reason

        if priority >= 3 and st.consecutive_noop >= int(autonomy.get("rule_noop_after", 2)):
            backoff = autonomy.get("noop_backoff_s", [15, 30, 60])
            idx = min(st.consecutive_noop - int(autonomy.get("rule_noop_after", 2)), len(backoff) - 1)
            wait_s = float(backoff[max(0, idx)])
            if now - st.last_think_at < wait_s:
                return f"noop_backoff({wait_s}s)"

        if (
            priority >= 2
            and autonomy.get("require_scene_change", True)
            and trigger not in ("social", "periodic")
        ):
            dramatic = self.scene_dramatic_change(st, scene_info)
            stale_s = float(autonomy.get("stale_force_think_s", 75))
            if not dramatic and st.last_think_at and now - st.last_think_at < stale_s:
                return "scene_unchanged"

        if priority >= 2 and trigger not in ("social", "periodic"):
            cd_reason = self.can_speak(st, priority=priority, trigger=trigger)
            if cd_reason:
                return cd_reason

        if priority >= 3:
            think_interval = float(autonomy.get("think_interval_s", 12))
            if st.last_think_at and now - st.last_think_at < think_interval:
                return f"think_interval({think_interval}s)"

        return None

    def validate_decision(
        self,
        decision: dict[str, Any],
        st: NarrativeState,
        scene_info: dict[str, Any],
        *,
        skip_dedup: bool = False,
        priority: int = 3,
    ) -> tuple[dict[str, Any], str | None]:
        dtype = decision.get("type")
        if dtype == "noop":
            return decision, None

        reply = str(decision.get("reply", "")).strip()
        if reply and not skip_dedup:
            recent = st.recent_reply_texts[-int(self._cfg.get("recent_lines_max", 8)) :]
            sim = float(self._cfg.get("duplicate_similarity", 0.55))
            if is_duplicate_reply(reply, recent, similarity_threshold=sim):
                if priority <= 1:
                    return {"type": "noop"}, "duplicate_reply_soft"
                return {"type": "noop"}, "duplicate_reply"

        if reply and self._violates_guard_dialogue(reply, scene_info):
            return {"type": "noop"}, "guard_coherence"

        if dtype == "command":
            stance = str(decision.get("stance", "")).strip()
            current = str(scene_info.get("ally_stance", "guard"))
            now = time.time()
            assault_ok = bool(scene_info.get("assault_recommended")) and stance == "assault"
            if stance == current:
                if reply:
                    return {"type": "dialogue", "reply": reply, "intent": decision.get("intent", "")}, None
                return {"type": "noop"}, "same_stance"

            stance_cd = (
                float(self._cfg.get("assault_stance_cooldown_s", 10))
                if assault_ok
                else float(self._cfg.get("stance_change_cooldown_s", 25))
            )
            if (
                not assault_ok
                and priority > 0
                and st.last_stance_change_at
                and now - st.last_stance_change_at < stance_cd
            ):
                if reply:
                    return {"type": "dialogue", "reply": reply, "intent": decision.get("intent", "")}, "stance_cooldown_dialogue"
                return {"type": "noop"}, "stance_cooldown"

            if (
                not assault_ok
                and priority > 0
                and st.tactical_intent
                and st.tactical_intent != stance
                and now - st.intent_set_at < float(self._cfg.get("intent_ttl_s", 30))
                and not self.scene_dramatic_change(st, scene_info)
                and st.combat_mood != "critical"
            ):
                if reply:
                    return {"type": "dialogue", "reply": reply}, "intent_conflict_dialogue"
                return {"type": "noop"}, "intent_conflict"

        return decision, None

    def record_outcome(
        self,
        player_id: str,
        npc_id: str,
        *,
        decision: dict[str, Any],
        scene_info: dict[str, Any],
        rejected_reason: str | None = None,
        intent: str = "",
    ) -> None:
        with self._lock:
            st = self._states.setdefault(self._key(player_id, npc_id), NarrativeState())
            now = time.time()
            st.last_think_at = now
            st.last_scene_hash = scene_hash(scene_info)
            st.prev_enemy_count = int(scene_info.get("enemy_count", st.prev_enemy_count))

            dtype = decision.get("type", "noop")
            if rejected_reason and rejected_reason not in ("duplicate_reply_soft",):
                dtype = "noop"

            line_intent = str(decision.get("intent") or intent or "")

            if dtype == "noop":
                st.consecutive_noop += 1
                st.recent_lines.append(AutonomousLine(at=now, type="noop", intent=line_intent))
            elif dtype == "command":
                st.consecutive_noop = 0
                reply = str(decision.get("reply", "")).strip()
                stance = str(decision.get("stance", ""))
                st.recent_lines.append(
                    AutonomousLine(at=now, type="command", text=reply, stance=stance, intent=line_intent)
                )
                if reply:
                    st.recent_reply_texts.append(reply)
                    st.last_speech_at = now
                    st.speech_timestamps.append(now)
                if stance:
                    prev = str(scene_info.get("ally_stance", ""))
                    if stance != prev:
                        st.last_stance_change_at = now
                    st.tactical_intent = stance
                    st.intent_reason = reply or f"切换至{stance}"
                    st.intent_set_at = now
                if line_intent == "floor_react":
                    st.last_floor_commented = int(scene_info.get("floor", st.last_floor_commented))
                if line_intent == "boss_spawn" or "boss_spawn" in (scene_info.get("recent_events") or []):
                    st.boss_warned_floor = int(scene_info.get("floor", -1))
            elif dtype == "dialogue":
                st.consecutive_noop = 0
                reply = str(decision.get("reply", "")).strip()
                st.recent_lines.append(
                    AutonomousLine(at=now, type="dialogue", text=reply, intent=line_intent)
                )
                if reply:
                    st.recent_reply_texts.append(reply)
                    st.last_speech_at = now
                    st.speech_timestamps.append(now)
                if line_intent == "floor_react":
                    st.last_floor_commented = int(scene_info.get("floor", st.last_floor_commented))
                if "boss_spawn" in (scene_info.get("recent_events") or []):
                    st.boss_warned_floor = int(scene_info.get("floor", -1))

            max_lines = int(self._cfg.get("recent_lines_max", 8))
            st.recent_lines = st.recent_lines[-max_lines:]
            st.recent_reply_texts = st.recent_reply_texts[-max_lines:]

    def context_for_prompt(self, player_id: str, npc_id: str) -> dict[str, str]:
        st = self.get(player_id, npc_id)
        return {
            "intent_block": self.intent_block(st),
            "recent_autonomous": self.recent_lines_for_prompt(
                st, int(self._cfg.get("recent_lines_max", 5))
            ),
            "mood_block": self.mood_block(st),
        }