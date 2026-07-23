from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from server.npc_backend.config import load_config
from server.npc_backend.inflight import InflightTracker
from server.npc_backend.log_config import npc_logger
from server.npc_backend.llm import (
    autonomous_decide,
    chat_completion_stream,
    classify_dialogue_memory,
    classify_intent,
    fast_command_intent,
    generate_no_hp_reply,
)
from server.npc_backend.memory import MemoryStore
from server.npc_backend.narrative import NarrativeState, NarrativeStore, scene_hash
from server.npc_backend.polish import polish_line
from server.npc_backend.postprocess import (
    anti_ai_postprocess,
    is_combat_scene,
    strip_emotion_tag,
)
from server.npc_backend.prompts import build_messages
from server.npc_backend.reflex import ReflexDecision, reflex_decide
from server.npc_backend.salient import (
    build_memory_search_query,
    build_salient_snapshot,
    command_reply_from_scene,
    infer_chat_mode,
)
from server.npc_backend.triggers import (
    TriggerSpeech,
    assault_recommended,
    evaluate_p0,
    evaluate_p1,
    evaluate_p2,
    evaluate_p3,
    hint_for_scene,
    stance_semantics,
    trigger_to_decision,
)
from server.npc_backend.schemas import ChatAction
from server.npc_backend.short_term import ShortTermMemory
from server.npc_backend.working import WorkingMemoryStore

_EMOTION_RE = re.compile(r"<emotion>([\w]+)</emotion>\s*$", re.IGNORECASE)
_VALID_EMOTIONS = {
    "neutral", "focused", "annoyed", "worried", "happy", "tense", "sarcastic"
}

# delta 批量缓冲阈值（字符数）：首包更快
_DELTA_BATCH_CHARS = 4


def _enrich_scene_info(
    scene_info: dict[str, Any],
    narrative_st: NarrativeState,
    narrative: NarrativeStore,
    autonomy_cfg: dict[str, Any],
    *,
    trigger: str = "",
    trigger_reason: str = "",
    priority: int = 3,
    opening: bool = False,
    last_npc_line: str = "",
    player_message: str = "",
    mode: str = "think",
) -> dict[str, Any]:
    assault_ok = assault_recommended(scene_info, autonomy_cfg, opening=opening)
    base = {
        **scene_info,
        "trigger": trigger or str(scene_info.get("trigger", "")),
        "trigger_reason": trigger_reason or str(scene_info.get("trigger_reason", "")),
        "priority": priority,
        "scene_hash": scene_hash(scene_info),
        "scene_dramatic_change": narrative.scene_dramatic_change(narrative_st, scene_info),
        "prev_enemy_count": narrative_st.prev_enemy_count,
        "assault_recommended": assault_ok,
        "combat_mood": narrative_st.combat_mood,
        "stance_semantics": stance_semantics(scene_info),
    }
    snap = build_salient_snapshot(
        base,
        last_npc_line=last_npc_line,
        player_message=player_message,
        mode=mode,
    )
    base["salient_snapshot"] = snap
    return base


def _resolve_run_id(payload: dict[str, Any], scene_info: dict[str, Any]) -> str:
    rid = str(payload.get("run_id") or scene_info.get("run_id") or "").strip()
    return rid


def _memory_channel_for_chat(chat_mode: str, scene_info: dict[str, Any]) -> str:
    if chat_mode in ("combat_ack", "combat_question") or is_combat_scene(scene_info):
        return "think_combat" if chat_mode in ("combat_ack", "combat_question") else "chat"
    return "chat"


def _memory_channel_for_think(scene_info: dict[str, Any], trigger: str) -> str:
    if trigger in ("opening",) or str(scene_info.get("scene_change_kind") or "") == "opening":
        return "run_start"
    if is_combat_scene(scene_info) or int(scene_info.get("enemy_count") or 0) > 0:
        return "think_combat"
    return "think_safe"


class NpcConversationEngine:
    def __init__(self) -> None:
        self._cfg = load_config()
        self._memory = MemoryStore()
        self._short_term = ShortTermMemory()
        self._working = WorkingMemoryStore()
        worker_threads = int(self._cfg.get("concurrency", {}).get("worker_threads", 8))
        self._pool = ThreadPoolExecutor(
            max_workers=worker_threads,
            thread_name_prefix="npc-worker",
        )
        self._inflight = InflightTracker()
        self._narrative = NarrativeStore(self._cfg.get("npc_autonomy", {}))
        self._log = npc_logger()

    @property
    def memory(self) -> MemoryStore:
        return self._memory

    @property
    def working(self) -> WorkingMemoryStore:
        return self._working

    def stream_chat(self, payload: dict[str, Any]) -> Iterator[str]:
        """
        统一流式入口，逐行 yield NDJSON 字符串。

        并行：意图分类 ∥ 对话线程 ∥ 长期记忆 ∥ narrative
        场面 salient 同步生成（规则，0ms）。
        command 立即返回模板确认语；dialogue 走 multi-turn 流式。
        """
        player_id: str = payload.get("player_id", "")
        npc_id: str = payload.get("npc_id", "")
        npc_name: str = payload.get("npc_name") or npc_id
        message: str = payload.get("message", "")
        scene_info: dict[str, Any] = payload.get("scene_info") or {}
        run_id = _resolve_run_id(payload, scene_info)
        if run_id:
            scene_info = {**scene_info, "run_id": run_id}
            self._working.set_run_id(player_id, npc_id, run_id)

        self._narrative.on_player_message(player_id, npc_id, message)
        narrative_st = self._narrative.get(player_id, npc_id)
        self._narrative.update_mood(narrative_st, scene_info)
        autonomy_cfg = self._cfg.get("npc_autonomy", {})
        last_line = (
            self._narrative.last_spoken_line(player_id, npc_id)
            or self._short_term.last_assistant_line(player_id, npc_id)
        )
        scene_info = _enrich_scene_info(
            scene_info,
            narrative_st,
            self._narrative,
            autonomy_cfg,
            trigger="reactive",
            last_npc_line=last_line,
            player_message=message,
            mode="chat",
        )
        chat_mode = infer_chat_mode(message, scene_info)
        scene_info["chat_mode"] = chat_mode

        _cancel, generation = self._inflight.begin(player_id, npc_id, "chat")
        yield _event("meta", {
            "npc_id": npc_id,
            "combat_mood": narrative_st.combat_mood,
            "chat_mode": chat_mode,
        })

        try:
            search_query = build_memory_search_query(
                player_message=message,
                scene_info=scene_info,
                npc_name=npc_name,
            )
            mem_channel = _memory_channel_for_chat(chat_mode, scene_info)
            fast_intent = fast_command_intent(message, scene_info)

            fut_chat: Future[list[dict[str, str]]] = self._pool.submit(
                self._short_term.get_chat_thread, player_id, npc_id, max_turns=8
            )
            # executor=None：在 worker 内用本地小池，避免与外层线程池嵌套死锁
            fut_long: Future[dict[str, list[str]]] = self._pool.submit(
                self._memory.search_context,
                search_query,
                player_id,
                npc_id,
                None,
                chat_mode=chat_mode,
                channel=mem_channel,
                run_id=run_id,
            )
            fut_narrative: Future[dict[str, str]] = self._pool.submit(
                self._narrative.context_for_prompt, player_id, npc_id
            )
            fut_intent: Future[dict[str, Any]] | None = None
            if fast_intent is None:
                fut_intent = self._pool.submit(
                    classify_intent,
                    message=message,
                    scene_info=scene_info,
                    npc_name=npc_name,
                )

            intent = fast_intent if fast_intent is not None else fut_intent.result()

            if self._inflight.is_cancelled(player_id, npc_id, generation):
                return

            if intent["type"] == "command":
                stance = intent["stance"]
                ally_hp = int(scene_info.get("ally_hp", 100))
                if stance == "assault" and ally_hp <= 0:
                    refusal = generate_no_hp_reply(npc_name=npc_name)
                    action = ChatAction(
                        action_type="dialogue", dialogue=refusal, emotion="annoyed"
                    )
                    self._schedule_memory_write(
                        player_id=player_id,
                        npc_id=npc_id,
                        scene_info=scene_info,
                        player_message=message,
                        reply=refusal,
                        memory_source="player",
                    )
                    yield _event("done", {"action": action.model_dump(exclude_none=True)})
                    return

                current = str(scene_info.get("ally_stance", ""))
                same = current == stance
                reply = str(intent.get("reply") or "").strip()
                if not reply or len(reply) < 4:
                    reply = command_reply_from_scene(stance, scene_info, same_stance=same)

                ops = intent.get("ops") or []
                limb_id = "assault_skirmish" if stance == "assault" else "guard_follow"
                self._working.note_player_order(player_id, npc_id, message)
                self._working.set_commitment(
                    player_id,
                    npc_id,
                    limb_id=limb_id,
                    stance=stance,
                    source="player",
                    ttl_sec=12.0,
                    reason=f"玩家指令:{message[:40]}",
                )
                # 本局事件：玩家战术指令
                if run_id:
                    self._pool.submit(
                        self._memory.add_run_event,
                        player_id=player_id,
                        npc_id=npc_id,
                        run_id=run_id,
                        text=(
                            f"玩家指令切换姿态为{stance}"
                            f"（第{scene_info.get('floor', '?')}层）"
                        ),
                        tier="major",
                        source="chat",
                        scene_info=scene_info,
                        tags=["player_order", stance],
                    )
                yield _event("command", {
                    "stance": stance,
                    "reply": reply,
                    "stance_changed": not same,
                    "ops": ops,
                    "limb_id": limb_id,
                })
                # 指令也进对话线程，便于后续承接
                self._schedule_memory_write(
                    player_id=player_id,
                    npc_id=npc_id,
                    scene_info=scene_info,
                    player_message=message,
                    reply=reply,
                    memory_source="player",
                    skip_long_term=True,
                    run_id=run_id,
                )
                # 可选异步润色（不阻塞姿态切换）
                polish_cfg = autonomy_cfg.get("polish", {})
                if polish_cfg.get("command_reply", True):
                    yield from self._maybe_polish(
                        reply,
                        player_id=player_id,
                        npc_id=npc_id,
                        npc_name=npc_name,
                        scene_info=scene_info,
                        generation=generation,
                        intent=f"command_{stance}",
                        skip=False,
                        quick=True,
                    )
                return

            # dialogue：等待并行结果
            chat_thread: list[dict[str, str]] = fut_chat.result()
            long_term_ctx: dict[str, list[str]] = fut_long.result()
            narrative_ctx: dict[str, str] = fut_narrative.result()

            if self._inflight.is_cancelled(player_id, npc_id, generation):
                return

            yield _event("typing", {"active": True})

            snap = str(scene_info.get("salient_snapshot") or "")
            if snap:
                self._working.note_scene_summary(player_id, npc_id, snap)
            working_block = self._working.prompt_block(player_id, npc_id)
            messages = build_messages(
                npc_name=npc_name,
                player_message=message,
                scene_info=scene_info,
                world_chunks=long_term_ctx.get("world_chunks", []),
                persona_chunks=long_term_ctx.get("persona_chunks", []),
                dialogue_daily_chunks=long_term_ctx.get("dialogue_daily_chunks", []),
                dialogue_important_chunks=long_term_ctx.get("dialogue_important_chunks", []),
                short_term_history=[],
                narrative_context=narrative_ctx,
                chat_thread=chat_thread,
                chat_mode=chat_mode,
                last_npc_line=last_line,
                run_event_chunks=long_term_ctx.get("run_event_chunks", []),
                reflection_chunks=long_term_ctx.get("reflection_chunks", []),
                working_block=working_block,
            )

            combat = is_combat_scene(scene_info) or chat_mode in (
                "combat_ack", "combat_question"
            )
            max_tokens = 100 if combat else 220
            temperature = 0.45 if chat_mode == "rest_banter" else 0.35

            yield from self._stream_dialogue(
                messages=messages,
                player_id=player_id,
                npc_id=npc_id,
                npc_name=npc_name,
                scene_info=scene_info,
                player_message=message,
                generation=generation,
                memory_source="player",
                max_tokens=max_tokens,
                temperature=temperature,
                combat=combat,
            )

        except Exception as exc:  # noqa: BLE001
            yield from self._yield_error(exc)
        finally:
            self._inflight.end(player_id, npc_id, generation)

    def stream_think(self, payload: dict[str, Any]) -> Iterator[str]:
        """自主思考：多触发源 + 规则台词 + LLM + 可选润色。"""
        autonomy_cfg = self._cfg.get("npc_autonomy", {})
        player_id: str = payload.get("player_id", "")
        npc_id: str = payload.get("npc_id", "")
        npc_name: str = payload.get("npc_name") or npc_id
        scene_info: dict[str, Any] = payload.get("scene_info") or {}
        trigger = str(payload.get("trigger", "periodic"))
        priority = int(payload.get("priority", 3))
        trigger_reason = str(payload.get("trigger_reason", ""))
        run_id = _resolve_run_id(payload, scene_info)
        if run_id:
            scene_info = {**scene_info, "run_id": run_id}
            self._working.set_run_id(player_id, npc_id, run_id)

        if not autonomy_cfg.get("enabled", True):
            yield _event("noop", {"reason": "disabled"})
            return

        if scene_info.get("can_autonomy_speak") is False:
            yield _event("noop", {"reason": "silent_phase"})
            return

        narrative_st = self._narrative.get(player_id, npc_id)
        self._narrative.update_mood(narrative_st, scene_info)
        opening = trigger_reason in ("floor_enter", "opening") or "floor_enter" in (
            scene_info.get("recent_events") or []
        )
        last_line = self._narrative.last_spoken_line(player_id, npc_id)
        scene_info = _enrich_scene_info(
            scene_info,
            narrative_st,
            self._narrative,
            autonomy_cfg,
            trigger=trigger,
            trigger_reason=trigger_reason,
            priority=priority,
            opening=opening,
            last_npc_line=last_line,
            mode="think",
        )

        situation = (
            f"第{scene_info.get('floor', '?')}层"
            f" 玩家HP={scene_info.get('player_hp', '?')}"
            f" 友军HP={scene_info.get('ally_hp', '?')}"
            f" 敌={scene_info.get('enemy_count', '?')}"
        )
        prefetch_query = build_memory_search_query(
            player_message=situation,
            scene_info=scene_info,
            npc_name=npc_name,
            trigger=trigger,
        )
        mem_channel = _memory_channel_for_think(scene_info, trigger)
        fut_chat = self._pool.submit(
            self._short_term.get_chat_thread, player_id, npc_id, max_turns=6
        )
        fut_long = self._pool.submit(
            self._memory.search_context,
            prefetch_query,
            player_id,
            npc_id,
            None,
            chat_mode="combat_ack",
            channel=mem_channel,
            run_id=run_id,
        )

        rule_speech: TriggerSpeech | None = None
        if priority <= 0:
            rule_speech = evaluate_p0(scene_info, autonomy_cfg)
        elif priority == 1:
            rule_speech = evaluate_p1(scene_info, narrative_st, autonomy_cfg)
        elif priority == 2:
            rule_speech = evaluate_p2(scene_info, narrative_st, autonomy_cfg)
        elif priority >= 3:
            rule_speech = evaluate_p3(scene_info, narrative_st, autonomy_cfg)

        if rule_speech is None or rule_speech.mode != "hard":
            rule_reason = self._narrative.should_rule_noop(
                narrative_st, scene_info, priority=priority
            )
            if rule_reason:
                self._log.info(
                    "think rule_noop player=%s trigger=%s p=%s reason=%s",
                    player_id, trigger, priority, rule_reason,
                )
                self._narrative.record_outcome(
                    player_id, npc_id, decision={"type": "noop"}, scene_info=scene_info,
                )
                yield _event("noop", {"reason": rule_reason})
                return

        self._log.info(
            "think triggered player=%s trigger=%s p=%s reason=%s mood=%s",
            player_id, trigger, priority, trigger_reason or "-", narrative_st.combat_mood,
        )

        _cancel, generation = self._inflight.begin(player_id, npc_id, "think")
        yield _event("meta", {
            "npc_id": npc_id,
            "combat_mood": narrative_st.combat_mood,
            "trigger": trigger,
        })

        try:
            if rule_speech is not None and rule_speech.mode == "hard":
                cd_reason = self._narrative.can_speak(
                    narrative_st, priority=priority, trigger=trigger,
                )
                if cd_reason:
                    self._narrative.record_outcome(
                        player_id, npc_id,
                        decision={"type": "noop"}, scene_info=scene_info,
                    )
                    yield _event("noop", {"reason": cd_reason})
                    return
                decision = trigger_to_decision(rule_speech)
                validated, reject = self._narrative.validate_decision(
                    decision,
                    narrative_st,
                    scene_info,
                    skip_dedup=rule_speech.skip_dedup,
                    priority=priority,
                )
                self._log.info(
                    "think rule_hard player=%s intent=%s type=%s reject=%s",
                    player_id, rule_speech.intent, validated.get("type"), reject or "-",
                )
                self._narrative.record_outcome(
                    player_id, npc_id,
                    decision=validated,
                    scene_info=scene_info,
                    rejected_reason=reject,
                    intent=rule_speech.intent,
                )
                yield from self._yield_validated_decision(
                    validated,
                    player_id=player_id,
                    npc_id=npc_id,
                    npc_name=npc_name,
                    scene_info=scene_info,
                    generation=generation,
                    memory_tag=f"[NPC自主/{rule_speech.intent}]",
                    polish_intent=rule_speech.intent,
                    priority=priority,
                    skip_polish=True,
                )
                return

            if priority <= 0 and bool(autonomy_cfg.get("reflex_enabled", True)):
                reflex = reflex_decide(scene_info, reflex_enabled=True)
                if reflex is not None:
                    reflex_decision = self._reflex_to_decision(reflex)
                    validated, reject = self._narrative.validate_decision(
                        reflex_decision, narrative_st, scene_info,
                        skip_dedup=True, priority=0,
                    )
                    self._narrative.record_outcome(
                        player_id, npc_id,
                        decision=validated, scene_info=scene_info,
                        rejected_reason=reject, intent="reflex",
                    )
                    yield from self._yield_validated_decision(
                        validated,
                        player_id=player_id,
                        npc_id=npc_id,
                        npc_name=npc_name,
                        scene_info=scene_info,
                        generation=generation,
                        memory_tag="[NPC自主/reflex]",
                        priority=priority,
                        skip_polish=True,
                    )
                    return

            allowed = self._allowed_intents_for_trigger(
                trigger, narrative_st, scene_info,
            )
            chat_thread = fut_chat.result()
            long_term_ctx = fut_long.result()

            if self._inflight.is_cancelled(player_id, npc_id, generation):
                return

            narrative_ctx = self._narrative.context_for_prompt(player_id, npc_id)
            llm_scene = {
                **scene_info,
                "rule_hints": hint_for_scene(rule_speech),
                "stance_semantics": stance_semantics(scene_info),
            }
            polish_cfg = autonomy_cfg.get("polish", {})
            llm_polish_enabled = bool(polish_cfg.get("autonomous_quick", True))
            snap = str(scene_info.get("salient_snapshot") or "")
            if snap:
                self._working.note_scene_summary(player_id, npc_id, snap)
            working_block = self._working.prompt_block(player_id, npc_id)
            decision = self._pool.submit(
                autonomous_decide,
                npc_name=npc_name,
                scene_info=llm_scene,
                world_chunks=long_term_ctx.get("world_chunks", []),
                persona_chunks=long_term_ctx.get("persona_chunks", []),
                dialogue_daily_chunks=long_term_ctx.get("dialogue_daily_chunks", []),
                dialogue_important_chunks=long_term_ctx.get("dialogue_important_chunks", []),
                short_term_history=[],
                narrative_context=narrative_ctx,
                allowed_intents=allowed,
                trigger=trigger,
                chat_thread=chat_thread,
                run_event_chunks=long_term_ctx.get("run_event_chunks", []),
                reflection_chunks=long_term_ctx.get("reflection_chunks", []),
                working_block=working_block,
            ).result()

            validated, reject = self._narrative.validate_decision(
                decision, narrative_st, llm_scene, priority=priority,
            )
            fallback_speech: TriggerSpeech | None = None
            if validated.get("type") == "noop" and trigger in ("periodic", "social", "scene_change"):
                fallback_speech = evaluate_p3(
                    scene_info, narrative_st, autonomy_cfg, force_banter=True,
                )
                if fallback_speech is not None and fallback_speech.mode == "hard":
                    fb_decision = trigger_to_decision(fallback_speech)
                    validated, reject = self._narrative.validate_decision(
                        fb_decision,
                        narrative_st,
                        scene_info,
                        priority=priority,
                    )
                    self._log.info(
                        "think fallback_hard player=%s intent=%s type=%s",
                        player_id, fallback_speech.intent, validated.get("type"),
                    )
                    decision = fb_decision
                    yield from self._yield_validated_decision(
                        validated,
                        player_id=player_id,
                        npc_id=npc_id,
                        npc_name=npc_name,
                        scene_info=scene_info,
                        generation=generation,
                        memory_tag=f"[NPC自主/{fallback_speech.intent}]",
                        polish_intent=fallback_speech.intent,
                        priority=priority,
                        skip_polish=True,
                    )
                    return
                if fallback_speech is not None:
                    fb_decision = trigger_to_decision(fallback_speech)
                    validated, reject = self._narrative.validate_decision(
                        fb_decision,
                        narrative_st,
                        llm_scene,
                        priority=priority,
                    )
                    self._log.info(
                        "think llm_noop_fallback player=%s intent=%s type=%s",
                        player_id, fallback_speech.intent, validated.get("type"),
                    )
                    decision = fb_decision
            if self._inflight.is_cancelled(player_id, npc_id, generation):
                return

            self._log.info(
                "think llm player=%s raw=%s val=%s reject=%s hints=%s",
                player_id, decision.get("type"), validated.get("type"), reject or "-",
                len(llm_scene.get("rule_hints") or []),
            )
            self._narrative.record_outcome(
                player_id, npc_id,
                decision=validated, scene_info=scene_info,
                rejected_reason=reject,
                intent=str(decision.get("intent", "")),
            )
            mem_tag = "[NPC自主/llm]"
            polish_intent = str(decision.get("intent", ""))
            skip_polish = not llm_polish_enabled
            polish_template = None
            if fallback_speech is not None and fallback_speech.mode == "fallback":
                mem_tag = f"[NPC自主/{fallback_speech.intent}]"
                polish_intent = fallback_speech.intent
                polish_template = fallback_speech.reply
            yield from self._yield_validated_decision(
                validated,
                player_id=player_id,
                npc_id=npc_id,
                npc_name=npc_name,
                scene_info=scene_info,
                generation=generation,
                memory_tag=mem_tag,
                polish_intent=polish_intent,
                polish_template=polish_template,
                priority=priority,
                skip_polish=skip_polish,
                quick=priority >= 1,
            )

        except Exception as exc:  # noqa: BLE001
            if not self._inflight.is_cancelled(player_id, npc_id, generation):
                self._log.info("think error player=%s err=%s", player_id, exc)
                yield from self._yield_error(exc)
        finally:
            self._inflight.end(player_id, npc_id, generation)

    @staticmethod
    def _allowed_intents_for_trigger(
        trigger: str,
        st: NarrativeState,
        scene_info: dict[str, Any] | None = None,
    ) -> list[str]:
        scene_info = scene_info or {}
        assault_ok = bool(scene_info.get("assault_recommended"))
        if assault_ok and trigger in ("periodic", "social", "scene_change"):
            return ["dialogue", "command", "noop"]
        if trigger == "social":
            return ["dialogue", "command", "noop"] if assault_ok else ["dialogue", "noop"]
        if trigger == "periodic":
            return ["dialogue", "command", "noop"] if assault_ok else ["dialogue", "noop"]
        if trigger == "scene_change":
            return ["dialogue", "command", "noop"]
        if st.combat_mood == "critical":
            return ["dialogue", "command", "noop"]
        return ["dialogue", "command", "noop"]

    def _stream_dialogue(
        self,
        *,
        messages: list[dict[str, str]],
        player_id: str,
        npc_id: str,
        npc_name: str,
        scene_info: dict[str, Any],
        player_message: str,
        generation: int,
        memory_source: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        combat: bool = False,
    ) -> Iterator[str]:
        full_reply_parts: list[str] = []
        batch: list[str] = []

        for delta in chat_completion_stream(
            messages, max_tokens=max_tokens, temperature=temperature
        ):
            if self._inflight.is_cancelled(player_id, npc_id, generation):
                return
            full_reply_parts.append(delta)
            batch.append(delta)
            if sum(len(s) for s in batch) >= _DELTA_BATCH_CHARS:
                yield _event("delta", {"text": "".join(batch)})
                batch = []

        if batch:
            yield _event("delta", {"text": "".join(batch)})

        raw_reply = "".join(full_reply_parts).strip()
        cleaned, emo_from_tag = strip_emotion_tag(raw_reply)
        emotion = emo_from_tag if emo_from_tag in _VALID_EMOTIONS else "neutral"
        if emo_from_tag is None:
            m = _EMOTION_RE.search(raw_reply)
            if m:
                candidate = m.group(1).lower()
                emotion = candidate if candidate in _VALID_EMOTIONS else "neutral"
                cleaned = _EMOTION_RE.sub("", raw_reply).strip()

        recent = self._narrative.get(player_id, npc_id).recent_reply_texts
        full_reply = anti_ai_postprocess(
            cleaned or "收到，我会继续和你协同。",
            combat=combat,
            max_chars=48 if combat else 160,
            recent_texts=recent,
        ) or "收到，我会继续和你协同。"

        action = ChatAction(action_type="dialogue", dialogue=full_reply, emotion=emotion)
        yield _event("done", {"action": action.model_dump(exclude_none=True)})

        self._narrative.note_chat_reply(player_id, npc_id, full_reply)
        self._schedule_memory_write(
            player_id=player_id,
            npc_id=npc_id,
            scene_info=scene_info,
            player_message=player_message,
            reply=full_reply,
            memory_source=memory_source,
        )

    @staticmethod
    def _reflex_to_decision(reflex: ReflexDecision) -> dict[str, Any]:
        if reflex.type == "noop":
            return {"type": "noop"}
        if reflex.type == "command":
            return {
                "type": "command",
                "stance": reflex.stance,
                "reply": reflex.reply or "收到。",
            }
        return {"type": "dialogue", "reply": reflex.reply or ""}

    def _yield_validated_decision(
        self,
        decision: dict[str, Any],
        *,
        player_id: str,
        npc_id: str,
        scene_info: dict[str, Any],
        generation: int,
        memory_tag: str,
        npc_name: str = "",
        polish_template: str | None = None,
        polish_intent: str = "",
        priority: int = 3,
        skip_polish: bool = False,
        quick: bool | None = None,
    ) -> Iterator[str]:
        if self._inflight.is_cancelled(player_id, npc_id, generation):
            return

        polish_quick = quick if quick is not None else priority >= 2
        intent_tag = polish_intent or str(decision.get("intent", ""))

        decision_type = decision.get("type")
        if decision_type == "noop":
            yield _event("noop", {})
            return

        if decision_type == "command":
            stance = decision["stance"]
            ally_hp = int(scene_info.get("ally_hp", 100))
            if stance == "assault" and ally_hp <= 0:
                yield _event("noop", {})
                return
            current = str(scene_info.get("ally_stance", ""))
            reply = str(decision.get("reply", "")).strip()
            if not reply:
                reply = command_reply_from_scene(
                    stance, scene_info, same_stance=(stance == current)
                )
            reply = anti_ai_postprocess(reply, combat=True, max_chars=48) or reply
            yield _event("command", {
                "stance": stance,
                "reply": reply,
                "stance_changed": stance != current,
                "intent": intent_tag,
            })
            if reply:
                self._schedule_memory_write(
                    player_id=player_id,
                    npc_id=npc_id,
                    scene_info=scene_info,
                    player_message=memory_tag,
                    reply=reply,
                    memory_source="autonomous",
                )
            if not skip_polish:
                yield from self._maybe_polish(
                    polish_template or reply,
                    player_id=player_id,
                    npc_id=npc_id,
                    npc_name=npc_name,
                    scene_info=scene_info,
                    generation=generation,
                    intent=intent_tag,
                    skip=False,
                    quick=polish_quick,
                )
            return

        if decision_type == "dialogue":
            reply = str(decision.get("reply", "")).strip()
            if not reply:
                yield _event("noop", {})
                return
            reply = anti_ai_postprocess(
                reply,
                combat=is_combat_scene(scene_info),
                max_chars=48,
                recent_texts=self._narrative.get(player_id, npc_id).recent_reply_texts,
            ) or reply
            emotion = "worried" if scene_info.get("ally_in_danger") else "focused"
            action = ChatAction(action_type="dialogue", dialogue=reply, emotion=emotion)
            yield _event("done", {
                "action": action.model_dump(exclude_none=True),
                "intent": intent_tag,
            })
            self._schedule_memory_write(
                player_id=player_id,
                npc_id=npc_id,
                scene_info=scene_info,
                player_message=memory_tag,
                reply=reply,
                memory_source="autonomous",
            )
            if not skip_polish:
                yield from self._maybe_polish(
                    polish_template or reply,
                    player_id=player_id,
                    npc_id=npc_id,
                    npc_name=npc_name,
                    scene_info=scene_info,
                    generation=generation,
                    intent=intent_tag,
                    skip=False,
                    quick=polish_quick,
                )
            return

    def _maybe_polish(
        self,
        template: str,
        *,
        player_id: str,
        npc_id: str,
        npc_name: str,
        scene_info: dict[str, Any],
        generation: int,
        intent: str,
        skip: bool = False,
        quick: bool = False,
    ) -> Iterator[str]:
        if skip or not template or not npc_name:
            return
        st = self._narrative.get(player_id, npc_id)
        fut = self._pool.submit(
            polish_line,
            template=template,
            npc_name=npc_name,
            scene_info=scene_info,
            recent_texts=st.recent_reply_texts,
            intent=intent,
        )
        timeout = 0.35 if quick else 0.9
        try:
            polished = fut.result(timeout=timeout)
        except Exception:
            return
        if not polished or self._inflight.is_cancelled(player_id, npc_id, generation):
            return
        polished = anti_ai_postprocess(
            polished, combat=is_combat_scene(scene_info), max_chars=48
        ) or polished
        yield _event("polish", {"dialogue": polished, "emotion": "focused"})

    def _schedule_memory_write(
        self,
        *,
        player_id: str,
        npc_id: str,
        scene_info: dict[str, Any],
        player_message: str,
        reply: str,
        memory_source: str,
        skip_long_term: bool = False,
        run_id: str = "",
    ) -> None:
        min_chars = int(self._cfg.get("memory", {}).get("min_store_chars", 6))
        rid = (run_id or str(scene_info.get("run_id") or "")).strip()

        def _write_memory() -> None:
            if memory_source == "autonomous":
                # 自主台词只进 autonomy_log，不污染对话线程
                self._short_term.add_autonomy(
                    player_id, npc_id, reply, intent=player_message
                )
                # 自主默认不写 long-term，或仅极短 daily 摘要；避免冲掉玩家对话记忆
                return
            self._short_term.add_chat(player_id, npc_id, "user", player_message)
            self._short_term.add_chat(player_id, npc_id, "assistant", reply)
            if skip_long_term or len(reply) < min_chars:
                return
            try:
                tier, text = classify_dialogue_memory(
                    player_message=player_message,
                    npc_reply=reply,
                    scene_info=scene_info,
                )
                self._memory.add_dialogue_memory(
                    player_id=player_id,
                    npc_id=npc_id,
                    dialogue_tier=tier,
                    text=text,
                    scene_info=scene_info,
                    run_id=rid,
                )
            except Exception:  # noqa: BLE001
                pass

        threading.Thread(target=_write_memory, daemon=True).start()

    def _yield_error(self, exc: Exception) -> Iterator[str]:
        fallback = ChatAction(
            action_type="dialogue",
            dialogue="本地 NPC 服务暂时繁忙，我会继续跟随你行动。",
            emotion="neutral",
        )
        yield _event("error", {
            "message": str(exc),
            "fallback": fallback.model_dump(exclude_none=True),
        })


def _event(event_type: str, data: dict[str, Any]) -> str:
    return json.dumps({"type": event_type, **data}, ensure_ascii=False) + "\n"
