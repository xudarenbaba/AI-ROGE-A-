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
    generate_no_hp_reply,
)
from server.npc_backend.memory import MemoryStore
from server.npc_backend.narrative import NarrativeState, NarrativeStore, scene_hash
from server.npc_backend.polish import polish_line
from server.npc_backend.prompts import build_messages
from server.npc_backend.reflex import ReflexDecision, reflex_decide
from server.npc_backend.triggers import (
    TriggerSpeech,
    assault_recommended,
    evaluate_p0,
    evaluate_p1,
    evaluate_p2,
    evaluate_p3,
    trigger_to_decision,
)
from server.npc_backend.schemas import ChatAction
from server.npc_backend.short_term import ShortTermMemory

_EMOTION_RE = re.compile(r"<emotion>([\w]+)</emotion>\s*$", re.IGNORECASE)
_VALID_EMOTIONS = {
    "neutral", "focused", "annoyed", "worried", "happy", "tense", "sarcastic"
}

# delta 批量缓冲阈值（字符数）：积累到此长度或收到 LLM 流结束后统一 flush
_DELTA_BATCH_CHARS = 8


class NpcConversationEngine:
    def __init__(self) -> None:
        self._cfg = load_config()
        self._memory = MemoryStore()
        self._short_term = ShortTermMemory()
        worker_threads = int(self._cfg.get("concurrency", {}).get("worker_threads", 8))
        self._pool = ThreadPoolExecutor(
            max_workers=worker_threads,
            thread_name_prefix="npc-worker",
        )
        self._inflight = InflightTracker()
        self._narrative = NarrativeStore(self._cfg.get("npc_autonomy", {}))
        self._log = npc_logger()

    def stream_chat(self, payload: dict[str, Any]) -> Iterator[str]:
        """
        统一流式入口，逐行 yield NDJSON 字符串。

        事件类型：
          meta    - 立即返回，携带 npc_id
          command - 战术指令，携带 stance + reply，前端切换姿态后结束
          delta   - 对话 token 批次（积累 _DELTA_BATCH_CHARS 字符后 flush）
          done    - 对话完整结束，携带最终 ChatAction
          error   - 出错，携带 fallback

        并行优化：收到请求后立即同时启动：
          - 意图分类 LLM（非流式）
          - 短期记忆 + 长期记忆检索
        若分类结果为 command，直接 yield command 事件，记忆检索结果丢弃。
        若分类结果为 dialogue，记忆检索结果（大概率已完成）直接用于构建 prompt。
        """
        player_id: str = payload.get("player_id", "")
        npc_id: str = payload.get("npc_id", "")
        npc_name: str = payload.get("npc_name") or npc_id
        message: str = payload.get("message", "")
        scene_info: dict[str, Any] = payload.get("scene_info") or {}

        self._narrative.on_player_message(player_id, npc_id)
        narrative_st = self._narrative.get(player_id, npc_id)
        self._narrative.update_mood(narrative_st, scene_info)
        _cancel, generation = self._inflight.begin(player_id, npc_id, "chat")
        yield _event("meta", {
            "npc_id": npc_id,
            "combat_mood": narrative_st.combat_mood,
        })

        try:
            narrative_ctx = self._narrative.context_for_prompt(player_id, npc_id)
            query = f"scene={scene_info}\nmessage={message}"

            fut_intent: Future[dict[str, Any]] = self._pool.submit(
                classify_intent,
                message=message,
                scene_info=scene_info,
                npc_name=npc_name,
            )
            fut_short: Future[list[dict[str, Any]]] = self._pool.submit(
                self._short_term.get_recent, player_id, npc_id
            )
            fut_long: Future[dict[str, list[str]]] = self._pool.submit(
                self._memory.search_context,
                query,
                player_id,
                npc_id,
                self._pool,
            )

            intent = fut_intent.result()

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
                    yield _event("done", {"action": action.model_dump(exclude_none=True)})
                    return
                yield _event("command", {
                    "stance": stance,
                    "reply": intent["reply"],
                })
                return

            short_term_history: list[dict[str, Any]] = fut_short.result()
            long_term_ctx: dict[str, list[str]] = fut_long.result()

            if self._inflight.is_cancelled(player_id, npc_id, generation):
                return

            messages = build_messages(
                npc_name=npc_name,
                player_message=message,
                scene_info=scene_info,
                world_chunks=long_term_ctx.get("world_chunks", []),
                persona_chunks=long_term_ctx.get("persona_chunks", []),
                dialogue_daily_chunks=long_term_ctx.get("dialogue_daily_chunks", []),
                dialogue_important_chunks=long_term_ctx.get("dialogue_important_chunks", []),
                short_term_history=short_term_history,
                narrative_context=narrative_ctx,
            )

            yield from self._stream_dialogue(
                messages=messages,
                player_id=player_id,
                npc_id=npc_id,
                npc_name=npc_name,
                scene_info=scene_info,
                player_message=message,
                generation=generation,
                memory_source="player",
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

        if not autonomy_cfg.get("enabled", True):
            yield _event("noop", {"reason": "disabled"})
            return

        narrative_st = self._narrative.get(player_id, npc_id)
        self._narrative.update_mood(narrative_st, scene_info)
        opening = trigger_reason in ("floor_enter", "opening") or "floor_enter" in (
            scene_info.get("recent_events") or []
        )
        assault_ok = assault_recommended(
            scene_info, autonomy_cfg, opening=opening,
        )
        scene_info = {
            **scene_info,
            "trigger": trigger,
            "trigger_reason": trigger_reason,
            "priority": priority,
            "scene_hash": scene_hash(scene_info),
            "scene_dramatic_change": self._narrative.scene_dramatic_change(
                narrative_st, scene_info
            ),
            "prev_enemy_count": narrative_st.prev_enemy_count,
            "assault_recommended": assault_ok,
        }

        rule_speech: TriggerSpeech | None = None
        if priority <= 0:
            rule_speech = evaluate_p0(scene_info, autonomy_cfg)
        elif priority == 1:
            rule_speech = evaluate_p1(scene_info, narrative_st, autonomy_cfg)
        elif priority == 2:
            rule_speech = evaluate_p2(scene_info, narrative_st, autonomy_cfg)
        elif priority >= 3:
            rule_speech = evaluate_p3(scene_info, narrative_st, autonomy_cfg)

        if rule_speech is None:
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
            if rule_speech is not None:
                decision = trigger_to_decision(rule_speech)
                validated, reject = self._narrative.validate_decision(
                    decision,
                    narrative_st,
                    scene_info,
                    skip_dedup=rule_speech.skip_dedup,
                    priority=priority,
                )
                self._log.info(
                    "think rule_speech player=%s intent=%s type=%s reject=%s",
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
                    polish_template=rule_speech.reply if rule_speech.allow_polish else None,
                    polish_intent=rule_speech.intent,
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
                    )
                    return

            allowed = self._allowed_intents_for_trigger(
                trigger, narrative_st, scene_info,
            )
            situation = (
                f"第{scene_info.get('floor', '?')}层"
                f" 玩家HP={scene_info.get('player_hp', '?')}"
                f" 友军HP={scene_info.get('ally_hp', '?')}"
                f" 敌={scene_info.get('enemy_count', '?')}"
            )
            query = f"scene={scene_info}\ntrigger={trigger}\nsituation={situation}"

            fut_short = self._pool.submit(self._short_term.get_recent, player_id, npc_id)
            fut_long = self._pool.submit(
                self._memory.search_context, query, player_id, npc_id, self._pool
            )
            short_term_history = fut_short.result()
            long_term_ctx = fut_long.result()

            if self._inflight.is_cancelled(player_id, npc_id, generation):
                return

            narrative_ctx = self._narrative.context_for_prompt(player_id, npc_id)
            decision = self._pool.submit(
                autonomous_decide,
                npc_name=npc_name,
                scene_info=scene_info,
                world_chunks=long_term_ctx.get("world_chunks", []),
                persona_chunks=long_term_ctx.get("persona_chunks", []),
                dialogue_daily_chunks=long_term_ctx.get("dialogue_daily_chunks", []),
                dialogue_important_chunks=long_term_ctx.get("dialogue_important_chunks", []),
                short_term_history=short_term_history,
                narrative_context=narrative_ctx,
                allowed_intents=allowed,
                trigger=trigger,
            ).result()

            validated, reject = self._narrative.validate_decision(
                decision, narrative_st, scene_info, priority=priority,
            )
            if validated.get("type") == "noop" and trigger in ("periodic", "social"):
                fallback = evaluate_p3(
                    scene_info, narrative_st, autonomy_cfg, force_banter=True,
                )
                if fallback is not None:
                    fb_decision = trigger_to_decision(fallback)
                    validated, reject = self._narrative.validate_decision(
                        fb_decision,
                        narrative_st,
                        scene_info,
                        priority=priority,
                    )
                    self._log.info(
                        "think llm_noop_fallback player=%s intent=%s type=%s",
                        player_id, fallback.intent, validated.get("type"),
                    )
                    decision = fb_decision
                    rule_speech = fallback
            if self._inflight.is_cancelled(player_id, npc_id, generation):
                return

            self._log.info(
                "think llm player=%s raw=%s val=%s reject=%s",
                player_id, decision.get("type"), validated.get("type"), reject or "-",
            )
            self._narrative.record_outcome(
                player_id, npc_id,
                decision=validated, scene_info=scene_info,
                rejected_reason=reject,
                intent=str(decision.get("intent", "")),
            )
            llm_polish = (
                rule_speech.reply
                if rule_speech is not None and rule_speech.allow_polish
                else None
            )
            yield from self._yield_validated_decision(
                validated,
                player_id=player_id,
                npc_id=npc_id,
                npc_name=npc_name,
                scene_info=scene_info,
                generation=generation,
                memory_tag="[NPC自主/llm]",
                polish_template=llm_polish,
                polish_intent=rule_speech.intent if rule_speech else "",
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
    ) -> Iterator[str]:
        full_reply_parts: list[str] = []
        batch: list[str] = []

        for delta in chat_completion_stream(messages):
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
        emotion = "neutral"
        m = _EMOTION_RE.search(raw_reply)
        if m:
            candidate = m.group(1).lower()
            emotion = candidate if candidate in _VALID_EMOTIONS else "neutral"
            raw_reply = _EMOTION_RE.sub("", raw_reply).strip()

        full_reply = raw_reply or "收到，我会继续和你协同。"
        action = ChatAction(action_type="dialogue", dialogue=full_reply, emotion=emotion)
        yield _event("done", {"action": action.model_dump(exclude_none=True)})

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
    ) -> Iterator[str]:
        if self._inflight.is_cancelled(player_id, npc_id, generation):
            return

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
            yield _event("command", {
                "stance": stance,
                "reply": decision.get("reply", "收到。"),
                "stance_changed": stance != current,
            })
            reply = str(decision.get("reply", "")).strip()
            if reply:
                self._schedule_memory_write(
                    player_id=player_id,
                    npc_id=npc_id,
                    scene_info=scene_info,
                    player_message=memory_tag,
                    reply=reply,
                    memory_source="autonomous",
                )
            yield from self._maybe_polish(
                polish_template or reply,
                player_id=player_id,
                npc_id=npc_id,
                npc_name=npc_name,
                scene_info=scene_info,
                generation=generation,
                intent=polish_intent or str(decision.get("intent", "")),
            )
            return

        if decision_type == "dialogue":
            reply = str(decision.get("reply", "")).strip()
            if not reply:
                yield _event("noop", {})
                return
            emotion = "worried" if scene_info.get("ally_in_danger") else "focused"
            action = ChatAction(action_type="dialogue", dialogue=reply, emotion=emotion)
            yield _event("done", {"action": action.model_dump(exclude_none=True)})
            self._schedule_memory_write(
                player_id=player_id,
                npc_id=npc_id,
                scene_info=scene_info,
                player_message=memory_tag,
                reply=reply,
                memory_source="autonomous",
            )
            yield from self._maybe_polish(
                polish_template or reply,
                player_id=player_id,
                npc_id=npc_id,
                npc_name=npc_name,
                scene_info=scene_info,
                generation=generation,
                intent=polish_intent or str(decision.get("intent", "")),
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
    ) -> Iterator[str]:
        if not template or not npc_name:
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
        try:
            polished = fut.result(timeout=0.9)
        except Exception:
            return
        if not polished or self._inflight.is_cancelled(player_id, npc_id, generation):
            return
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
    ) -> None:
        min_chars = int(self._cfg.get("memory", {}).get("min_store_chars", 6))

        def _write_memory() -> None:
            if memory_source == "autonomous":
                self._short_term.add_turn(
                    player_id, npc_id, "system", f"[自主思考] {player_message}"
                )
            else:
                self._short_term.add_turn(player_id, npc_id, "user", player_message)
            self._short_term.add_turn(player_id, npc_id, "assistant", reply)
            if len(reply) >= min_chars:
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