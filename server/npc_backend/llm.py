from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from server.npc_backend.config import load_config
from server.npc_backend.prompts import build_memory_classify_messages
from server.npc_backend.salient import command_reply_from_scene

# 支持的姿态集合（前端 game.js 中的 state.ally.stance 枚举值）
VALID_STANCES = {"guard", "assault"}

# 姿态 → NPC 确认短句（符合话痨嘴臭人设）
_STANCE_REPLIES: dict[str, str] = {
    "guard":  "行，收拢了，你别乱跑，我贴着你。",
    "assault": "好嘞，我去前面撕，你别拖后腿。",
}

_INTENT_SYSTEM_PROMPT = (
    "你是战术指令解析器，只输出 JSON，格式二选一：\n"
    '{"type":"dialogue"}\n'
    '{"type":"command","stance":"guard|assault","reply":"NPC一句确认话"}\n\n'
    "判断规则：\n"
    "1. command：玩家明确要求改变 NPC 行动模式，"
    "如[贴着我/守护/别乱跑/回来]->guard，[上去打/突击/压制/冲]->assault。\n"
    "2. dialogue：情绪交流、世界观追问、模糊意图、战斗评论 → 一律 dialogue。\n"
    "3. reply 要符合嘴臭话痨风格，简短，1句话。\n"
    "4. 不确定时默认 dialogue，宁可多对话不乱改姿态。"
)


def _client() -> OpenAI:
    cfg = load_config().get("llm", {})
    return OpenAI(
        api_key=cfg.get("api_key") or "dummy",
        base_url=cfg.get("base_url"),
    )


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout_s: int | None = None,
) -> str:
    cfg = load_config().get("llm", {})
    use_model = model or cfg.get("model", "deepseek-chat")
    kwargs: dict[str, Any] = {
        "model": use_model,
        "messages": messages,
        "temperature": float(cfg.get("temperature", 0.3) if temperature is None else temperature),
        "timeout": int(cfg.get("timeout_s", 60) if timeout_s is None else timeout_s),
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    resp = _client().chat.completions.create(**kwargs)
    choice = resp.choices[0] if resp.choices else None
    if not choice or not choice.message:
        return ""
    return (choice.message.content or "").strip()


def chat_completion_stream(
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> Iterator[str]:
    """逐 token yield 文本 delta，不处理业务状态和记忆。"""
    cfg = load_config().get("llm", {})
    kwargs: dict[str, Any] = {
        "model": cfg.get("model", "deepseek-chat"),
        "messages": messages,
        "temperature": float(cfg.get("temperature", 0.3) if temperature is None else temperature),
        "timeout": int(cfg.get("timeout_s", 60)),
        "stream": True,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    stream = _client().chat.completions.create(**kwargs)
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def autonomous_decide(
    *,
    npc_name: str,
    scene_info: dict[str, Any],
    world_chunks: list[str],
    persona_chunks: list[str],
    dialogue_daily_chunks: list[str],
    dialogue_important_chunks: list[str],
    short_term_history: list[dict[str, Any]],
    narrative_context: dict[str, str] | None = None,
    allowed_intents: list[str] | None = None,
    trigger: str = "periodic",
    chat_thread: list[dict[str, str]] | None = None,
    run_event_chunks: list[str] | None = None,
    reflection_chunks: list[str] | None = None,
    working_block: str = "",
) -> dict[str, Any]:
    """
    自主思考：根据局面决定 noop / command / dialogue。

    返回：
      - {"type": "noop"}
      - {"type": "command", "stance": "guard|assault", "reply": "..."}
      - {"type": "dialogue", "reply": "..."}
    """
    from server.npc_backend.prompts import build_autonomous_decide_messages

    messages = build_autonomous_decide_messages(
        npc_name=npc_name,
        scene_info=scene_info,
        world_chunks=world_chunks,
        persona_chunks=persona_chunks,
        dialogue_daily_chunks=dialogue_daily_chunks,
        dialogue_important_chunks=dialogue_important_chunks,
        short_term_history=short_term_history,
        narrative_context=narrative_context,
        allowed_intents=allowed_intents,
        trigger=trigger or str(scene_info.get("trigger", "periodic")),
        chat_thread=chat_thread,
        run_event_chunks=run_event_chunks,
        reflection_chunks=reflection_chunks,
        working_block=working_block,
    )
    cfg = load_config().get("llm", {})
    decide_model = cfg.get("decide_model") or cfg.get("model", "deepseek-chat")
    try:
        raw = chat_completion(messages, model=decide_model, max_tokens=120, timeout_s=25)
        stripped = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data: dict[str, Any] = json.loads(stripped)
        decision_type = str(data.get("type", "")).strip()
        if decision_type == "noop":
            return {"type": "noop"}
        if decision_type == "command":
            stance = str(data.get("stance", "")).strip()
            if stance not in VALID_STANCES:
                raise ValueError(f"invalid stance: {stance}")
            reply = str(data.get("reply", _STANCE_REPLIES.get(stance, "收到。"))).strip()
            return {"type": "command", "stance": stance, "reply": reply}
        if decision_type == "dialogue":
            reply = str(data.get("reply", "")).strip()
            if reply:
                return {"type": "dialogue", "reply": reply}
    except Exception:
        pass
    return {"type": "noop"}


_GUARD_HINTS = ("守护我", "跟着我", "贴着我", "别乱跑", "回来", "护着我", "守住", "守护")
_ASSAULT_HINTS = ("突击", "冲上去", "开路", "压制", "上去打", "前锋", "进攻", "去突击")
_INFO_HINTS = ("顺序", "怎么点", "报一下", "哪根柱")


def fast_command_intent(
    message: str,
    scene_info: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """短句战术指令快路径：仅姿态；技能不走对话。"""
    text = message.strip()
    if not text or len(text) > 28:
        return None
    scene_info = scene_info or {}
    current = str(scene_info.get("ally_stance", "") or "")

    # 判词报点：规则真值，禁止幻觉
    if scene_info.get("info_active") and any(h in text for h in _INFO_HINTS):
        report = str(scene_info.get("info_report") or "").strip()
        if report:
            return {
                "type": "dialogue",
                "reply": f"听好了——{report}",
            }

    for hint in _GUARD_HINTS:
        if hint in text:
            same = current == "guard"
            return {
                "type": "command",
                "stance": "guard",
                "reply": command_reply_from_scene("guard", scene_info, same_stance=same),
            }
    for hint in _ASSAULT_HINTS:
        if hint in text:
            same = current == "assault"
            return {
                "type": "command",
                "stance": "assault",
                "reply": command_reply_from_scene("assault", scene_info, same_stance=same),
            }
    return None


def classify_intent(
    *,
    message: str,
    scene_info: dict[str, Any],
    npc_name: str,
) -> dict[str, Any]:
    """
    判断玩家输入是"普通对话"还是"战术指令"。

    返回：
      - {"type": "dialogue"}
      - {"type": "command", "stance": "guard|assault", "reply": "..."}
    """
    from server.npc_backend.prompts import build_intent_classify_prompt

    user_prompt = build_intent_classify_prompt(
        npc_name=npc_name,
        message=message,
        scene_info=scene_info,
    )
    try:
        raw = chat_completion(
            [
                {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=80,
            timeout_s=12,
            temperature=0.1,
        )
        stripped = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data: dict[str, Any] = json.loads(stripped)
        if str(data.get("type", "")).strip() == "command":
            stance = str(data.get("stance", "")).strip()
            if stance not in VALID_STANCES:
                raise ValueError(f"invalid stance: {stance}")
            current = str(scene_info.get("ally_stance", "") or "")
            same = current == stance
            reply = str(
                data.get("reply")
                or command_reply_from_scene(stance, scene_info, same_stance=same)
                or _STANCE_REPLIES.get(stance, "收到。")
            ).strip()
            # 若模型确认语太空，用场面模板
            if len(reply) < 4:
                reply = command_reply_from_scene(stance, scene_info, same_stance=same)
            return {"type": "command", "stance": stance, "reply": reply}
    except Exception:
        pass
    return {"type": "dialogue"}


def generate_no_hp_reply(*, npc_name: str) -> str:
    """灵核失稳时拒绝突击：优先硬编码，避免额外 LLM 延迟。"""
    _ = npc_name
    return "灵核失稳了，冲不动，别催。"


def classify_dialogue_memory(
    *,
    player_message: str,
    npc_reply: str,
    scene_info: dict[str, Any],
) -> tuple[str, str]:
    raw_text = f"玩家说：{player_message}；NPC 回复：{npc_reply}"
    messages = build_memory_classify_messages(
        player_message=player_message,
        npc_reply=npc_reply,
        scene_info=scene_info,
    )
    try:
        data: dict[str, Any] = json.loads(chat_completion(messages))
        tier = str(data.get("dialogue_tier", "")).strip()
        processed = str(data.get("processed_text", "")).strip()
        if tier not in {"daily", "important"} or not processed:
            raise ValueError("invalid memory classify result")
        if tier == "important":
            return "important", raw_text
        return "daily", processed
    except Exception:
        return "important", raw_text

