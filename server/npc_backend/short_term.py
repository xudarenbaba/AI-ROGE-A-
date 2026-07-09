from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

from server.npc_backend.config import load_config


class ShortTermMemory:
    """短期记忆：玩家对话线程与自主发言日志分离。"""

    def __init__(self) -> None:
        cfg = load_config().get("memory", {})
        turns = int(cfg.get("short_term_turns", 10))
        self._max_chat = max(4, turns * 2)
        self._max_autonomy = int(cfg.get("autonomy_log_max", 8))
        self._chat: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self._max_chat)
        )
        self._autonomy: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self._max_autonomy)
        )

    @staticmethod
    def _key(player_id: str, npc_id: str) -> str:
        return f"{player_id}:{npc_id}"

    def add_turn(self, player_id: str, npc_id: str, role: str, content: str) -> None:
        """兼容旧接口：assistant/user 进对话线程；system 进自主日志。"""
        if not content.strip():
            return
        if role == "system" or content.strip().startswith("[自主") or content.strip().startswith("[NPC自主"):
            self.add_autonomy(player_id, npc_id, content)
            return
        mapped = "user" if role == "user" else "assistant"
        self.add_chat(player_id, npc_id, mapped, content)

    def add_chat(self, player_id: str, npc_id: str, role: str, content: str) -> None:
        if not content.strip():
            return
        if role not in {"user", "assistant"}:
            role = "assistant" if role != "user" else "user"
        self._chat[self._key(player_id, npc_id)].append(
            {
                "role": role,
                "content": content.strip(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def add_autonomy(self, player_id: str, npc_id: str, content: str, *, intent: str = "") -> None:
        text = content.strip()
        if not text:
            return
        # 去掉内部标签前缀，保留可读句子
        for prefix in ("[自主思考]", "[NPC自主]", "[NPC自主/llm]"):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                break
        if text.startswith("[") and "]" in text[:24]:
            # [NPC自主/xxx] rest
            text = text.split("]", 1)[-1].strip()
        self._autonomy[self._key(player_id, npc_id)].append(
            {
                "role": "autonomy",
                "content": text,
                "intent": intent,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_recent(self, player_id: str, npc_id: str) -> list[dict[str, Any]]:
        """兼容旧接口：对话线程 + 最近自主，扁平列表。"""
        chat = list(self._chat[self._key(player_id, npc_id)])
        auto = list(self._autonomy[self._key(player_id, npc_id)])
        # 旧调用方期望 role/content；自主用 system 标记
        out = list(chat)
        for item in auto[-3:]:
            out.append(
                {
                    "role": "system",
                    "content": f"[我刚自主说过] {item.get('content', '')}",
                    "timestamp": item.get("timestamp"),
                }
            )
        return out

    def get_chat_thread(
        self,
        player_id: str,
        npc_id: str,
        *,
        max_turns: int | None = None,
    ) -> list[dict[str, str]]:
        items = list(self._chat[self._key(player_id, npc_id)])
        if max_turns is not None:
            items = items[-(max_turns * 2) :]
        return [
            {"role": str(i.get("role", "user")), "content": str(i.get("content", ""))}
            for i in items
            if i.get("content")
        ]

    def get_autonomy_log(
        self,
        player_id: str,
        npc_id: str,
        *,
        max_items: int = 5,
    ) -> list[dict[str, Any]]:
        return list(self._autonomy[self._key(player_id, npc_id)])[-max_items:]

    def last_assistant_line(self, player_id: str, npc_id: str) -> str:
        for item in reversed(self._chat[self._key(player_id, npc_id)]):
            if item.get("role") == "assistant" and item.get("content"):
                return str(item["content"])
        auto = self._autonomy[self._key(player_id, npc_id)]
        if auto:
            return str(auto[-1].get("content") or "")
        return ""
