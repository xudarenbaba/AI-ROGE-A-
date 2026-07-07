from __future__ import annotations

import threading
from typing import Any


class InflightTracker:
    """跟踪每个 player:npc 会话的进行中请求，支持自主思考被玩家对话抢占。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _key(player_id: str, npc_id: str) -> str:
        return f"{player_id}:{npc_id}"

    def begin(self, player_id: str, npc_id: str, kind: str) -> tuple[threading.Event, int]:
        cancel_event = threading.Event()
        with self._lock:
            key = self._key(player_id, npc_id)
            existing = self._sessions.get(key)
            if kind == "chat" and existing and existing.get("kind") == "think":
                existing["cancel_event"].set()
            generation = (existing.get("generation", 0) + 1) if existing else 1
            self._sessions[key] = {
                "kind": kind,
                "cancel_event": cancel_event,
                "generation": generation,
            }
            return cancel_event, generation

    def is_cancelled(self, player_id: str, npc_id: str, generation: int) -> bool:
        with self._lock:
            sess = self._sessions.get(self._key(player_id, npc_id))
            if not sess:
                return True
            return sess["cancel_event"].is_set() or sess["generation"] != generation

    def end(self, player_id: str, npc_id: str, generation: int) -> None:
        with self._lock:
            key = self._key(player_id, npc_id)
            sess = self._sessions.get(key)
            if sess and sess.get("generation") == generation:
                del self._sessions[key]