"""工作记忆与承诺：不进向量库，进程内、按 player+npc 存放。"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Commitment:
    limb_id: str = "guard_follow"
    stance: str = "guard"
    source: str = "default"  # player | auto | reflex | default
    plan_id: str = ""
    until_ts: float = 0.0  # monotonic deadline; 0 = no ttl
    reason: str = ""

    def active(self, now: float | None = None) -> bool:
        if self.until_ts <= 0:
            return True
        return (now if now is not None else time.monotonic()) < self.until_ts


@dataclass
class WorkingState:
    run_id: str = ""
    commitment: Commitment = field(default_factory=Commitment)
    last_player_order: str = ""
    last_scene_summary: str = ""
    updated_at: float = field(default_factory=time.monotonic)


class WorkingMemoryStore:
    def __init__(self) -> None:
        self._states: dict[str, WorkingState] = {}

    def _key(self, player_id: str, npc_id: str) -> str:
        return f"{player_id}::{npc_id}"

    def get(self, player_id: str, npc_id: str) -> WorkingState:
        k = self._key(player_id, npc_id)
        if k not in self._states:
            self._states[k] = WorkingState()
        return self._states[k]

    def set_run_id(self, player_id: str, npc_id: str, run_id: str) -> None:
        st = self.get(player_id, npc_id)
        rid = (run_id or "").strip()
        if rid and st.run_id and st.run_id != rid:
            # 新局：清空 commitment 到默认
            st.commitment = Commitment()
            st.last_player_order = ""
            st.last_scene_summary = ""
        if rid:
            st.run_id = rid
        st.updated_at = time.monotonic()

    def set_commitment(
        self,
        player_id: str,
        npc_id: str,
        *,
        limb_id: str = "",
        stance: str = "",
        source: str = "auto",
        ttl_sec: float = 12.0,
        reason: str = "",
        plan_id: str = "",
    ) -> Commitment:
        st = self.get(player_id, npc_id)
        now = time.monotonic()
        limb = limb_id or (
            "assault_skirmish" if stance == "assault" else "guard_follow"
        )
        st_stance = stance or (
            "assault" if "assault" in limb else "guard"
        )
        st.commitment = Commitment(
            limb_id=limb,
            stance=st_stance,
            source=source,
            plan_id=plan_id or f"{source}-{int(now)}",
            until_ts=now + ttl_sec if ttl_sec > 0 else 0.0,
            reason=reason,
        )
        st.updated_at = now
        return st.commitment

    def note_player_order(self, player_id: str, npc_id: str, order: str) -> None:
        st = self.get(player_id, npc_id)
        st.last_player_order = (order or "").strip()[:160]
        st.updated_at = time.monotonic()

    def note_scene_summary(self, player_id: str, npc_id: str, summary: str) -> None:
        st = self.get(player_id, npc_id)
        st.last_scene_summary = (summary or "").strip()[:400]
        st.updated_at = time.monotonic()

    def prompt_block(self, player_id: str, npc_id: str) -> str:
        st = self.get(player_id, npc_id)
        c = st.commitment
        now = time.monotonic()
        remain = ""
        if c.until_ts > 0:
            left = max(0, int(c.until_ts - now))
            remain = f"，剩余约{left}s" if c.active(now) else "（已到期）"
        lines = [
            f"run_id={st.run_id or '未知'}",
            f"当前承诺: limb={c.limb_id} stance={c.stance} 来源={c.source}{remain}",
        ]
        if c.reason:
            lines.append(f"承诺原因: {c.reason}")
        if st.last_player_order:
            lines.append(f"最近玩家指令: {st.last_player_order}")
        if st.last_scene_summary:
            lines.append(f"场面摘要: {st.last_scene_summary}")
        return "\n".join(lines)

    def to_public(self, player_id: str, npc_id: str) -> dict[str, Any]:
        st = self.get(player_id, npc_id)
        c = st.commitment
        return {
            "run_id": st.run_id,
            "commitment": {
                "limb_id": c.limb_id,
                "stance": c.stance,
                "source": c.source,
                "active": c.active(),
                "reason": c.reason,
            },
            "last_player_order": st.last_player_order,
        }
