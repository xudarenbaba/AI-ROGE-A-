from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    player_id: str = Field(min_length=1)
    npc_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    scene_info: dict[str, Any] = Field(default_factory=dict)
    npc_name: str | None = None
    run_id: str = ""


class ThinkRequest(BaseModel):
    player_id: str = Field(min_length=1)
    npc_id: str = Field(min_length=1)
    scene_info: dict[str, Any] = Field(default_factory=dict)
    npc_name: str | None = None
    trigger: str = "periodic"
    priority: int = 3
    trigger_reason: str = ""
    run_id: str = ""


class RunEventRequest(BaseModel):
    player_id: str = Field(min_length=1)
    npc_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    tier: str = "major"
    source: str = "system"
    scene_info: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class ReflectionRequest(BaseModel):
    player_id: str = Field(min_length=1)
    npc_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_run_id: str = ""
    source: str = "system"


class ChatAction(BaseModel):
    action_type: str = "dialogue"
    dialogue: str
    emotion: str | None = None

