"""NPC API server — 只提供 AI 接口，不托管游戏静态文件。"""
from __future__ import annotations

import json
from typing import Any

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

from server.npc_backend.graph import NpcConversationEngine
from server.npc_backend.log_config import npc_logger, setup_npc_logging
from server.npc_backend.schemas import (
    ChatRequest,
    ReflectionRequest,
    RunEventRequest,
    ThinkRequest,
)


def create_app() -> Flask:
    setup_npc_logging()
    app = Flask(__name__, static_folder=None)
    CORS(app)
    engine = NpcConversationEngine()

    @app.get("/health")
    def health() -> tuple[Any, int]:
        return jsonify({"status": "ok"}), 200

    @app.post("/api/chat/stream")
    def api_chat_stream() -> Response:
        body = request.get_json(force=True, silent=True) or {}
        try:
            req = ChatRequest(
                player_id=str(body.get("player_id", "")).strip(),
                npc_id=str(body.get("npc_id", "")).strip(),
                message=str(body.get("message", "")).strip(),
                scene_info=body.get("scene_info") or {},
                npc_name=str(body.get("npc_name", "")).strip() or None,
                run_id=str(body.get("run_id", "")).strip(),
            )
        except Exception as e:  # noqa: BLE001
            return Response(
                json.dumps({"type": "error", "message": f"invalid_request: {e}"}, ensure_ascii=False) + "\n",
                mimetype="application/x-ndjson",
                status=400,
            )

        if not req.player_id or not req.npc_id or not req.message:
            return Response(
                json.dumps({"type": "error", "message": "player_id, npc_id and message are required"}, ensure_ascii=False) + "\n",
                mimetype="application/x-ndjson",
                status=400,
            )

        payload = {
            "player_id": req.player_id,
            "npc_id": req.npc_id,
            "npc_name": req.npc_name,
            "message": req.message,
            "scene_info": req.scene_info,
            "run_id": req.run_id,
        }

        return Response(
            stream_with_context(engine.stream_chat(payload)),
            mimetype="application/x-ndjson",
        )

    @app.post("/api/npc/think")
    def api_npc_think() -> Response:
        body = request.get_json(force=True, silent=True) or {}
        try:
            req = ThinkRequest(
                player_id=str(body.get("player_id", "")).strip(),
                npc_id=str(body.get("npc_id", "")).strip(),
                scene_info=body.get("scene_info") or {},
                npc_name=str(body.get("npc_name", "")).strip() or None,
                trigger=str(body.get("trigger", "periodic")).strip() or "periodic",
                priority=int(body.get("priority", 3)),
                trigger_reason=str(body.get("trigger_reason", "")).strip(),
                run_id=str(body.get("run_id", "")).strip(),
            )
        except Exception as e:  # noqa: BLE001
            return Response(
                json.dumps({"type": "error", "message": f"invalid_request: {e}"}, ensure_ascii=False) + "\n",
                mimetype="application/x-ndjson",
                status=400,
            )

        if not req.player_id or not req.npc_id:
            return Response(
                json.dumps({"type": "error", "message": "player_id and npc_id are required"}, ensure_ascii=False) + "\n",
                mimetype="application/x-ndjson",
                status=400,
            )

        npc_logger().info(
            "HTTP /api/npc/think player=%s trigger=%s p=%s reason=%s run=%s",
            req.player_id, req.trigger, req.priority, req.trigger_reason or "-",
            req.run_id or "-",
        )

        payload = {
            "player_id": req.player_id,
            "npc_id": req.npc_id,
            "npc_name": req.npc_name,
            "scene_info": req.scene_info,
            "trigger": req.trigger,
            "priority": req.priority,
            "trigger_reason": req.trigger_reason,
            "run_id": req.run_id,
        }

        return Response(
            stream_with_context(engine.stream_think(payload)),
            mimetype="application/x-ndjson",
        )

    @app.post("/api/memory/run_event")
    def api_memory_run_event() -> tuple[Any, int]:
        body = request.get_json(force=True, silent=True) or {}
        try:
            req = RunEventRequest(
                player_id=str(body.get("player_id", "")).strip(),
                npc_id=str(body.get("npc_id", "")).strip(),
                run_id=str(body.get("run_id", "")).strip(),
                text=str(body.get("text", "")).strip(),
                tier=str(body.get("tier", "major")).strip() or "major",
                source=str(body.get("source", "system")).strip() or "system",
                scene_info=body.get("scene_info") or {},
                tags=list(body.get("tags") or []),
            )
        except Exception as e:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"invalid_request: {e}"}), 400

        if not req.player_id or not req.npc_id or not req.run_id or not req.text:
            return jsonify({"ok": False, "error": "player_id, npc_id, run_id, text required"}), 400

        ok = engine.memory.add_run_event(
            player_id=req.player_id,
            npc_id=req.npc_id,
            run_id=req.run_id,
            text=req.text,
            tier=req.tier,
            source=req.source,
            scene_info=req.scene_info,
            tags=req.tags,
        )
        return jsonify({"ok": ok, "type": "run_event"}), 200 if ok else 200

    @app.post("/api/memory/reflection")
    def api_memory_reflection() -> tuple[Any, int]:
        body = request.get_json(force=True, silent=True) or {}
        try:
            req = ReflectionRequest(
                player_id=str(body.get("player_id", "")).strip(),
                npc_id=str(body.get("npc_id", "")).strip(),
                text=str(body.get("text", "")).strip(),
                source_run_id=str(body.get("source_run_id", "")).strip(),
                source=str(body.get("source", "system")).strip() or "system",
            )
        except Exception as e:  # noqa: BLE001
            return jsonify({"ok": False, "error": f"invalid_request: {e}"}), 400

        if not req.player_id or not req.npc_id or not req.text:
            return jsonify({"ok": False, "error": "player_id, npc_id, text required"}), 400

        ok = engine.memory.add_reflection(
            player_id=req.player_id,
            npc_id=req.npc_id,
            text=req.text,
            source_run_id=req.source_run_id,
            source=req.source,
        )
        return jsonify({"ok": ok, "type": "reflection"}), 200

    return app


app = create_app()
