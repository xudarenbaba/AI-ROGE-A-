"""NPC API server — 只提供 AI 接口，不托管游戏静态文件。"""
from __future__ import annotations

import json
from typing import Any

from flask import Flask, Response, jsonify, request, stream_with_context
from flask_cors import CORS

from server.npc_backend.graph import NpcConversationEngine
from server.npc_backend.log_config import npc_logger, setup_npc_logging
from server.npc_backend.schemas import ChatRequest, ThinkRequest


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
            "npc_id":    req.npc_id,
            "npc_name":  req.npc_name,
            "message":   req.message,
            "scene_info": req.scene_info,
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
            "HTTP /api/npc/think player=%s trigger=%s p=%s reason=%s",
            req.player_id, req.trigger, req.priority, req.trigger_reason or "-",
        )

        payload = {
            "player_id": req.player_id,
            "npc_id": req.npc_id,
            "npc_name": req.npc_name,
            "scene_info": req.scene_info,
            "trigger": req.trigger,
            "priority": req.priority,
            "trigger_reason": req.trigger_reason,
        }

        return Response(
            stream_with_context(engine.stream_think(payload)),
            mimetype="application/x-ndjson",
        )

    return app


app = create_app()
