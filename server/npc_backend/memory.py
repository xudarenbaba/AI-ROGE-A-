from __future__ import annotations

import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from server.npc_backend.config import load_config

_EMBED_MODEL: SentenceTransformer | None = None

# 统一 type（与方案 v3 对齐）
MEMORY_TYPES = frozenset(
    {"world", "persona", "dialogue", "run_event", "reflection"}
)
DIALOGUE_TIERS = frozenset({"daily", "important"})
RUN_EVENT_TIERS = frozenset({"major", "minor"})

# 查询 channel → 预算与是否限 run_id
CHANNEL_PRESETS: dict[str, dict[str, Any]] = {
    "chat": {
        "k_world": 2,
        "k_persona": 3,
        "k_daily": 3,
        "k_important": 5,
        "k_run_event": 3,
        "k_reflection": 1,
        "run_event_current_only": True,
    },
    "think_combat": {
        "k_world": 0,
        "k_persona": 1,
        "k_daily": 0,
        "k_important": 3,
        "k_run_event": 5,
        "k_reflection": 1,
        "run_event_current_only": True,
    },
    "think_safe": {
        "k_world": 1,
        "k_persona": 3,
        "k_daily": 3,
        "k_important": 3,
        "k_run_event": 2,
        "k_reflection": 1,
        "run_event_current_only": True,
    },
    "run_start": {
        "k_world": 0,
        "k_persona": 1,
        "k_daily": 0,
        "k_important": 3,
        "k_run_event": 0,  # 新局不灌上一局事件
        "k_reflection": 2,
        "run_event_current_only": True,
    },
    "default": {
        "k_world": 3,
        "k_persona": 3,
        "k_daily": 4,
        "k_important": 6,
        "k_run_event": 4,
        "k_reflection": 1,
        "run_event_current_only": True,
    },
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _get_embed_model() -> SentenceTransformer:
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL

    cfg = load_config()
    emb_cfg = cfg.get("embeddings", {})
    model_name = emb_cfg.get("model", "BAAI/bge-small-zh-v1.5")
    cache_dir = Path(emb_cfg.get("cache_dir", "models"))
    if not cache_dir.is_absolute():
        cache_dir = _project_root() / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_files_only = bool(emb_cfg.get("local_files_only", False))
    _EMBED_MODEL = SentenceTransformer(
        model_name,
        cache_folder=str(cache_dir),
        local_files_only=local_files_only,
    )
    return _EMBED_MODEL


def _embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_embed_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_memory_context() -> dict[str, list[str]]:
    return {
        "world_chunks": [],
        "persona_chunks": [],
        "dialogue_daily_chunks": [],
        "dialogue_important_chunks": [],
        "run_event_chunks": [],
        "reflection_chunks": [],
    }


def format_memory_prompt_blocks(ctx: dict[str, list[str]]) -> str:
    """统一分块标签；空块省略。"""
    sections: list[tuple[str, list[str]]] = [
        ("[角色设定]", ctx.get("persona_chunks") or []),
        ("[世界设定]", ctx.get("world_chunks") or []),
        ("[长期反思]", ctx.get("reflection_chunks") or []),
        ("[对话记忆-重要]", ctx.get("dialogue_important_chunks") or []),
        ("[对话记忆-日常]", ctx.get("dialogue_daily_chunks") or []),
        ("[本局事件]", ctx.get("run_event_chunks") or []),
    ]
    parts: list[str] = []
    for title, lines in sections:
        cleaned = [str(x).strip() for x in lines if str(x).strip()]
        if not cleaned:
            continue
        body = "\n".join(f"- {line}" for line in cleaned)
        parts.append(f"{title}\n{body}")
    return "\n\n".join(parts) if parts else ""


class MemoryStore:
    """
    统一记忆：type = world | persona | dialogue | run_event | reflection。
    本局事件靠 metadata.run_id 过滤；反思跨局。
    """

    def __init__(self) -> None:
        cfg = load_config()
        vs_cfg = cfg.get("vectorstore", {})
        persist_dir = Path(vs_cfg.get("persist_dir", "data/chroma"))
        if not persist_dir.is_absolute():
            persist_dir = _project_root() / persist_dir
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = vs_cfg.get("collection_name", "npc_memory")
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        memory_cfg = cfg.get("memory", {})
        self._k_world = int(memory_cfg.get("k_world", 3))
        self._k_persona = int(memory_cfg.get("k_persona", 3))
        self._k_daily = int(memory_cfg.get("k_dialogue_daily", 4))
        self._k_important = int(memory_cfg.get("k_dialogue_important", 6))
        self._k_run_event = int(memory_cfg.get("k_run_event", 5))
        self._k_reflection = int(memory_cfg.get("k_reflection", 2))
        self._min_store_chars = int(memory_cfg.get("min_store_chars", 6))

    def _collection(self):
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"description": "npc memory unified types"},
        )

    def _query_with_embedding(
        self, embedding: list[float], where: dict[str, Any], limit: int
    ) -> list[str]:
        if limit <= 0:
            return []
        coll = self._collection()
        try:
            result = coll.query(
                query_embeddings=[embedding],
                n_results=limit,
                where=where,
                include=["documents"],
            )
            documents = result.get("documents", [[]])[0] if result else []
            return [doc for doc in documents if doc]
        except Exception:
            return []

    def upsert_seed_memory(
        self,
        *,
        memory_type: str,
        npc_id: str,
        texts: list[str],
        extra_metadata: dict[str, Any] | None = None,
        replace_existing: bool = False,
    ) -> None:
        if not texts:
            return
        coll = self._collection()
        normalized = [text.strip() for text in texts if text.strip()]
        if not normalized:
            return
        if replace_existing:
            self.delete_seed_memory(memory_type=memory_type, npc_id=npc_id)
        ids = [
            f"{memory_type}:{npc_id}:{hashlib.sha1(text.encode('utf-8')).hexdigest()}"
            for text in normalized
        ]
        metadatas = [
            {
                "memory_type": memory_type,
                "npc_id": npc_id,
                "source": "seed",
                "created_at": _now_iso(),
                **(extra_metadata or {}),
            }
            for _ in normalized
        ]
        coll.upsert(
            ids=ids,
            embeddings=_embed_texts(normalized),
            documents=normalized,
            metadatas=metadatas,
        )

    def delete_seed_memory(self, *, memory_type: str, npc_id: str) -> int:
        coll = self._collection()
        try:
            existing = coll.get(
                where={"memory_type": memory_type, "npc_id": npc_id, "source": "seed"}
            )
            ids = existing.get("ids", []) if existing else []
            if ids:
                coll.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0

    def add_world_seed(self, texts: list[str]) -> None:
        self.upsert_seed_memory(
            memory_type="world",
            npc_id="global",
            texts=texts,
            extra_metadata={"scope": "global"},
            replace_existing=True,
        )

    def ensure_persona_seeded(self, npc_id: str, persona_lines: list[str]) -> None:
        self.upsert_seed_memory(
            memory_type="persona",
            npc_id=npc_id,
            texts=persona_lines,
            extra_metadata={"scope": "npc"},
            replace_existing=False,
        )

    def add_dialogue_memory(
        self,
        player_id: str,
        npc_id: str,
        dialogue_tier: str,
        text: str,
        scene_info: dict[str, Any] | None = None,
        *,
        run_id: str = "",
    ) -> None:
        text = (text or "").strip()
        if len(text) < self._min_store_chars:
            return
        tier = dialogue_tier if dialogue_tier in DIALOGUE_TIERS else "daily"
        scene_info = scene_info or {}
        rid = (run_id or str(scene_info.get("run_id") or "")).strip()
        meta: dict[str, Any] = {
            "memory_type": "dialogue",
            "dialogue_tier": tier,
            "tier": tier,
            "player_id": player_id,
            "npc_id": npc_id,
            "source": "runtime",
            "created_at": _now_iso(),
        }
        if rid:
            meta["run_id"] = rid
        floor = scene_info.get("floor")
        if floor is not None:
            meta["floor"] = int(floor) if str(floor).isdigit() else str(floor)
        coll = self._collection()
        coll.add(
            ids=[str(uuid.uuid4())],
            embeddings=_embed_texts([text]),
            documents=[text],
            metadatas=[meta],
        )

    def add_run_event(
        self,
        *,
        player_id: str,
        npc_id: str,
        run_id: str,
        text: str,
        tier: str = "major",
        source: str = "system",
        scene_info: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        importance: float | None = None,
    ) -> bool:
        """本局事件摘要。必须 run_id；默认仅当前局可检索。"""
        text = (text or "").strip()
        rid = (run_id or "").strip()
        if not rid or len(text) < self._min_store_chars:
            return False
        if self._recent_duplicate(
            memory_type="run_event",
            player_id=player_id,
            npc_id=npc_id,
            run_id=rid,
            text=text,
        ):
            return False
        scene_info = scene_info or {}
        event_tier = tier if tier in RUN_EVENT_TIERS else "major"
        meta: dict[str, Any] = {
            "memory_type": "run_event",
            "tier": event_tier,
            "player_id": player_id,
            "npc_id": npc_id,
            "run_id": rid,
            "source": source or "system",
            "created_at": _now_iso(),
        }
        if importance is not None:
            meta["importance"] = float(importance)
        floor = scene_info.get("floor")
        if floor is not None:
            try:
                meta["floor"] = int(floor)
            except (TypeError, ValueError):
                meta["floor"] = str(floor)
        room = scene_info.get("room_label") or scene_info.get("room_type")
        if room:
            meta["room_id"] = str(room)[:64]
        if tags:
            meta["tags"] = ",".join(str(t) for t in tags)[:200]
        coll = self._collection()
        coll.add(
            ids=[str(uuid.uuid4())],
            embeddings=_embed_texts([text]),
            documents=[text],
            metadatas=[meta],
        )
        return True

    def add_reflection(
        self,
        *,
        player_id: str,
        npc_id: str,
        text: str,
        source_run_id: str = "",
        source: str = "system",
    ) -> bool:
        """跨局反思/教训。"""
        text = (text or "").strip()
        if len(text) < self._min_store_chars:
            return False
        meta: dict[str, Any] = {
            "memory_type": "reflection",
            "player_id": player_id,
            "npc_id": npc_id,
            "source": source or "system",
            "created_at": _now_iso(),
        }
        if source_run_id:
            meta["source_run_id"] = source_run_id
            meta["run_id"] = source_run_id
        coll = self._collection()
        coll.add(
            ids=[str(uuid.uuid4())],
            embeddings=_embed_texts([text]),
            documents=[text],
            metadatas=[meta],
        )
        return True

    def _recent_duplicate(
        self,
        *,
        memory_type: str,
        player_id: str,
        npc_id: str,
        run_id: str,
        text: str,
    ) -> bool:
        """短窗近义去重：同 run 同 type 正文完全一致则跳过。"""
        coll = self._collection()
        try:
            where: dict[str, Any] = {
                "memory_type": memory_type,
                "player_id": player_id,
                "npc_id": npc_id,
            }
            if run_id:
                where["run_id"] = run_id
            existing = coll.get(where=where, include=["documents"])
            docs = (existing.get("documents") or [])[:40]
            needle = text.strip()
            return any(str(d).strip() == needle for d in docs if d)
        except Exception:
            return False

    def search_context(
        self,
        query: str,
        player_id: str,
        npc_id: str,
        executor: ThreadPoolExecutor | None = None,
        *,
        chat_mode: str = "",
        channel: str = "",
        run_id: str = "",
        weights: dict[str, int] | None = None,
    ) -> dict[str, list[str]]:
        """
        一次 embed，并行多 type 检索。
        channel: chat | think_combat | think_safe | run_start | default
        兼容旧调用：仅 chat_mode 时走旧 _k_for_mode + 仍查 run_event/reflection。
        """
        embedding = _embed_texts([query or "战场"])[0]
        budget = self._budget_for(channel=channel, chat_mode=chat_mode, weights=weights)
        rid = (run_id or "").strip()

        tasks: dict[str, tuple[dict[str, Any], int]] = {}

        if budget["k_world"] > 0:
            tasks["world_chunks"] = (
                {"memory_type": "world", "npc_id": "global"},
                budget["k_world"],
            )
        if budget["k_persona"] > 0:
            tasks["persona_chunks"] = (
                {"memory_type": "persona", "npc_id": npc_id},
                budget["k_persona"],
            )
        if budget["k_daily"] > 0:
            tasks["dialogue_daily_chunks"] = (
                {
                    "memory_type": "dialogue",
                    "dialogue_tier": "daily",
                    "player_id": player_id,
                    "npc_id": npc_id,
                },
                budget["k_daily"],
            )
        if budget["k_important"] > 0:
            tasks["dialogue_important_chunks"] = (
                {
                    "memory_type": "dialogue",
                    "dialogue_tier": "important",
                    "player_id": player_id,
                    "npc_id": npc_id,
                },
                budget["k_important"],
            )
        if budget["k_run_event"] > 0 and rid:
            where_ev: dict[str, Any] = {
                "memory_type": "run_event",
                "player_id": player_id,
                "npc_id": npc_id,
            }
            if budget.get("run_event_current_only", True):
                where_ev["run_id"] = rid
            tasks["run_event_chunks"] = (where_ev, budget["k_run_event"])
        if budget["k_reflection"] > 0:
            tasks["reflection_chunks"] = (
                {
                    "memory_type": "reflection",
                    "player_id": player_id,
                    "npc_id": npc_id,
                },
                budget["k_reflection"],
            )

        if not tasks:
            return empty_memory_context()

        out = self._parallel_query_tasks(tasks, embedding, executor)
        # 保证 key 齐全
        base = empty_memory_context()
        base.update(out)
        return base

    def _budget_for(
        self,
        *,
        channel: str,
        chat_mode: str,
        weights: dict[str, int] | None,
    ) -> dict[str, Any]:
        if weights:
            return {
                "k_world": int(weights.get("world", self._k_world)),
                "k_persona": int(weights.get("persona", self._k_persona)),
                "k_daily": int(weights.get("daily", self._k_daily)),
                "k_important": int(weights.get("important", self._k_important)),
                "k_run_event": int(weights.get("run_event", self._k_run_event)),
                "k_reflection": int(weights.get("reflection", self._k_reflection)),
                "run_event_current_only": True,
            }

        ch = (channel or "").strip()
        if ch in CHANNEL_PRESETS:
            return dict(CHANNEL_PRESETS[ch])

        # 兼容旧 chat_mode 加权
        kw, kp, kd, ki = self._k_for_mode(chat_mode, None)
        return {
            "k_world": kw,
            "k_persona": kp,
            "k_daily": kd,
            "k_important": ki,
            "k_run_event": self._k_run_event,
            "k_reflection": self._k_reflection,
            "run_event_current_only": True,
        }

    def _k_for_mode(
        self,
        chat_mode: str,
        weights: dict[str, int] | None,
    ) -> tuple[int, int, int, int]:
        if weights:
            return (
                int(weights.get("world", self._k_world)),
                int(weights.get("persona", self._k_persona)),
                int(weights.get("daily", self._k_daily)),
                int(weights.get("important", self._k_important)),
            )
        if chat_mode == "meta_lore":
            return (max(self._k_world, 4), max(self._k_persona, 3), 2, max(self._k_important, 4))
        if chat_mode == "emotional":
            return (1, max(self._k_persona, 3), 2, max(self._k_important, 5))
        if chat_mode in ("combat_ack", "combat_question"):
            return (1, 2, max(self._k_daily, 4), max(self._k_important, 3))
        if chat_mode == "rest_banter":
            return (2, max(self._k_persona, 3), max(self._k_daily, 4), max(self._k_important, 4))
        return (self._k_world, self._k_persona, self._k_daily, self._k_important)

    def _parallel_query_tasks(
        self,
        tasks: dict[str, tuple[dict[str, Any], int]],
        embedding: list[float],
        executor: ThreadPoolExecutor | None,
    ) -> dict[str, list[str]]:
        def _collect(pool: ThreadPoolExecutor) -> dict[str, list[str]]:
            futures = {
                pool.submit(self._query_with_embedding, embedding, where, limit): key
                for key, (where, limit) in tasks.items()
            }
            out: dict[str, list[str]] = {}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    out[key] = future.result()
                except Exception:
                    out[key] = []
            return out

        workers = max(4, len(tasks))
        if executor is not None:
            return _collect(executor)

        with ThreadPoolExecutor(max_workers=workers) as local_pool:
            return _collect(local_pool)
