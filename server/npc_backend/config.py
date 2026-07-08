from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_CACHE: dict[str, Any] | None = None

_DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "model": "deepseek-chat",
        "decide_model": "deepseek-chat",
        "api_key": "",
        "base_url": "https://api.deepseek.com",
        "temperature": 0.3,
        "timeout_s": 60,
    },
    "embeddings": {
        "model": "BAAI/bge-small-zh-v1.5",
        "cache_dir": "models",
        "local_files_only": True,
    },
    "vectorstore": {
        "persist_dir": "data/chroma",
        "collection_name": "npc_memory",
    },
    "memory": {
        "short_term_turns": 10,
        "k_world": 3,
        "k_persona": 3,
        "k_dialogue_daily": 4,
        "k_dialogue_important": 6,
        "min_store_chars": 6,
    },
    "npc_autonomy": {
        "enabled": True,
        "reflex_enabled": True,
        "idle_threshold_s": 5,
        "think_interval_s": 8,
        "speech_cooldown_s": 12,
        "scene_speech_cooldown_s": 4,
        "critical_speech_cooldown_s": 6,
        "critical_speech_cd_mul": 1.0,
        "stance_change_cooldown_s": 25,
        "intent_ttl_s": 30,
        "noop_backoff_s": [12, 24, 48],
        "rule_noop_after": 2,
        "require_scene_change": True,
        "stale_force_think_s": 55,
        "recent_lines_max": 8,
        "duplicate_similarity": 0.55,
        "max_speech_per_minute": 5,
        "triggers": {
            "ally_critical_hp_pct": 0.25,
            "ally_dire_hp_pct": 0.10,
            "player_danger_hp_pct": 0.30,
            "player_idle_for_help_s": 6,
            "social_beat_s": 38,
            "periodic_fallback_s": 60,
            "assault_guard_min_s": 10,
            "assault_player_active_s": 6,
            "assault_min_ally_hp_pct": 0.35,
            "assault_min_player_hp_pct": 0.35,
            "assault_max_enemies": 8,
            "assault_max_nearest_dist": 420,
        },
        "assault_stance_cooldown_s": 10,
        "polish": {
            "autonomous_quick": True,
        },
    },
    "concurrency": {
        "worker_threads": 8,
    },
    "logging": {
        "file": "data/logs/npc.log",
        "level": "INFO",
        "console": True,
        "clear_on_start": True,
        "max_bytes": 2000000,
        "backup_count": 3,
    },
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_reference_config() -> dict[str, Any]:
    # 用户要求参考 D:\otherwise\AI-NPC 的 LLM 与 embedding 配置。
    reference = Path(r"D:\otherwise\AI-NPC\config.yaml")
    return _read_yaml(reference)


def load_config(force_reload: bool = False) -> dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    project_cfg = _read_yaml(_project_root() / "config.yaml")
    cfg = _deep_merge(_DEFAULT_CONFIG, project_cfg)

    if not (cfg.get("llm", {}).get("api_key") or "").strip():
        ref_cfg = _read_reference_config()
        llm = ref_cfg.get("llm", {})
        emb = ref_cfg.get("embeddings", {})
        if llm:
            cfg["llm"] = _deep_merge(cfg.get("llm", {}), llm)
        if emb:
            cfg["embeddings"] = _deep_merge(cfg.get("embeddings", {}), emb)

    if env_key := os.environ.get("AI_NPC_LLM_API_KEY"):
        cfg.setdefault("llm", {})["api_key"] = env_key
    if env_base := os.environ.get("AI_NPC_LLM_BASE_URL"):
        cfg.setdefault("llm", {})["base_url"] = env_base
    if env_model := os.environ.get("AI_NPC_LLM_MODEL"):
        cfg.setdefault("llm", {})["model"] = env_model

    _CONFIG_CACHE = cfg
    return cfg

