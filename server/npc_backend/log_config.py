from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from server.npc_backend.config import load_config

_CONFIGURED = False


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def setup_npc_logging(force: bool = False) -> logging.Logger:
    """初始化 NPC 日志，写入文件（默认 data/logs/npc.log）。每次启动清空。"""
    global _CONFIGURED
    logger = logging.getLogger("npc")

    cfg = load_config().get("logging", {})
    log_file = Path(cfg.get("file", "data/logs/npc.log"))
    if not log_file.is_absolute():
        log_file = _project_root() / log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if cfg.get("clear_on_start", True):
        log_file.write_text("", encoding="utf-8")

    if _CONFIGURED and not force:
        logger.info("NPC server started (logging already configured) → %s", log_file)
        return logger

    level_name = str(cfg.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    handler = RotatingFileHandler(
        log_file,
        maxBytes=int(cfg.get("max_bytes", 2_000_000)),
        backupCount=int(cfg.get("backup_count", 3)),
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(handler)

    if cfg.get("console", True):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("%(asctime)s [npc] %(message)s"))
        logger.addHandler(console)

    _CONFIGURED = True
    logger.info("NPC logging initialized → %s", log_file)
    return logger


def npc_logger() -> logging.Logger:
    if not _CONFIGURED:
        return setup_npc_logging()
    return logging.getLogger("npc")