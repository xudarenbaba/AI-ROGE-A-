"""
assault_skirmish — 突击微操 RL 肢（现有 assault 网络）。

训练：python -m rl.limbs.assault_skirmish.train
导出：python -m rl.limbs.assault_skirmish.export_onnx
游戏：game/resources/limbs/assault_skirmish/policy.onnx
"""

from rl.limbs.assault_skirmish.env import (
    ACTION_VECTORS,
    AssaultEnv,
    N_ACTIONS,
    OBS_DIM,
)

__all__ = [
    "ACTION_VECTORS",
    "AssaultEnv",
    "N_ACTIONS",
    "OBS_DIM",
]
