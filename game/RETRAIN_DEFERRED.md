# 暂缓项：需要 RL 重训或同步 `rl/env.py` 后再做

当前 ONNX / `_rlBuildObs()` / `rl/env.py` 训练分布保持不变。以下改动**尚未实现**，训完模型后再评估。

## 必须重训或同步环境

| 项 | 原因 | 涉及文件 |
|----|------|----------|
| 障碍布局同步到训练环境 | 新布局改变射线/LOS 分布 | `rl/env.py` `OBSTACLE_LAYOUTS`，`game/obstacles/layouts.js` |
| 可破坏 / 移动障碍 | 战斗期障碍动态变化，obs 分布偏移 | `game.js` + `rl/env.py` |
| 追踪弹、反弹弹、超高密度弹幕 | 子弹轨迹与训练集差异大 | `game/bullets/patterns.js`，`rl/env.py` 奖励与 spawn |
| 修改 `_RL_OBS_DIM` 或 `_rlBuildObs` 特征 | 输入维度或语义变化 | `game/game.js`，`rl/env.py` |
| 修改 assault 动作空间 / LSTM 结构 | 模型结构不兼容 | `game/game.js`，`rl/train.py`，`rl/export_onnx.py` |
| 修改 `_RL_CANVAS_W/H` 或 ally 碰撞半径 | 归一化与训练不一致 | `game/game.js`，`rl/env.py` |
| 训练期修改 `ally.speed` / assault `cfg` | 影响 RL 相对运动 | `game.js` `allyConfig()` assault 分支 |

## 建议流程（训后）

1. 导出最终 `assault_policy.onnx` 覆盖 `game/`
2. 将 `game/obstacles/layouts.js` 碰撞盒同步到 `rl/env.py`
3. 视情况 `python rl/train.py` 微调 50k–100k steps
4. 对比：卡墙率、COMBAT 退出频率、通关层数

## 本次已实现（无需重训）

见 `game/index.html` 加载的 `fx.js`、`obstacles/layouts.js`、`bullets/patterns.js`、`meta/blessings.js` 及 `game.js` 中标注 `SAFE UPGRADE` 的改动。