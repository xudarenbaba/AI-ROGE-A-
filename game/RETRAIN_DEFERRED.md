# RL 环境同步与暂缓项

## 已同步到 `rl/limbs/assault_skirmish/env.py`（需微调）

| 项 | 说明 |
|----|------|
| 障碍布局 | 碰撞盒与 `game/obstacles/layouts.js` 四组一致 |
| Boss 扇形弹幕 | 38% 预警 + 5 发散弹，与 `game/bullets/patterns.js` 对齐 |

观测维度 / 动作空间 / 画布常量未改，可在现有 checkpoint 上 **resume 微调**。

## 仍暂缓（未在游戏实现或需改 obs）

| 项 | 原因 |
|----|------|
| 可破坏 / 移动障碍 | 游戏未实现 |
| 追踪弹、反弹弹、超高密度弹幕 | 游戏未实现 |
| 修改 `_RL_OBS_DIM` / 动作空间 / LSTM | 结构不兼容，需全量重训 |
| 修改画布尺寸或 ally 碰撞半径 | 归一化分布变化 |
| 训练期改 assault `cfg` / `ally.speed` | 相对运动变化 |

## 微调流程

```bash
# 1. 在最新 checkpoint 上再训 10 万步（总目标 = 当前步数 + 10 万）
python -m rl.limbs.assault_skirmish.train \
  --resume rl/limbs/assault_skirmish/checkpoints/assault_79999360_steps.zip \
  --timesteps 80100000 \
  --run-name assault_combat_v3_finetune

# 2. 导出到 game/resources（无需再 cp）
python -m rl.limbs.assault_skirmish.export_onnx \
  --model rl/limbs/assault_skirmish/checkpoints/assault_best/best_model.zip --verify
```

对比：卡墙率、COMBAT 退出频率、Boss 战躲扇形弹、通关层数。

## 本次已实现（无需重训）

`fx.js`、`obstacles/layouts.js`（仅渲染）、`meta/blessings.js`，以及 `game.js` 中非 RL 冻结的 UI/手感改动。
