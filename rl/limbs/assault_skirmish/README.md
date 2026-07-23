# Limb: assault_skirmish

突击姿态 RL 策略。本目录自包含训练代码与训练产物。

## 结构

```text
rl/limbs/assault_skirmish/
  env.py              # AssaultEnv
  train.py            # RecurrentPPO
  export_onnx.py      # 导出 → game/resources/...
  config.yaml
  __init__.py
  checkpoints/        # 训练检查点（gitignore）
  logs/               # TensorBoard（gitignore）
```

## 命令（项目根目录）

```bash
python -m rl.limbs.assault_skirmish.train
python -m rl.limbs.assault_skirmish.train --timesteps 50000 --run-name debug
python -m rl.limbs.assault_skirmish.train \
  --resume rl/limbs/assault_skirmish/checkpoints/assault_best/best_model.zip

python -m rl.limbs.assault_skirmish.export_onnx --verify
```

导出目标（仅此路径）：

```text
game/resources/limbs/assault_skirmish/policy.onnx
game/resources/limbs/assault_skirmish/meta.json
```

## 新肢

复制本目录为 `rl/limbs/<new_id>/`，改 env / `LIMB_ID` / export 输出路径，更新 `game/resources/limbs/manifest.json`。
