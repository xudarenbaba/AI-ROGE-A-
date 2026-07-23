# RL Limb 目录规范

## 原则

- **一肢一目录、自包含**：代码 + checkpoints + logs 都在 `rl/limbs/<limb_id>/`
- **导出唯一落点**：`game/resources/limbs/<limb_id>/`
- **`rl/` 根目录只有** `__init__.py`、`limbs/`、可选 `_shared/`
- **`game/` 根目录不放** `.onnx`

## 结构

```text
rl/
  __init__.py
  _shared/                    # 可选公共工具
  limbs/
    assault_skirmish/
      env.py train.py export_onnx.py
      config.yaml README.md
      checkpoints/            # gitignore
      logs/                   # gitignore

game/resources/limbs/
  manifest.json
  assault_skirmish/
    policy.onnx               # 提交 / 部署
    meta.json
```

## 命令

```bash
python -m rl.limbs.assault_skirmish.train
python -m rl.limbs.assault_skirmish.export_onnx --verify
```

## 新肢

1. `mkdir -p rl/limbs/<id>/{checkpoints,logs}`
2. 以 assault_skirmish 为模板改 env / LIMB_ID / export 路径
3. 更新 `game/resources/limbs/manifest.json`
