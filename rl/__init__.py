"""
rl/ — 强化学习包

每个策略网络独占：
    rl/limbs/<limb_id>/
      env.py train.py export_onnx.py
      checkpoints/ logs/ config.yaml

突击网络：
    python -m rl.limbs.assault_skirmish.train
    python -m rl.limbs.assault_skirmish.export_onnx

游戏只加载：
    game/resources/limbs/<limb_id>/policy.onnx
"""
