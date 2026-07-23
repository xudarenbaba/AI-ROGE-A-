# rl/_shared

可选：多肢共用的纯工具代码（例如通用 obs 归一化 helper）。

**不要**在这里放某一肢的 env/train/checkpoint/onnx。
每肢业务代码与产物只放在 `rl/limbs/<limb_id>/`。
