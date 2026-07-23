"""
export_onnx.py — 将 assault_skirmish RecurrentPPO (LSTM) 导出为 ONNX。

用法：
    python -m rl.limbs.assault_skirmish.export_onnx --verify
    python -m rl.limbs.assault_skirmish.export_onnx \\
        --model rl/limbs/assault_skirmish/checkpoints/assault_best/best_model.zip --verify

输出（默认，仅 resources）：
    game/resources/limbs/assault_skirmish/policy.onnx
    game/resources/limbs/assault_skirmish/meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# __file__ = .../rl/limbs/assault_skirmish/export_onnx.py
# parents[0]=limb dir, [1]=limbs, [2]=rl, [3]=project root
_THIS_FILE = Path(__file__).resolve()
_LIMB_DIR = _THIS_FILE.parent
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rl.limbs.assault_skirmish.env import ACTION_VECTORS, N_ACTIONS, OBS_DIM

LIMB_ID = "assault_skirmish"


class LstmActorOnly(nn.Module):
    """包装 RecurrentPPO actor（含 LSTM）用于 ONNX 导出。"""

    def __init__(self, sb3_policy) -> None:  # noqa: ANN001
        super().__init__()
        self.lstm = sb3_policy.lstm_actor
        self.policy_net = sb3_policy.mlp_extractor.policy_net
        self.action_net = sb3_policy.action_net
        self.hidden_size = self.lstm.hidden_size
        self.num_layers = self.lstm.num_layers

    def forward(
        self,
        obs: torch.Tensor,
        h_in: torch.Tensor,
        c_in: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lstm_in = obs.unsqueeze(0)
        lstm_out, (h_out, c_out) = self.lstm(lstm_in, (h_in, c_in))
        latent = lstm_out.squeeze(0)
        latent_pi = self.policy_net(latent)
        logits = self.action_net(latent_pi)
        return logits, h_out, c_out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Export {LIMB_ID} policy to ONNX")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型路径（.zip）。默认自动查找本肢 checkpoints",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(
            _PROJECT_ROOT
            / "game"
            / "resources"
            / "limbs"
            / LIMB_ID
            / "policy.onnx"
        ),
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="导出后验证 ONNX 与 PyTorch 推理一致性",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset（onnxruntime-web wasm 推荐 12）",
    )
    return p.parse_args()


def find_model(model_arg: str | None) -> Path:
    if model_arg:
        p = Path(model_arg)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        return p

    ckpt = _LIMB_DIR / "checkpoints"
    candidates = [
        ckpt / "assault_best" / "best_model.zip",
        *sorted(ckpt.glob("assault_lstm*_final.zip")),
        *sorted(ckpt.glob("assault*_final.zip")),
        *sorted(ckpt.glob("assault_*_steps.zip")),
    ]
    for c in candidates:
        if c.exists():
            print(f"Auto-selected model: {c}")
            return c
    raise FileNotFoundError(
        f"No model found under {ckpt}. "
        f"Run `python -m rl.limbs.{LIMB_ID}.train` first, "
        "or specify --model path/to/model.zip"
    )


def main() -> None:
    args = parse_args()

    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        print("ERROR: sb3-contrib not installed. Run: pip install sb3-contrib")
        sys.exit(1)

    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnx/onnxruntime not installed. Run: pip install onnx onnxruntime")
        sys.exit(1)

    model_path = find_model(args.model)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _PROJECT_ROOT / output_path
    if output_path.name == "policy.onnx":
        meta_path = output_path.with_name("meta.json")
    else:
        meta_path = output_path.with_name(output_path.stem + "_meta.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Limb: {LIMB_ID}")
    print(f"Loading model from: {model_path}")
    model = RecurrentPPO.load(str(model_path), device="cpu")
    policy = model.policy.to("cpu")
    policy.eval()

    actor = LstmActorOnly(policy)
    actor.eval()

    H = actor.hidden_size
    L = actor.num_layers
    print(f"LSTM: hidden_size={H}, num_layers={L}")

    dummy_obs = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    dummy_h = torch.zeros(L, 1, H, dtype=torch.float32)
    dummy_c = torch.zeros(L, 1, H, dtype=torch.float32)

    print(f"Exporting to ONNX (opset {args.opset})...")
    with torch.no_grad():
        torch.onnx.export(
            actor,
            (dummy_obs, dummy_h, dummy_c),
            str(output_path),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["obs", "h_in", "c_in"],
            output_names=["logits", "h_out", "c_out"],
            dynamo=False,
        )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed.")
    print(f"Model size: {output_path.stat().st_size / 1024:.1f} KB")

    if args.verify:
        print("\nVerifying ONNX vs PyTorch (10-step rollout with hidden state)...")
        sess = ort.InferenceSession(str(output_path))
        rng = np.random.default_rng(0)

        h_t = np.zeros((L, 1, H), dtype=np.float32)
        c_t = np.zeros((L, 1, H), dtype=np.float32)
        h_p = torch.zeros(L, 1, H)
        c_p = torch.zeros(L, 1, H)

        max_err = 0.0
        for _ in range(10):
            obs_np = rng.uniform(-1, 1, (1, OBS_DIM)).astype(np.float32)
            ort_logits, h_t, c_t = sess.run(
                ["logits", "h_out", "c_out"],
                {"obs": obs_np, "h_in": h_t, "c_in": c_t},
            )
            with torch.no_grad():
                pt_logits, h_p, c_p = actor(torch.from_numpy(obs_np), h_p, c_p)
            err = np.abs(pt_logits.numpy() - ort_logits).max()
            max_err = max(max_err, err)

        print(f"Max abs error over 10-step rollout: {max_err:.2e}")
        if max_err < 1e-4:
            print("Verification PASSED (error < 1e-4)")
        else:
            print(f"WARNING: error {max_err:.2e} higher than expected.")

    meta = {
        "limb_id": LIMB_ID,
        "obs_dim": OBS_DIM,
        "n_actions": N_ACTIONS,
        "action_vectors": ACTION_VECTORS,
        "lstm_hidden": H,
        "lstm_layers": L,
        "model_file": output_path.name,
        "opset": args.opset,
        "recurrent": True,
        "description": (
            f"{LIMB_ID} RecurrentPPO (LSTM). "
            "Inputs: obs[1,obs_dim], h_in, c_in. "
            "Outputs: logits, h_out, c_out."
        ),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\nMeta written to: {meta_path}")
    print(f"ONNX model written to: {output_path}")
    print("\nNext step: reload the game (game/resources/limbs/).")


if __name__ == "__main__":
    main()
