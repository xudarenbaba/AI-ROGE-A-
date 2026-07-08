"""
train.py — 训练 assault 姿态策略网络。

用法：
    # 默认配置（2M 步，约 1-2 小时 CPU）
    python -m rl.train

    # 快速验证（5 万步，几分钟）
    python -m rl.train --timesteps 50000 --run-name debug

    # 从检查点继续，训练到累计 8000 万步（--timesteps 为总目标，非额外步数）
    python -m rl.train --resume rl/checkpoints/assault_1000000_steps.zip

    # 在 8000 万步基础上再训 500 万（总目标 8500 万）
    python -m rl.train --resume rl/checkpoints/assault_79999360_steps.zip --timesteps 85000000

参数说明见 argparse 部分。
输出：
    rl/checkpoints/assault_<steps>_steps.zip  — 定期检查点
    rl/checkpoints/assault_best/              — 最优模型（按 episode reward）
    rl/logs/                                  — TensorBoard 日志（可选）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，使 rl.env 可正常 import
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from rl.env import AssaultEnv


# ── 超参数 ────────────────────────────────────────────────────────────────────

# RecurrentPPO（LSTM）超参数。LSTM 适合较小的 batch、较短的 n_steps。
DEFAULTS = {
    "timesteps":    80_000_000,
    "n_envs":       28,           # 并行环境数
    "n_steps":      512,         # 每个 env 每次 rollout 的步数（LSTM 用短一点）
    "batch_size":   256,         # minibatch 大小（需能整除 n_envs*n_steps）
    "n_epochs":     10,
    "lr":           3e-4,
    "gamma":        0.995,
    "gae_lambda":   0.95,
    "clip_range":   0.2,
    "ent_coef":     0.01,
    "vf_coef":      0.5,
    "max_grad_norm":0.5,
    "net_arch":     [256, 256],  # LSTM 之后的 MLP 头
    "lstm_hidden":  128,         # LSTM 隐藏单元数
    "run_name":     "assault_combat_v3",
    "checkpoint_freq": 500_000,
    "eval_freq":    100_000,
    "eval_episodes":20,
    "resume":       None,
}


# ── 回调 ──────────────────────────────────────────────────────────────────────

class ProgressCallback(BaseCallback):
    """每 10 万步打印一次简单的训练统计。"""

    def __init__(self, print_freq: int = 100_000) -> None:
        super().__init__()
        self.print_freq = print_freq
        self._last_print = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print >= self.print_freq:
            self._last_print = self.num_timesteps
            ep_info = self.model.ep_info_buffer
            if ep_info:
                rewards = [e["r"] for e in ep_info]
                lengths = [e["l"] for e in ep_info]
                print(
                    f"[{self.num_timesteps:>9,}] "
                    f"ep_rew mean={np.mean(rewards):6.1f} "
                    f"max={np.max(rewards):6.1f}  "
                    f"ep_len mean={np.mean(lengths):5.0f}"
                )
            else:
                print(f"[{self.num_timesteps:>9,}] collecting...")
        return True


# ── 主函数 ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train assault RL policy (PPO)")
    p.add_argument("--timesteps",      type=int,   default=DEFAULTS["timesteps"],
                   help="累计训练总步数目标（resume 时也是总目标，不是额外步数）")
    p.add_argument("--n-envs",         type=int,   default=DEFAULTS["n_envs"])
    p.add_argument("--n-steps",        type=int,   default=DEFAULTS["n_steps"])
    p.add_argument("--batch-size",     type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--n-epochs",       type=int,   default=DEFAULTS["n_epochs"])
    p.add_argument("--lr",             type=float, default=DEFAULTS["lr"])
    p.add_argument("--gamma",          type=float, default=DEFAULTS["gamma"])
    p.add_argument("--ent-coef",       type=float, default=DEFAULTS["ent_coef"])
    p.add_argument("--run-name",       type=str,   default=DEFAULTS["run_name"])
    p.add_argument("--checkpoint-freq",type=int,   default=DEFAULTS["checkpoint_freq"])
    p.add_argument("--eval-freq",      type=int,   default=DEFAULTS["eval_freq"])
    p.add_argument("--eval-episodes",  type=int,   default=DEFAULTS["eval_episodes"])
    p.add_argument("--resume",         type=str,   default=DEFAULTS["resume"])
    p.add_argument("--no-subproc",     action="store_true",
                   help="禁用 SubprocVecEnv，使用 DummyVecEnv（调试用）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_dir = _ROOT / "rl" / "checkpoints"
    log_dir  = _ROOT / "rl" / "logs"
    best_dir = ckpt_dir / "assault_best"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── 快速 sanity check（单环境）───────────────────────────────────────────
    print("Running env sanity check...")
    check_env(AssaultEnv(), warn=True)
    print("Env check passed.")

    # ── 创建并行训练环境 ─────────────────────────────────────────────────────
    vec_cls = "dummy" if args.no_subproc else "subproc"
    train_env = make_vec_env(
        AssaultEnv,
        n_envs=args.n_envs,
        vec_env_cls=SubprocVecEnv if not args.no_subproc else None,
    )
    train_env = VecMonitor(train_env)

    # 评估环境（单独一个，不影响训练统计）
    eval_env = VecMonitor(make_vec_env(AssaultEnv, n_envs=1))

    # ── 回调 ─────────────────────────────────────────────────────────────────
    callbacks = [
        ProgressCallback(print_freq=100_000),
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=str(ckpt_dir),
            name_prefix="assault",
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(log_dir),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            verbose=1,
        ),
    ]

    # ── 模型创建或恢复 ───────────────────────────────────────────────────────
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = RecurrentPPO.load(
            args.resume,
            env=train_env,
            device="auto",
        )
        start_steps = int(model.num_timesteps)
        remaining = args.timesteps - start_steps
        if remaining <= 0:
            print(f"Already at {start_steps:,} steps (target {args.timesteps:,}), nothing to do.")
            return
        # SB3 resume 时会把 learn(total_timesteps) 加上已有步数，故只传剩余步数。
        total_ts = remaining
    else:
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=train_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=DEFAULTS["gae_lambda"],
            clip_range=DEFAULTS["clip_range"],
            ent_coef=args.ent_coef,
            vf_coef=DEFAULTS["vf_coef"],
            max_grad_norm=DEFAULTS["max_grad_norm"],
            policy_kwargs=dict(
                net_arch=DEFAULTS["net_arch"],
                lstm_hidden_size=DEFAULTS["lstm_hidden"],
                enable_critic_lstm=True,
            ),
            tensorboard_log=str(log_dir),
            verbose=0,
            device="auto",
        )
        total_ts = args.timesteps

    print(f"\nAlgorithm: RecurrentPPO (LSTM hidden={DEFAULTS['lstm_hidden']})")
    print(f"Policy MLP head: {DEFAULTS['net_arch']}")
    print(f"Observation dim: {AssaultEnv().observation_space.shape[0]}")
    print(f"Action space: {AssaultEnv().action_space.n} discrete actions")
    if args.resume:
        print(
            f"Resume: {start_steps:,} → {args.timesteps:,} "
            f"(+{total_ts:,} remaining, {args.n_envs} envs)\n"
        )
    else:
        print(f"Training for {total_ts:,} steps with {args.n_envs} envs\n")

    # ── 训练 ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=total_ts,
        callback=callbacks,
        reset_num_timesteps=args.resume is None,
        tb_log_name=args.run_name,
        progress_bar=False,
    )

    # ── 保存最终模型 ─────────────────────────────────────────────────────────
    final_path = ckpt_dir / f"{args.run_name}_final"
    model.save(str(final_path))
    print(f"\nTraining complete. Final model saved to: {final_path}.zip")
    print(f"Best model saved to: {best_dir}/best_model.zip")
    print("\nNext step: python -m rl.export_onnx")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
