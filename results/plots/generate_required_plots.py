import csv
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import A2C, DQN, PPO

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from environment.custom_env import HospitalTriageEnv


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


def ensure_dirs() -> None:
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)


def load_best_experiments() -> Dict[str, int]:
    with open("results/logs/final_logs.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return {algo: int(meta["best_experiment"]) for algo, meta in data.items()}


def run_eval_episodes_sb3(model, episodes: int = 100) -> List[float]:
    env = HospitalTriageEnv(max_steps=50, max_losses=3)
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
        rewards.append(float(total_reward))
    env.close()
    return rewards


def run_eval_episodes_reinforce(model_path: str, episodes: int = 100) -> List[float]:
    env = HospitalTriageEnv(max_steps=50, max_losses=3)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim)
    policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    policy.eval()

    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            state = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                probs = policy(state)
            action = int(torch.argmax(probs).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(float(total_reward))

    env.close()
    return rewards


def rolling_mean(values: List[float], window: int = 10) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    out = np.copy(arr)
    for i in range(window - 1, len(arr)):
        out[i] = np.mean(arr[i - window + 1 : i + 1])
    return out


def rolling_std(values: List[float], window: int = 10) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return np.zeros_like(arr)
    out = np.zeros_like(arr)
    for i in range(window - 1, len(arr)):
        out[i] = np.std(arr[i - window + 1 : i + 1])
    return out


def estimate_convergence_episode(values: List[float], window: int = 10, tol_ratio: float = 0.05) -> int:
    rm = rolling_mean(values, window)
    if len(rm) < window + 5:
        return len(values)

    final_target = float(np.mean(rm[-window:]))
    tol = max(1.0, abs(final_target) * tol_ratio)

    for idx in range(window - 1, len(rm) - 5):
        trailing = rm[idx : idx + 5]
        if np.all(np.abs(trailing - final_target) <= tol):
            return idx + 1
    return len(values)


def save_convergence_table(convergence: Dict[str, Tuple[int, float]]) -> None:
    with open("results/tables/episodes_to_converge.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Estimated Convergence Episode", "Final Rolling Mean Reward"])
        for algo, (ep, final_rm) in convergence.items():
            writer.writerow([algo, ep, round(final_rm, 2)])


def plot_cumulative_rewards(eval_rewards: Dict[str, List[float]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    for ax, (algo, rewards) in zip(axes, eval_rewards.items()):
        cumulative = np.cumsum(rewards)
        ax.plot(range(1, len(cumulative) + 1), cumulative, linewidth=2)
        ax.set_title(f"{algo} Cumulative Reward")
        ax.set_xlabel("Evaluation Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Cumulative Rewards Over Episodes (Best Models)")
    plt.tight_layout()
    plt.savefig("results/plots/cumulative_rewards_best_models_subplots.png", dpi=200)
    plt.close()


def plot_training_stability(eval_rewards: Dict[str, List[float]]) -> Dict[str, Tuple[int, float]]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    convergence: Dict[str, Tuple[int, float]] = {}

    for ax, (algo, rewards) in zip(axes, eval_rewards.items()):
        x = np.arange(1, len(rewards) + 1)
        rm = rolling_mean(rewards, window=10)
        rs = rolling_std(rewards, window=10)
        conv_ep = estimate_convergence_episode(rewards, window=10, tol_ratio=0.05)

        ax.plot(x, rewards, alpha=0.35, label="Episode Reward")
        ax.plot(x, rm, linewidth=2, label="Rolling Mean (10)")
        ax.fill_between(x, rm - rs, rm + rs, alpha=0.2, label="Rolling Std")
        ax.axvline(conv_ep, color="red", linestyle="--", linewidth=1.5, label=f"Converge Ep: {conv_ep}")
        ax.set_title(f"{algo} Stability")
        ax.set_xlabel("Evaluation Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        final_rm = float(np.mean(rm[-10:])) if len(rm) >= 10 else float(np.mean(rm))
        convergence[algo] = (conv_ep, final_rm)

    plt.suptitle("Training Stability Proxy (Best Models)")
    plt.tight_layout()
    plt.savefig("results/plots/training_stability_subplots.png", dpi=200)
    plt.close()

    return convergence


def plot_episodes_to_converge(convergence: Dict[str, Tuple[int, float]]) -> None:
    algos = list(convergence.keys())
    episodes = [convergence[a][0] for a in algos]
    final_rm = [convergence[a][1] for a in algos]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(algos, episodes)
    axes[0].set_title("Estimated Episodes to Converge")
    axes[0].set_ylabel("Episode")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(algos, final_rm)
    axes[1].set_title("Final Rolling Mean Reward (Window=10)")
    axes[1].set_ylabel("Reward")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/episodes_to_converge_subplots.png", dpi=200)
    plt.close()


def parse_monitor_rewards(monitor_file: str) -> List[float]:
    rewards = []
    with open(monitor_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.strip().split(",")
        if len(parts) >= 1:
            try:
                rewards.append(float(parts[0]))
            except ValueError:
                continue
    return rewards


def plot_objective_and_entropy_if_available() -> None:
    dqn_files = sorted(glob.glob("results/logs/training/dqn/*_monitor.csv"))
    reinforce_files = sorted(glob.glob("results/logs/training/reinforce/*_training.csv"))
    note_lines = []

    if dqn_files:
        rewards = parse_monitor_rewards(dqn_files[-1])
        if rewards:
            plt.figure(figsize=(10, 5))
            plt.plot(rolling_mean(rewards, window=20), linewidth=2)
            plt.title("DQN Objective Proxy (Episode Reward Rolling Mean)")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("results/plots/dqn_objective_curve.png", dpi=200)
            plt.close()
    else:
        note_lines.append("No DQN monitor logs found. Re-run DQN training to generate dqn_objective_curve.png.")

    if reinforce_files:
        plt.figure(figsize=(10, 5))
        for file_path in reinforce_files[-3:]:
            entropies = []
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entropies.append(float(row["policy_entropy"]))
            if entropies:
                label = os.path.basename(file_path).replace("_training.csv", "")
                plt.plot(rolling_mean(entropies, window=20), label=label)

        plt.title("Policy Entropy Curves (REINFORCE)")
        plt.xlabel("Episode")
        plt.ylabel("Entropy")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig("results/plots/pg_entropy_curves.png", dpi=200)
        plt.close()
    else:
        note_lines.append("No REINFORCE training CSV logs found. Re-run REINFORCE training to generate pg_entropy_curves.png.")

    if note_lines:
        with open("results/plots/missing_training_logs_note.txt", "w", encoding="utf-8") as f:
            for line in note_lines:
                f.write(line + "\n")


def main() -> None:
    ensure_dirs()

    best = load_best_experiments()

    dqn_model = DQN.load(f"models/dqn_improved/dqn_improved_exp_{best['DQN']}")
    ppo_model = PPO.load(f"models/ppo/ppo_exp_{best['PPO']}")
    a2c_model = A2C.load(f"models/a2c/a2c_exp_{best['A2C']}")

    eval_rewards = {
        "DQN": run_eval_episodes_sb3(dqn_model, episodes=100),
        "PPO": run_eval_episodes_sb3(ppo_model, episodes=100),
        "A2C": run_eval_episodes_sb3(a2c_model, episodes=100),
        "REINFORCE": run_eval_episodes_reinforce(
            f"models/reinforce/reinforce_exp_{best['REINFORCE']}.pth", episodes=100
        ),
    }

    plot_cumulative_rewards(eval_rewards)
    convergence = plot_training_stability(eval_rewards)
    plot_episodes_to_converge(convergence)
    save_convergence_table(convergence)
    plot_objective_and_entropy_if_available()

    print("Generated required plots in results/plots and convergence table in results/tables.")


if __name__ == "__main__":
    main()
