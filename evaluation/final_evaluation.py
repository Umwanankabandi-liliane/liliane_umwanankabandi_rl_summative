import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN, PPO, A2C
from environment.custom_env import HospitalTriageEnv


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
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


def evaluate_sb3_model(model, env, episodes=10):
    rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    return float(np.mean(rewards))


def evaluate_reinforce_model(policy, env, episodes=10):
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

        rewards.append(total_reward)

    return float(np.mean(rewards))


def main():
    results = []

    print("\n===== DQN =====")
    for i in range(1, 11):
        env = HospitalTriageEnv(max_steps=50, max_losses=3)
        model = DQN.load(f"models/dqn_improved/dqn_improved_exp_{i}")
        score = evaluate_sb3_model(model, env, episodes=10)
        print(f"DQN Exp {i}: {score:.2f}")
        results.append(("DQN", i, score))
        env.close()

    print("\n===== PPO =====")
    for i in range(1, 11):
        env = HospitalTriageEnv(max_steps=50, max_losses=3)
        model = PPO.load(f"models/ppo/ppo_exp_{i}")
        score = evaluate_sb3_model(model, env, episodes=10)
        print(f"PPO Exp {i}: {score:.2f}")
        results.append(("PPO", i, score))
        env.close()

    print("\n===== A2C =====")
    for i in range(1, 11):
        env = HospitalTriageEnv(max_steps=50, max_losses=3)
        model = A2C.load(f"models/a2c/a2c_exp_{i}")
        score = evaluate_sb3_model(model, env, episodes=10)
        print(f"A2C Exp {i}: {score:.2f}")
        results.append(("A2C", i, score))
        env.close()

    print("\n===== REINFORCE =====")
    temp_env = HospitalTriageEnv(max_steps=50, max_losses=3)
    state_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.n
    temp_env.close()

    for i in range(1, 11):
        env = HospitalTriageEnv(max_steps=50, max_losses=3)
        policy = PolicyNetwork(state_dim, action_dim)
        policy.load_state_dict(
            torch.load(f"models/reinforce/reinforce_exp_{i}.pth", map_location="cpu")
        )
        policy.eval()

        score = evaluate_reinforce_model(policy, env, episodes=10)
        print(f"REINFORCE Exp {i}: {score:.2f}")
        results.append(("REINFORCE", i, score))
        env.close()

    best_overall = max(results, key=lambda x: x[2])

    print("\n===== BEST PER ALGORITHM =====")
    for algo in ["DQN", "PPO", "A2C", "REINFORCE"]:
        algo_results = [r for r in results if r[0] == algo]
        best = max(algo_results, key=lambda x: x[2])
        print(f"{algo}: Exp {best[1]} with reward {best[2]:.2f}")

    print("\n🏆 BEST OVERALL MODEL:")
    print(f"{best_overall[0]} Exp {best_overall[1]} with reward {best_overall[2]:.2f}")


if __name__ == "__main__":
    main()