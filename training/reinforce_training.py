import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def compute_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)

    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return returns


def train_reinforce(exp_id, lr, gamma):
    env = HospitalTriageEnv(max_steps=100, max_losses=3)
    os.makedirs("results/logs/training/reinforce", exist_ok=True)
    log_path = f"results/logs/training/reinforce/reinforce_exp_{exp_id}_training.csv"

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    num_episodes = 600

    with open(log_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "episode_reward", "loss", "policy_entropy"])

        for episode in range(num_episodes):
            log_probs = []
            rewards = []
            entropies = []

            state, _ = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                probs = policy(state_tensor)

                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropies.append(float(dist.entropy().item()))

                next_state, reward, terminated, truncated, _ = env.step(action.item())

                log_probs.append(log_prob)
                rewards.append(reward)
                episode_reward += reward

                state = next_state
                done = terminated or truncated

            returns = compute_returns(rewards, gamma)

            loss = 0
            for log_prob, G in zip(log_probs, returns):
                loss += -log_prob * G

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_entropy = float(np.mean(entropies)) if entropies else 0.0
            writer.writerow(
                [episode + 1, round(episode_reward, 4), round(float(loss.item()), 6), round(avg_entropy, 6)]
            )

            if (episode + 1) % 100 == 0:
                print(f"Exp {exp_id} | Episode {episode+1}/{num_episodes} completed")

    os.makedirs("models/reinforce", exist_ok=True)
    torch.save(policy.state_dict(), f"models/reinforce/reinforce_exp_{exp_id}.pth")
    env.close()


experiments = [
    {"lr": 1e-3, "gamma": 0.99},
    {"lr": 5e-4, "gamma": 0.99},
    {"lr": 1e-4, "gamma": 0.99},
    {"lr": 2e-3, "gamma": 0.99},
    {"lr": 8e-4, "gamma": 0.99},
    {"lr": 6e-4, "gamma": 0.95},
    {"lr": 3e-4, "gamma": 0.95},
    {"lr": 7e-4, "gamma": 0.95},
    {"lr": 9e-4, "gamma": 0.90},
    {"lr": 4e-4, "gamma": 0.90},
]


def run_all():
    for i, p in enumerate(experiments, start=1):
        print(f"\nRunning REINFORCE Experiment {i}/10")
        print(p)
        train_reinforce(i, p["lr"], p["gamma"])

    print("\nAll REINFORCE experiments completed!")


if __name__ == "__main__":
    run_all()