import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from environment.custom_env import HospitalTriageEnv


def evaluate_model(model_path, episodes=10):
    env = HospitalTriageEnv(max_steps=50, max_losses=3)
    model = DQN.load(model_path)

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

    env.close()
    return np.mean(rewards)


def main():
    results = []

    for i in range(1, 11):
        path = f"models/dqn_improved/dqn_improved_exp_{i}"
        score = evaluate_model(path)
        print(f"Improved DQN Exp {i}: {score:.2f}")
        results.append((i, score))

    best = max(results, key=lambda x: x[1])
    print(f"\nBest Improved DQN: Exp {best[0]} with reward {best[1]:.2f}")


if __name__ == "__main__":
    main()