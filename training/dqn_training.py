import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import HospitalTriageEnv


experiments = [
    {"lr": 1e-3, "gamma": 0.99, "batch": 32, "explore": 0.30, "target": 500},
    {"lr": 5e-4, "gamma": 0.99, "batch": 32, "explore": 0.30, "target": 500},
    {"lr": 1e-4, "gamma": 0.99, "batch": 32, "explore": 0.30, "target": 500},
    {"lr": 1e-3, "gamma": 0.95, "batch": 64, "explore": 0.25, "target": 500},
    {"lr": 5e-4, "gamma": 0.95, "batch": 64, "explore": 0.25, "target": 500},
    {"lr": 1e-4, "gamma": 0.95, "batch": 64, "explore": 0.25, "target": 500},
    {"lr": 5e-4, "gamma": 0.99, "batch": 64, "explore": 0.20, "target": 1000},
    {"lr": 1e-3, "gamma": 0.99, "batch": 64, "explore": 0.20, "target": 1000},
    {"lr": 5e-4, "gamma": 0.90, "batch": 32, "explore": 0.30, "target": 1000},
    {"lr": 1e-4, "gamma": 0.90, "batch": 32, "explore": 0.30, "target": 1000},
]


def run_dqn_experiments():
    os.makedirs("models/dqn_improved", exist_ok=True)
    os.makedirs("results/logs/training/dqn", exist_ok=True)
    os.makedirs("results/tensorboard/dqn", exist_ok=True)

    for i, p in enumerate(experiments, start=1):
        print(f"\nRunning Improved DQN Experiment {i}/10")
        print(p)

        env = HospitalTriageEnv(max_steps=100, max_losses=3)
        env = Monitor(env, filename=f"results/logs/training/dqn/dqn_exp_{i}_monitor.csv")

        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=p["lr"],
            gamma=p["gamma"],
            batch_size=p["batch"],
            buffer_size=20000,
            learning_starts=1000,
            exploration_fraction=p["explore"],
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            target_update_interval=p["target"],
            train_freq=4,
            gradient_steps=1,
            tensorboard_log="results/tensorboard/dqn",
            verbose=1,
        )

        model.learn(total_timesteps=100000, tb_log_name=f"dqn_exp_{i}")
        model.save(f"models/dqn_improved/dqn_improved_exp_{i}")
        env.close()

    print("\nAll improved DQN experiments completed!")


if __name__ == "__main__":
    run_dqn_experiments()