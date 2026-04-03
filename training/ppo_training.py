import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import HospitalTriageEnv


experiments = [
    {"lr": 3e-4, "gamma": 0.99, "n_steps": 128, "ent_coef": 0.01},
    {"lr": 1e-4, "gamma": 0.99, "n_steps": 128, "ent_coef": 0.01},
    {"lr": 5e-4, "gamma": 0.99, "n_steps": 128, "ent_coef": 0.01},
    {"lr": 3e-4, "gamma": 0.95, "n_steps": 128, "ent_coef": 0.01},
    {"lr": 1e-4, "gamma": 0.95, "n_steps": 128, "ent_coef": 0.01},
    {"lr": 5e-4, "gamma": 0.95, "n_steps": 128, "ent_coef": 0.01},
    {"lr": 3e-4, "gamma": 0.99, "n_steps": 256, "ent_coef": 0.02},
    {"lr": 1e-4, "gamma": 0.99, "n_steps": 256, "ent_coef": 0.02},
    {"lr": 5e-4, "gamma": 0.99, "n_steps": 256, "ent_coef": 0.02},
    {"lr": 2e-4, "gamma": 0.99, "n_steps": 256, "ent_coef": 0.005},
]


def run_ppo_experiments():
    os.makedirs("models/ppo", exist_ok=True)
    os.makedirs("results/logs/training/ppo", exist_ok=True)
    os.makedirs("results/tensorboard/ppo", exist_ok=True)

    for i, p in enumerate(experiments, start=1):
        print(f"\nRunning PPO Experiment {i}/10")
        print(p)

        env = HospitalTriageEnv(max_steps=100, max_losses=3)
        env = Monitor(env, filename=f"results/logs/training/ppo/ppo_exp_{i}_monitor.csv")

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=p["lr"],
            gamma=p["gamma"],
            n_steps=p["n_steps"],
            ent_coef=p["ent_coef"],
            batch_size=64,
            tensorboard_log="results/tensorboard/ppo",
            verbose=1,
        )

        model.learn(total_timesteps=100000, tb_log_name=f"ppo_exp_{i}")
        model.save(f"models/ppo/ppo_exp_{i}")

        env.close()

    print("\nAll PPO experiments completed!")


if __name__ == "__main__":
    run_ppo_experiments()