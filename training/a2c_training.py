import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import HospitalTriageEnv


experiments = [
    {"lr": 7e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.01},
    {"lr": 1e-3, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.01},
    {"lr": 5e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.01},
    {"lr": 7e-4, "gamma": 0.95, "n_steps": 5, "ent_coef": 0.01},
    {"lr": 1e-3, "gamma": 0.95, "n_steps": 5, "ent_coef": 0.01},
    {"lr": 5e-4, "gamma": 0.95, "n_steps": 5, "ent_coef": 0.01},
    {"lr": 7e-4, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.02},
    {"lr": 1e-3, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.02},
    {"lr": 5e-4, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.02},
    {"lr": 8e-4, "gamma": 0.99, "n_steps": 10, "ent_coef": 0.005},
]


def run_a2c_experiments():
    os.makedirs("models/a2c", exist_ok=True)
    os.makedirs("results/logs/training/a2c", exist_ok=True)
    os.makedirs("results/tensorboard/a2c", exist_ok=True)

    for i, p in enumerate(experiments, start=1):
        print(f"\nRunning A2C Experiment {i}/10")
        print(p)

        env = HospitalTriageEnv(max_steps=100, max_losses=3)
        env = Monitor(env, filename=f"results/logs/training/a2c/a2c_exp_{i}_monitor.csv")

        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=p["lr"],
            gamma=p["gamma"],
            n_steps=p["n_steps"],
            ent_coef=p["ent_coef"],
            tensorboard_log="results/tensorboard/a2c",
            verbose=1,
        )

        model.learn(total_timesteps=100000, tb_log_name=f"a2c_exp_{i}")
        model.save(f"models/a2c/a2c_exp_{i}")

        env.close()

    print("\nAll A2C experiments completed!")


if __name__ == "__main__":
    run_a2c_experiments()