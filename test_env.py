import os
import sys
import time
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env import HospitalTriageEnv
from environment.rendering import HospitalRenderer


ACTION_NAMES = {
    0: "Treat Bed 1",
    1: "Treat Bed 2",
    2: "Treat Bed 3",
    3: "Treat Bed 4",
    4: "Wait",
}


def main():
    env = HospitalTriageEnv(max_steps=30, max_losses=3)
    renderer = HospitalRenderer()

    obs, info = env.reset()
    done = False

    renderer.render(env.beds, env.nurse_position, info, last_action="Start")
    time.sleep(1)

    while not done:
        action = random.randint(0, 4)
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Action: {ACTION_NAMES[action]} | Reward: {reward:.2f} | Info: {info}")

        for _ in range(8):
            renderer.render(
                env.beds,
                env.nurse_position,
                info,
                last_action=ACTION_NAMES[action]
            )
            time.sleep(0.03)

        done = terminated or truncated

    time.sleep(2)
    renderer.close()
    env.close()
    print("Simulation finished.")


if __name__ == "__main__":
    main()