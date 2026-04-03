import os
import sys
import time
import pygame

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import A2C
from environment.custom_env import HospitalTriageEnv
from environment.rendering import HospitalRenderer


ACTION_NAMES = {
    0: "Treat Bed 1",
    1: "Treat Bed 2",
    2: "Treat Bed 3",
    3: "Treat Bed 4",
    4: "Wait",
}


def run_demo(episodes=2):
    model = A2C.load("models/a2c/a2c_exp_8")
    renderer = HospitalRenderer()

    print("\n--- Running Best Agent Demo (A2C Exp 8) ---\n")
    print("Controls:")
    print("  SPACE = start / resume")
    print("  P = pause")
    print("  N = skip to next episode")
    print("  ESC = quit\n")

    paused = True
    quit_demo = False

    for ep in range(1, episodes + 1):
        if quit_demo:
            break

        env = HospitalTriageEnv(max_steps=120, max_losses=10)

        obs, info = env.reset()
        done = False
        total_reward = 0.0
        last_action = "Paused - Press SPACE to start"

        print(f"\n===== Episode {ep} =====\n")

        while not done:
            skip_episode = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit_demo = True
                    done = True
                    break

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = False
                        last_action = "Running"
                        print("Simulation started/resumed.")
                    elif event.key == pygame.K_p:
                        paused = True
                        last_action = "Paused"
                        print("Simulation paused.")
                    elif event.key == pygame.K_n:
                        skip_episode = True
                        print("Skipping to next episode.")
                    elif event.key == pygame.K_ESCAPE:
                        quit_demo = True
                        done = True
                        break

            if quit_demo:
                break

            if skip_episode:
                break

            if paused:
                renderer.render(
                    env.beds,
                    env.nurse_position,
                    info,
                    last_action=last_action
                )
                time.sleep(0.03)
                continue

            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            last_action = ACTION_NAMES[action]

            print(f"Action: {ACTION_NAMES[action]} | Reward: {reward:.2f}")

            for _ in range(14):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_demo = True
                        done = True
                        break
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_p:
                            paused = True
                            last_action = "Paused"
                            print("Simulation paused.")
                        elif event.key == pygame.K_ESCAPE:
                            quit_demo = True
                            done = True
                            break

                renderer.render(
                    env.beds,
                    env.nurse_position,
                    info,
                    last_action=last_action
                )
                time.sleep(0.04)

                if paused or quit_demo:
                    break

            total_reward += reward
            done = done or terminated or truncated

        print(f"\nEpisode {ep} Total Reward: {total_reward:.2f}")
        print("Episode finished.\n")

        env.close()
        time.sleep(1)

    renderer.close()
    print("Demo finished.")


if __name__ == "__main__":
    run_demo()