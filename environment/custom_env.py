import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class HospitalTriageEnv(gym.Env):
    """
    Hospital triage RL environment.

    Actions:
        0 -> Treat Bed 1
        1 -> Treat Bed 2
        2 -> Treat Bed 3
        3 -> Treat Bed 4
        4 -> Wait

    Nurse positions for rendering:
        0 -> Nurse station
        1 -> Bed 1
        2 -> Bed 2
        3 -> Bed 3
        4 -> Bed 4
    """

    metadata = {"render_modes": ["human"], "render_fps": 6}

    def __init__(self, max_steps: int = 100, max_losses: int = 3):
        super().__init__()

        self.num_beds = 4
        self.max_steps = max_steps
        self.max_losses = max_losses

        self.action_space = spaces.Discrete(5)

        # Per bed:
        # active, severity, waiting_time, treatment_remaining,
        # deterioration_counter, time_to_deteriorate
        self.features_per_bed = 6
        obs_size = self.num_beds * self.features_per_bed + 4

        low = np.zeros(obs_size, dtype=np.float32)
        high = np.array(
            (
                [1, 2, 100, 10, 20, 20] * self.num_beds
                + [4, self.max_steps, 100, self.max_losses]
            ),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.beds: List[Optional[Dict[str, Any]]] = [None] * self.num_beds
        self.nurse_position = 0
        self.current_step = 0
        self.patients_treated = 0
        self.patients_lost = 0
        self.total_reward = 0.0

    def _treatment_time(self, severity: int) -> int:
        return {0: 1, 1: 2, 2: 3}[severity]

    def _time_to_deteriorate(self, severity: int, counter: int) -> int:
        thresholds = {0: 10, 1: 8, 2: 7}
        return max(0, thresholds[severity] - counter)

    def _create_patient(self) -> Dict[str, Any]:
        severity = random.choices([0, 1, 2], weights=[0.4, 0.35, 0.25], k=1)[0]
        return {
            "active": 1,
            "severity": severity,
            "waiting_time": 0,
            "treatment_remaining": self._treatment_time(severity),
            "deterioration_counter": 0,
            "time_to_deteriorate": self._time_to_deteriorate(severity, 0),
            "status": "waiting",
        }

    def _spawn_initial_patients(self) -> None:
        self.beds = [None] * self.num_beds
        initial_patients = 2
        selected_beds = random.sample(range(self.num_beds), initial_patients)
        for idx in selected_beds:
            self.beds[idx] = self._create_patient()

    def _spawn_new_patients(self) -> None:
        for i in range(self.num_beds):
            if self.beds[i] is None:
                if random.random() < 0.15:
                    self.beds[i] = self._create_patient()

    def _update_waiting_and_deterioration(self) -> float:
        reward = 0.0

        for i in range(self.num_beds):
            bed = self.beds[i]
            if bed is None:
                continue

            if bed["status"] == "waiting":
                bed["waiting_time"] += 1
                bed["deterioration_counter"] += 1

                if bed["severity"] == 1:
                    reward -= 0.5
                elif bed["severity"] == 2:
                    reward -= 1.0

                severity = bed["severity"]
                counter = bed["deterioration_counter"]
                bed["time_to_deteriorate"] = self._time_to_deteriorate(severity, counter)

                # mild -> moderate after 10
                if severity == 0 and counter >= 10:
                    bed["severity"] = 1
                    bed["deterioration_counter"] = 0
                    bed["treatment_remaining"] = self._treatment_time(1)
                    bed["time_to_deteriorate"] = self._time_to_deteriorate(1, 0)
                    reward -= 6

                # moderate -> critical after 8
                elif severity == 1 and counter >= 8:
                    bed["severity"] = 2
                    bed["deterioration_counter"] = 0
                    bed["treatment_remaining"] = self._treatment_time(2)
                    bed["time_to_deteriorate"] = self._time_to_deteriorate(2, 0)
                    reward -= 6

                # critical lost after 7
                elif severity == 2 and counter >= 7:
                    self.beds[i] = None
                    self.patients_lost += 1
                    reward -= 15

        return reward

    def _treat_patient(self, bed_index: int) -> float:
        bed = self.beds[bed_index]
        self.nurse_position = bed_index + 1

        if bed is None:
            return -4.0

        reward = 2.0
        bed["status"] = "in_treatment"
        bed["treatment_remaining"] -= 1

        if bed["treatment_remaining"] <= 0:
            severity = bed["severity"]
            waiting_time = bed["waiting_time"]

            if severity == 0:
                reward += 15
            elif severity == 1:
                reward += 25
            else:
                reward += 40
                if waiting_time <= 2:
                    reward += 15

            self.patients_treated += 1
            self.beds[bed_index] = None
        else:
            # treatment stabilizes patient
            bed["deterioration_counter"] = max(0, bed["deterioration_counter"] - 1)
            bed["time_to_deteriorate"] = self._time_to_deteriorate(
                bed["severity"], bed["deterioration_counter"]
            )

        return reward

    def _handle_wait_action(self) -> float:
        self.nurse_position = 0
        critical_exists = any(
            bed is not None and bed["severity"] == 2 and bed["status"] == "waiting"
            for bed in self.beds
        )
        if critical_exists:
            return -4.0
        return -0.5

    def _get_obs(self) -> np.ndarray:
        obs: List[float] = []

        for bed in self.beds:
            if bed is None:
                obs.extend([0, 0, 0, 0, 0, 0])
            else:
                obs.extend(
                    [
                        bed["active"],
                        bed["severity"],
                        bed["waiting_time"],
                        bed["treatment_remaining"],
                        bed["deterioration_counter"],
                        bed["time_to_deteriorate"],
                    ]
                )

        obs.extend(
            [
                self.nurse_position,
                self.current_step,
                self.patients_treated,
                self.patients_lost,
            ]
        )

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "current_step": self.current_step,
            "patients_treated": self.patients_treated,
            "patients_lost": self.patients_lost,
            "total_reward": self.total_reward,
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.current_step = 0
        self.patients_treated = 0
        self.patients_lost = 0
        self.total_reward = 0.0
        self.nurse_position = 0

        self._spawn_initial_patients()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        reward = 0.0
        self.current_step += 1

        if action in [0, 1, 2, 3]:
            reward += self._treat_patient(action)
        elif action == 4:
            reward += self._handle_wait_action()
        else:
            reward -= 5.0

        reward += self._update_waiting_and_deterioration()
        self._spawn_new_patients()

        self.total_reward += reward

        terminated = self.patients_lost >= self.max_losses
        truncated = self.current_step >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        print(f"Step: {self.current_step}")
        print(f"Nurse position: {self.nurse_position}")
        print(f"Treated: {self.patients_treated} | Lost: {self.patients_lost}")
        for i, bed in enumerate(self.beds, start=1):
            print(f"Bed {i}: {bed}")
        print("-" * 50)

    def close(self):
        pass