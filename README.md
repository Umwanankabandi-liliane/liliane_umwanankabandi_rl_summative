# Liliane Umwanankabandi RL Summative

Mission-based reinforcement learning project for a custom hospital triage simulation.

## Project Overview
This project compares value-based and policy-based RL methods on the same custom environment:
- DQN (value-based)
- REINFORCE (policy gradient)
- PPO (policy gradient)
- A2C (actor-critic)

The agent learns how to prioritize treatment decisions for multiple beds while balancing:
- patient severity,
- waiting-time penalties,
- deterioration risk,
- and episode-level mission objectives.

## Environment Summary
Custom Gymnasium environment: `HospitalTriageEnv` in `environment/custom_env.py`.

Action space (`Discrete(5)`):
- `0`: Treat Bed 1
- `1`: Treat Bed 2
- `2`: Treat Bed 3
- `3`: Treat Bed 4
- `4`: Wait

Observation space:
- Bed-level features (for each of 4 beds):
	- active
	- severity
	- waiting_time
	- treatment_remaining
	- deterioration_counter
	- time_to_deteriorate
- Global features:
	- nurse_position
	- current_step
	- patients_treated
	- patients_lost

Reward design highlights:
- Positive rewards for successful treatments (higher for severe/urgent patients)
- Penalties for waiting and deterioration
- Strong penalties for patient loss
- Mild penalty for waiting when action is not appropriate

Termination conditions:
- `terminated` when patient losses reach `max_losses`
- `truncated` when `current_step` reaches `max_steps`

## Visualization
Pygame-based rendering is implemented in `environment/rendering.py`.

Random action demo (no trained model):
- `environment/random_demo.py`

Best-agent interactive demo:
- `main.py`

Controls in `main.py` demo:
- `SPACE`: start/resume
- `P`: pause
- `N`: skip episode
- `ESC`: quit

## Repository Structure
```
project_root/
|- environment/
|  |- custom_env.py
|  |- random_demo.py
|  |- rendering.py
|- training/
|  |- dqn_training.py
|  |- reinforce_training.py
|  |- ppo_training.py
|  |- a2c_training.py
|- evaluation/
|  |- final_evaluation.py
|  |- dqn_improved_evaluation.py
|- models/
|  |- dqn/
|  |- dqn_improved/
|  |- reinforce/
|  |- ppo/
|  |- a2c/
|- results/
|  |- logs/
|  |- plots/
|  |- tables/
|- main.py
|- test_env.py
|- requirements.txt
```

## Setup
1. Clone the repository.
2. Create and activate a virtual environment.
3. Install dependencies.

```bash
pip install -r requirements.txt
```

Note: make sure `requirements.txt` contains all required packages for your environment.

## How To Run
Run random environment demo:
```bash
python environment/random_demo.py
```

Run trained best-model demo:
```bash
python main.py
```

Run training scripts:
```bash
python training/dqn_training.py
python training/reinforce_training.py
python training/ppo_training.py
python training/a2c_training.py
```

Run evaluation scripts:
```bash
python evaluation/final_evaluation.py
python evaluation/dqn_improved_evaluation.py
```

## Experiments
Each algorithm includes 10 hyperparameter runs to support objective comparison and tuning analysis.

Trained models are saved under:
- `models/dqn/`
- `models/dqn_improved/`
- `models/reinforce/`
- `models/ppo/`
- `models/a2c/`

Generated outputs are stored in:
- `results/logs/`
- `results/plots/`
- `results/tables/`

