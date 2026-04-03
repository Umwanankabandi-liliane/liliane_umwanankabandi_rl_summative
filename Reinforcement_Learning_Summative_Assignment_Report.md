# Reinforcement Learning Summative Assignment Report

Student Name: [Your Name]  
Video Recording: [Link to your Video (3 minutes max, Camera On, Share entire Screen)]  
GitHub Repository: [Link to your repository]

## Project Overview
This project applies reinforcement learning (RL) to a hospital emergency-room triage problem. In a busy emergency room, treatment order is critical because patients can deteriorate while waiting. I designed a custom simulation environment where an RL agent acts as a nurse and learns which patient bed to treat at each decision step.

Four RL methods were trained and compared:
- Deep Q-Network (DQN)
- Proximal Policy Optimization (PPO)
- Advantage Actor-Critic (A2C)
- REINFORCE

The objective was to maximize cumulative reward by prioritizing urgent patients, reducing delays, and minimizing patient loss.

## Environment Description
### Agent(s)
The single agent represents a nurse in the emergency room. At each step, it decides whether to treat one of four beds or wait.

### Action Space
The action space is discrete with 5 actions:
1. Treat Bed 1
2. Treat Bed 2
3. Treat Bed 3
4. Treat Bed 4
5. Wait

### Observation Space
The state is a numeric vector describing all beds plus global status. For each bed, the environment provides:
- Active flag (patient present or not)
- Severity level (0 = mild, 1 = moderate, 2 = critical)
- Waiting time
- Remaining treatment time
- Deterioration counter
- Time to next deterioration

Global values also include:
- Nurse position
- Current step
- Number of patients treated
- Number of patients lost

### Reward Structure
The reward function balances urgency, speed, and safety:
- Positive reward for treating patients, with higher reward for higher severity
- Extra bonus for treating critical patients quickly
- Small penalty for waiting
- Stronger waiting penalty when critical patients are waiting
- Penalty for delaying moderate/critical waiting patients each step
- Additional penalty when a patient deteriorates
- Large negative penalty when a critical patient is lost
- Penalty for trying to treat an empty bed

This design encourages the agent to prioritize critical and deteriorating patients while avoiding unnecessary delays.

## System Analysis And Design
### Deep Q-Network (DQN)
DQN learns an action-value function Q(s, a), estimating expected return for each action in each state. It uses:
- A neural network policy (MlpPolicy)
- Experience replay via replay buffer
- Epsilon-greedy exploration schedule
- Target network updates for stability

### Policy Gradient Methods (REINFORCE / PPO / A2C)
These methods learn policies directly:
- REINFORCE: Monte Carlo policy gradient using full episodes and return normalization
- PPO: Clipped policy updates for safer, stable improvement
- A2C: Actor-critic method using value baseline for lower-variance updates

All policy-gradient methods used neural networks to map state observations to action probabilities.

## Implementation
### DQN (10 Experiments)
| Experiment | Learning Rate | Gamma | Replay Buffer Size | Batch Size | Exploration Strategy | Mean Reward |
|---|---:|---:|---:|---:|---|---:|
| 1 | 0.0010 | 0.99 | 20000 | 32 | exploration_fraction=0.30, eps: 1.0→0.05 | 360.25 |
| 2 | 0.0005 | 0.99 | 20000 | 32 | exploration_fraction=0.30, eps: 1.0→0.05 | 284.85 |
| 3 | 0.0001 | 0.99 | 20000 | 32 | exploration_fraction=0.30, eps: 1.0→0.05 | 437.60 |
| 4 | 0.0010 | 0.95 | 20000 | 64 | exploration_fraction=0.25, eps: 1.0→0.05 | 684.00 |
| 5 | 0.0005 | 0.95 | 20000 | 64 | exploration_fraction=0.25, eps: 1.0→0.05 | 706.50 |
| 6 | 0.0001 | 0.95 | 20000 | 64 | exploration_fraction=0.25, eps: 1.0→0.05 | 640.25 |
| 7 | 0.0005 | 0.99 | 20000 | 64 | exploration_fraction=0.20, eps: 1.0→0.05 | 315.25 |
| 8 | 0.0010 | 0.99 | 20000 | 64 | exploration_fraction=0.20, eps: 1.0→0.05 | 441.15 |
| 9 | 0.0005 | 0.90 | 20000 | 32 | exploration_fraction=0.30, eps: 1.0→0.05 | 667.20 |
| 10 | 0.0001 | 0.90 | 20000 | 32 | exploration_fraction=0.30, eps: 1.0→0.05 | 652.45 |

### REINFORCE (10 Experiments)
| Experiment | Learning Rate | Gamma | Episodes | Network Architecture | Mean Reward |
|---|---:|---:|---:|---|---:|
| 1 | 0.0010 | 0.99 | 600 | 26→128→64→5 (Softmax) | 351.10 |
| 2 | 0.0005 | 0.99 | 600 | 26→128→64→5 (Softmax) | 392.15 |
| 3 | 0.0001 | 0.99 | 600 | 26→128→64→5 (Softmax) | -13.15 |
| 4 | 0.0020 | 0.99 | 600 | 26→128→64→5 (Softmax) | 454.15 |
| 5 | 0.0008 | 0.99 | 600 | 26→128→64→5 (Softmax) | 252.55 |
| 6 | 0.0006 | 0.95 | 600 | 26→128→64→5 (Softmax) | 662.85 |
| 7 | 0.0003 | 0.95 | 600 | 26→128→64→5 (Softmax) | 553.50 |
| 8 | 0.0007 | 0.95 | 600 | 26→128→64→5 (Softmax) | 692.00 |
| 9 | 0.0009 | 0.90 | 600 | 26→128→64→5 (Softmax) | 616.15 |
| 10 | 0.0004 | 0.90 | 600 | 26→128→64→5 (Softmax) | 653.00 |

### PPO (10 Experiments)
| Experiment | Learning Rate | Gamma | n_steps | Entropy Coef | Batch Size | Mean Reward |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.0003 | 0.99 | 128 | 0.010 | 64 | 651.30 |
| 2 | 0.0001 | 0.99 | 128 | 0.010 | 64 | 656.60 |
| 3 | 0.0005 | 0.99 | 128 | 0.010 | 64 | 621.45 |
| 4 | 0.0003 | 0.95 | 128 | 0.010 | 64 | 625.50 |
| 5 | 0.0001 | 0.95 | 128 | 0.010 | 64 | 603.20 |
| 6 | 0.0005 | 0.95 | 128 | 0.010 | 64 | 638.00 |
| 7 | 0.0003 | 0.99 | 256 | 0.020 | 64 | 661.45 |
| 8 | 0.0001 | 0.99 | 256 | 0.020 | 64 | 610.40 |
| 9 | 0.0005 | 0.99 | 256 | 0.020 | 64 | 670.20 |
| 10 | 0.0002 | 0.99 | 256 | 0.005 | 64 | 641.40 |

### A2C (10 Experiments)
| Experiment | Learning Rate | Gamma | n_steps | Entropy Coef | Mean Reward |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0007 | 0.99 | 5 | 0.010 | 639.05 |
| 2 | 0.0010 | 0.99 | 5 | 0.010 | 658.30 |
| 3 | 0.0005 | 0.99 | 5 | 0.010 | 681.65 |
| 4 | 0.0007 | 0.95 | 5 | 0.010 | 672.95 |
| 5 | 0.0010 | 0.95 | 5 | 0.010 | 549.60 |
| 6 | 0.0005 | 0.95 | 5 | 0.010 | 635.85 |
| 7 | 0.0007 | 0.99 | 10 | 0.020 | 640.80 |
| 8 | 0.0010 | 0.99 | 10 | 0.020 | 715.30 |
| 9 | 0.0005 | 0.99 | 10 | 0.020 | 621.00 |
| 10 | 0.0008 | 0.99 | 10 | 0.005 | 638.30 |

## Results Discussion
### Cumulative Rewards
Insert the cumulative reward comparison figure from your generated plots:
- Line comparison of all methods across 10 experiments

Interpretation:
- A2C achieved the highest best run (715.30) and the highest mean score (645.28).
- PPO had slightly lower peak than A2C but very consistent performance with tight spread.
- DQN produced strong top runs but large variation between experiments.
- REINFORCE showed the highest instability, ranging from negative reward to strong high-performing runs.

### Training Stability
Insert stability visualizations:
- Reward line comparison plot
- Boxplot comparison plot
- Best-per-algorithm bar chart

Quantitative stability summary (from final experiment rewards):
- DQN: mean 518.95, std 167.03, range 421.65
- PPO: mean 637.95, std 22.42, range 67.00
- A2C: mean 645.28, std 43.59, range 165.70
- REINFORCE: mean 461.43, std 223.29, range 705.15

Interpretation:
- PPO is the most stable algorithm (lowest standard deviation and smallest range).
- A2C balances strong performance and acceptable stability.
- DQN and REINFORCE are much more sensitive to hyperparameters in this environment.

Note on objective/entropy curves:
- Detailed per-episode objective-function curves (for DQN) and entropy curves (for policy-gradient methods) were not saved in the current logs, so this section is interpreted from final cross-experiment distributions.

### Episodes To Converge
Because per-episode logs were not recorded, exact convergence episode counts cannot be measured directly.

However, training budget and final consistency suggest:
- DQN, PPO, and A2C were each trained for 100000 timesteps per experiment.
- REINFORCE was trained for 600 episodes per experiment.
- PPO and A2C appear to converge more reliably within the given training budget.
- DQN and REINFORCE can converge to high scores, but convergence is less consistent across settings.

### Generalization
Generalization was tested by evaluating trained models over multiple episodes with fresh environment resets (unseen initial patient placements and severity combinations due stochastic spawning and random initialization).

Findings:
- A2C and PPO generalized best, maintaining high rewards across varied resets.
- DQN generalized moderately, but performance depended strongly on the chosen hyperparameter setting.
- REINFORCE generalized inconsistently, with both very strong and very weak runs.

## Conclusion and Discussion
This project shows that reinforcement learning can effectively learn triage decisions in a hospital simulation.

Main conclusions:
- Best single model: A2C Experiment 8 (715.30)
- Best average performance: A2C (645.28)
- Best stability: PPO (std 22.42)
- Most sensitive methods: DQN and REINFORCE

Strengths and weaknesses in this environment:
- DQN
  - Strength: Can reach high rewards with tuned settings
  - Weakness: High variance and sensitivity to exploration and discount settings
- PPO
  - Strength: Very stable and reliable across experiments
  - Weakness: Slightly lower peak reward than the very best A2C run
- A2C
  - Strength: Best overall trade-off between peak and average reward
  - Weakness: More variance than PPO
- REINFORCE
  - Strength: Simple implementation and occasional high scores
  - Weakness: Highest instability and occasional failure cases

Possible improvements with more time/resources:
- Log per-episode rewards, losses, and entropy for deeper convergence analysis
- Run multiple random seeds per experiment for statistically stronger conclusions
- Add prioritized replay or double DQN variants for DQN stability
- Tune network size and regularization systematically for policy-gradient methods
- Extend environment realism (resource constraints, variable nurse count, patient categories)
