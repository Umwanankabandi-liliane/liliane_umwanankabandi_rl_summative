import os
import matplotlib.pyplot as plt

os.makedirs("results/plots", exist_ok=True)

# Your real results
dqn = [360.25, 284.85, 437.60, 684.00, 706.50, 640.25, 315.25, 441.15, 667.20, 652.45]
ppo = [651.30, 656.60, 621.45, 625.50, 603.20, 638.00, 661.45, 610.40, 670.20, 641.40]
a2c = [639.05, 658.30, 681.65, 672.95, 549.60, 635.85, 640.80, 715.30, 621.00, 638.30]
reinforce = [351.10, 392.15, -13.15, 454.15, 252.55, 662.85, 553.50, 692.00, 616.15, 653.00]

experiments = list(range(1, 11))

# LINE PLOT (MAIN ONE YOU NEED)
plt.figure()
plt.plot(experiments, dqn, marker='o', label='DQN')
plt.plot(experiments, ppo, marker='o', label='PPO')
plt.plot(experiments, a2c, marker='o', label='A2C')
plt.plot(experiments, reinforce, marker='o', label='REINFORCE')

plt.title("Cumulative Reward Comparison")
plt.xlabel("Experiment")
plt.ylabel("Reward")
plt.legend()
plt.grid()

plt.savefig("results/plots/line_plot.png")
plt.show()

# BAR PLOT (SECOND ONE)
best_scores = [max(dqn), max(ppo), max(a2c), max(reinforce)]
labels = ["DQN", "PPO", "A2C", "REINFORCE"]

plt.figure()
plt.bar(labels, best_scores)

plt.title("Best Reward per Algorithm")
plt.ylabel("Reward")

plt.savefig("results/plots/bar_plot.png")
plt.show()

print("Plots saved in results/plots/")