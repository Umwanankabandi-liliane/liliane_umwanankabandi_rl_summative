import os
import csv

os.makedirs("results/tables", exist_ok=True)

results = {
    "DQN": [360.25, 284.85, 437.60, 684.00, 706.50, 640.25, 315.25, 441.15, 667.20, 652.45],
    "PPO": [651.30, 656.60, 621.45, 625.50, 603.20, 638.00, 661.45, 610.40, 670.20, 641.40],
    "A2C": [639.05, 658.30, 681.65, 672.95, 549.60, 635.85, 640.80, 715.30, 621.00, 638.30],
    "REINFORCE": [351.10, 392.15, -13.15, 454.15, 252.55, 662.85, 553.50, 692.00, 616.15, 653.00],
}

with open("results/tables/final_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Algorithm", "Experiment", "Average Reward"])

    for algo, scores in results.items():
        for i, score in enumerate(scores, start=1):
            writer.writerow([algo, i, score])

print("Saved: results/tables/final_results.csv")