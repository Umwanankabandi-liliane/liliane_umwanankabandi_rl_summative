import os
import json

os.makedirs("results/logs", exist_ok=True)

results = {
    "DQN": [360.25, 284.85, 437.60, 684.00, 706.50, 640.25, 315.25, 441.15, 667.20, 652.45],
    "PPO": [651.30, 656.60, 621.45, 625.50, 603.20, 638.00, 661.45, 610.40, 670.20, 641.40],
    "A2C": [639.05, 658.30, 681.65, 672.95, 549.60, 635.85, 640.80, 715.30, 621.00, 638.30],
    "REINFORCE": [351.10, 392.15, -13.15, 454.15, 252.55, 662.85, 553.50, 692.00, 616.15, 653.00],
}

log_data = {}

# Build structured logs
for algo, scores in results.items():
    best_score = max(scores)
    best_exp = scores.index(best_score) + 1
    avg_score = sum(scores) / len(scores)

    log_data[algo] = {
        "experiments": scores,
        "best_experiment": best_exp,
        "best_score": best_score,
        "average_score": round(avg_score, 2),
    }

# Save JSON log
with open("results/logs/final_logs.json", "w") as f:
    json.dump(log_data, f, indent=4)

print("Saved: results/logs/final_logs.json")


# Save readable TXT log
with open("results/logs/final_logs.txt", "w") as f:
    for algo, data in log_data.items():
        f.write(f"\n===== {algo} =====\n")
        for i, score in enumerate(data["experiments"], start=1):
            f.write(f"{algo} Exp {i}: {score:.2f}\n")

        f.write(f"\nBest: Exp {data['best_experiment']} ({data['best_score']:.2f})\n")
        f.write(f"Average: {data['average_score']:.2f}\n")

print("Saved: results/logs/final_logs.txt")