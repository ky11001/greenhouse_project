import json
import os
import random

# CONFIG
DEMO_DIR = "assets/demonstrations"
OUTPUT_DIR = "assets/task_sets"

def create_split():
    # Get all 40 tasks
    all_tasks = sorted([f for f in os.listdir(DEMO_DIR) if f.startswith("demo_task_full_")])

    # Shuffle them so we get a mix of positions in Train vs Val
    random.seed(42) # Fixed seed so results are reproducible
    random.shuffle(all_tasks)

    # 90% Train (36 demos), 10% Validation (4 demos)
    split_idx = int(len(all_tasks) * 0.9)
    train_tasks = all_tasks[:split_idx]
    val_tasks = all_tasks[split_idx:]

    data = {
        "train": train_tasks,
        "validation": val_tasks
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "greenhouse_tasks.json")

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"✅ Created Task Set: {save_path}")
    print(f"   Train: {len(train_tasks)} | Val: {len(val_tasks)}")

if __name__ == "__main__":
    create_split()