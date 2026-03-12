import os

# Point this to where your demos live
DEMO_ROOT = "assets/demonstrations"

def fix_all():
    # Find all folders that look like tasks
    tasks = [d for d in os.listdir(DEMO_ROOT) if os.path.isdir(os.path.join(DEMO_ROOT, d))]
    
    count = 0
    for task in tasks:
        task_path = os.path.join(DEMO_ROOT, task)
        desc_path = os.path.join(task_path, "task_description.txt")
        
        # Only create if missing
        if not os.path.exists(desc_path):
            with open(desc_path, "w") as f:
                # We give a generic description for the greenhouse task
                f.write("pick up the tomato from plant and once you grab it, stop")
            count += 1
            
    print(f"✅ Fixed: Created descriptions for {count} tasks.")

if __name__ == "__main__":
    if os.path.exists(DEMO_ROOT):
        fix_all()
    else:
        print(f"❌ Error: Could not find folder {DEMO_ROOT}")