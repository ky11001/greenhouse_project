import pickle
import torch
import os

def check_stats():
    print("========================================")
    print(" 📊 CHECKING: dataset_stats.pkl")
    print("========================================")
    if not os.path.exists('dataset_stats.pkl'):
        print("File not found!\n")
        return

    with open('dataset_stats.pkl', 'rb') as f:
        stats = pickle.load(f)

    for key, value in stats.items():
        # Usually contains qpos_mean, qpos_std, action_mean, action_std
        if hasattr(value, 'shape'): 
            # It's a numpy array or torch tensor
            length = value.shape[0] if len(value.shape) > 0 else 0
            
            # Add a visual flag if it's exactly 7
            flag = "✅ (7D Confirmed)" if length == 7 else f"⚠️ (Expected 7, got {length})"
            print(f"  {key:<15} | Shape: {str(value.shape):<10} {flag}")
        else:
            print(f"  {key:<15} | Type: {type(value)}")
    print("\n")

def check_checkpoint():
    print("========================================")
    print(" 🧠 CHECKING: policy_best.ckpt")
    print("========================================")
    if not os.path.exists('policy_best.ckpt'):
        print("File not found!\n")
        return

    # Load checkpoint safely to CPU
    ckpt = torch.load('policy_best.ckpt', map_location='cpu')
    
    # Check if the weights are nested under 'model_state_dict' (standard for ACT)
    state_dict = ckpt.get('model_state_dict', ckpt)

    # In ACT's DETR architecture, we look for the action output head and the state input
    # Typically named something like 'action_head.weight' or 'model.env_runner.action_head.weight'
    found_action = False
    
    for key, tensor in state_dict.items():
        # The final layer predicting the action
        if 'action_head.weight' in key or 'a_hat' in key or 'action_proj' in key:
            found_action = True
            out_features, in_features = tensor.shape
            flag = "✅ (7D Confirmed)" if out_features == 7 else f"⚠️ (Expected 7, got {out_features})"
            print(f"  Action Output Layer  ({key})")
            print(f"  --> Output features: {out_features} {flag}")
            print(f"  --> Input features:  {in_features}\n")
            
        # The first layer ingesting the qpos state
        if 'qpos_proj' in key and 'weight' in key:
            out_features, in_features = tensor.shape
            flag = "✅ (7D Confirmed)" if in_features == 7 else f"⚠️ (Expected 7, got {in_features})"
            print(f"  State Input Layer    ({key})")
            print(f"  --> Input features:  {in_features} {flag}")
            print(f"  --> Hidden features: {out_features}\n")

    if not found_action:
        print("  Could not automatically find the action_head layer names in this checkpoint.")
        print("  Here are some layer shapes for manual inspection:")
        # Just print the first few and last few layers
        keys = list(state_dict.keys())
        for k in keys[:5] + keys[-5:]:
            print(f"  {k}: {state_dict[k].shape}")

if __name__ == "__main__":
    check_stats()
    check_checkpoint()