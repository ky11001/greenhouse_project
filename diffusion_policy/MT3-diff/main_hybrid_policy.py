import torch
from model_diffusion import DiffusionAlignment
from mt3_retrieval import MT3Interaction

def run_greenhouse_task(robot, diff_path, demo_path):
    aligner = DiffusionAlignment().cuda().eval()
    aligner.load_state_dict(torch.load(diff_path))
    
    interactor = MT3Interaction(torch.load(demo_path))
    
    while True:
        obs = robot.get_observation() 
        
        print("Running Diffusion Alignment...")
        alignment_traj = aligner.sample(obs['visual_feature'])
        robot.execute(alignment_traj)
        
        print("Switching to MT3 Interaction...")
        interaction_traj = interactor.get_best_interaction(obs['visual_feature'])
        robot.execute(interaction_traj)
        
        if robot.check_grasp_success():
            print("Tomato Harvested!")
            break
        else:
            print("Grasp failed. Retrying alignment...")
            robot.reset_to_pre_grasp() 