import torch
import torch.nn as nn
from diffusers import DDPMScheduler

class DiffusionAlignment(nn.Module):
    def __init__(self, action_dim=7, obs_dim=512):
        super().__init__()
        self.obs_encoder = nn.Linear(obs_dim, 256)
        self.noise_pred_net = nn.Sequential(
            nn.Linear(action_dim + 256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=100)

    def forward(self, noisy_action, timestep, obs_feat):
        obs_encoded = self.obs_encoder(obs_feat)
        inputs = torch.cat([noisy_action, obs_encoded], dim=-1)
        return self.noise_pred_net(inputs)

    @torch.no_grad()
    def sample(self, obs_feat, horizon=16):
        device = obs_feat.device
        action = torch.randn((1, horizon, 7), device=device)
        self.scheduler.set_timesteps(50)
        
        for t in self.scheduler.timesteps:
            model_output = self.forward(action, t, obs_feat)
            action = self.scheduler.step(model_output, t, action).prev_sample
        return action