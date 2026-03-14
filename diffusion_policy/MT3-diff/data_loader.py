import torch
from torch.utils.data import Dataset

class GreenhouseDataset(Dataset):
    def __init__(self, data_path, phase='alignment'):
        self.data = torch.load(data_path)
        self.phase = phase

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        episode = self.data[idx]
        split_idx = int(len(episode['actions']) * 0.8)
        
        if self.phase == 'alignment':
            return {
                'obs': episode['visual_features'][:split_idx],
                'actions': episode['actions'][:split_idx]
            }
        else:
            return {
                'obs': episode['visual_features'][split_idx:],
                'actions': episode['actions'][split_idx:]
            }