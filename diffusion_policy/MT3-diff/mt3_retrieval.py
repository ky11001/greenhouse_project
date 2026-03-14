import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MT3Interaction:
    def __init__(self, demo_library):
        self.demos = demo_library # List of episodes
        self.embeddings = np.array([d['mean_feature'] for d in self.demos])

    def get_best_interaction(self, current_feature):
        query = current_feature.detach().cpu().numpy().reshape(1, -1)
        similarities = cosine_similarity(query, self.embeddings)
        best_idx = np.argmax(similarities)
        
        return self.demos[best_idx]['interaction_actions']