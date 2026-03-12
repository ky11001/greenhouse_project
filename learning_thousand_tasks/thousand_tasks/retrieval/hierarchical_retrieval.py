import glob
from os.path import join, basename, normpath, exists, isdir
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from thousand_tasks.core.globals import TASKS_DIR, ASSETS_DIR
from thousand_tasks.core.utils.scene_state import SceneState
from thousand_tasks.retrieval.auto_encoder import load_encoder, get_pcd_embd
from thousand_tasks.retrieval.language_based_retrieval import LanguageBasedRetrieval
from thousand_tasks.data.utils import load_demo_scene_state


class HierarchicalRetrieval:
    def __init__(self,
                 T_WC_demo: np.ndarray,
                 learned_tasks_dir=None,
                 T_WC_live: np.ndarray = None,
                 ):

        if learned_tasks_dir is None:
            learned_tasks_dir = TASKS_DIR

        non_task_folder_dirs = ['interaction_processed', 'bn_reaching_processed', 'processed']

        self.T_WC_demo = T_WC_demo
        self.T_WC_live = T_WC_live if T_WC_live is not None else T_WC_demo
        # Geometry encoder is in assets/
        self.model_path = ASSETS_DIR
        self.root_dir = learned_tasks_dir
        self.tasks_folder_names = np.sort(
            [basename(normpath(task)) for task in glob.glob(join(self.root_dir, '*')) if
             (isdir(task) and basename(normpath(task)) not in non_task_folder_dirs)]).tolist()

        self.encoder = load_encoder(self.model_path)
        self.encoder = self.encoder.to('cuda:0')
        self.encoder.eval()
        self.language_based_retrieval = LanguageBasedRetrieval(learned_tasks_dir=self.root_dir, verbose=False)

        self._load_task_embeddings()

    def _load_task_embeddings(self):

        self.embeddings = {}
        iterable = tqdm(self.tasks_folder_names)
        for task_folder_name in iterable:
            # try:
            iterable.set_description('Loading embeddings of learned tasks...', refresh=False)
            encoding_path = join(self.root_dir, task_folder_name, 'geometry_encoding.npy')
            if exists(encoding_path):
                self.embeddings[task_folder_name] = np.load(encoding_path)
            else:
                scene_state = load_demo_scene_state(task_name=task_folder_name,
                                                    load_segmap_if_exists=True,
                                                    learned_tasks_dir=self.root_dir)
                assert scene_state.segmap_was_set, f'Segmentation failed to load for task {task_folder_name}'
                scene_state.T_WC = self.T_WC_demo
                try:
                    with torch.no_grad():
                        embedding = get_pcd_embd(self.encoder, scene_state)
                except Exception as e:
                    print(task_folder_name)
                    print()
                    print(e)

                np.save(encoding_path, embedding)
                self.embeddings[task_folder_name] = embedding


    def get_task_embeddings(self, task_names: List[str]):
        embeddings = []
        for task_name in task_names:
            embeddings.append(self.embeddings[task_name])

        return np.array(embeddings)

    def get_most_similar_demo_name(self, scene_state: SceneState, template_task_description: str) -> str:

        candidate_tasks = self.language_based_retrieval.retrieve_relevant_tasks(template_task_description)
        print(candidate_tasks is None)
        if len(candidate_tasks) == 0:
            print(f'No demonstrations exist for skill {template_task_description}')
        elif len(candidate_tasks) == 1:
            return candidate_tasks[0]
        else:
            candidate_embeddings = self.get_task_embeddings(candidate_tasks)

            scene_state.T_WC = self.T_WC_live

            with torch.no_grad():
                test_embedding = np.expand_dims(get_pcd_embd(self.encoder, scene_state), 0)

            # Option 1: Cosine similarity

            test_embedding = test_embedding / np.linalg.norm(test_embedding)
            candidate_embeddings /= np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
            similarity = test_embedding @ candidate_embeddings.T
            closest_task = candidate_tasks[np.argmax(similarity)]

            return closest_task
