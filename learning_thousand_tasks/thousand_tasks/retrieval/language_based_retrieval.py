import glob
from os.path import join, basename, normpath, isdir

import numpy as np

from thousand_tasks.core.globals import TASKS_DIR
from thousand_tasks.retrieval.language_template_parser import parse_task_folder_name, SKILL_NAMES


class LanguageBasedRetrieval:

    def __init__(self, learned_tasks_dir=TASKS_DIR, verbose=False):
        non_task_folder_dirs = ['interaction_processed', 'bn_reaching_processed', 'processed']
        self.root_dir = learned_tasks_dir
        self.tasks_folder_names = np.sort(
            [basename(normpath(task)) for task in glob.glob(join(self.root_dir, '*')) if
             (isdir(task) and basename(normpath(task)) not in non_task_folder_dirs)]).tolist()

        self._parse_learned_tasks(verbose)

    @staticmethod
    def _create_task_tag(tgt_obj_class, other_obj_class, other_obj_des):
        tag = f'tgt_obj_{str(tgt_obj_class)}_other_obj_{str(other_obj_des)}_{str(other_obj_class)}'
        return tag

    def _parse_learned_tasks(self, verbose=False):

        self.learned_skills = {}
        for skill_name in SKILL_NAMES:
            self.learned_skills[skill_name] = {}

        for folder_name in self.tasks_folder_names:
            skill_type, tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_folder_name(folder_name)

            task_tag = LanguageBasedRetrieval._create_task_tag(tgt_obj_class=tgt_obj_class,
                                                               other_obj_class=other_obj_class,
                                                               other_obj_des=other_obj_des)

            try:
                if task_tag not in list(self.learned_skills[skill_type].keys()):
                    self.learned_skills[skill_type][task_tag] = []
            except:
                print('Here')

            self.learned_skills[skill_type][task_tag].append(dict(task_dir=folder_name,
                                                                  tgt_obj_class=tgt_obj_class,
                                                                  tgt_obj_des=tgt_obj_des,
                                                                  other_obj_class=other_obj_class,
                                                                  other_obj_des=other_obj_des))

            if verbose:
                print(
                    f'skill: {skill_type} - tgt obj: ({tgt_obj_des}) {tgt_obj_class} - other obj: ({other_obj_des}) {other_obj_class}')

    def retrieve_relevant_tasks(self, task_name):
        # if task_name in self.tasks_folder_names:
        #     return [task_name]
        # else:
        skill_type, tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_folder_name(task_name)
        task_tag = LanguageBasedRetrieval._create_task_tag(tgt_obj_class=tgt_obj_class,
                                                           other_obj_class=other_obj_class,
                                                           other_obj_des=other_obj_des)
        try:
            relevant_tasks = self.learned_skills[skill_type][task_tag]
        except KeyError as e:
            print(f'\n\nNo tasks have been taught for skill {skill_type} with task tag {task_tag}\n\n')
            relevant_tasks = []

        relevant_task_dirs = [task_name['task_dir'] for task_name in relevant_tasks]

        return relevant_task_dirs


if __name__ == '__main__':
    retrieval = LanguageBasedRetrieval(learned_tasks_dir='/mnt/data/kamil/1000_tasks', verbose=False)

    print(retrieval.retrieve_relevant_tasks('insert large square plate into rack'.replace(' ', '_')))

    # print(retrieval.retrieve_relevant_tasks('pick_up_mug'))

    # from create_final_task_catalogue import load_task_catalogue
    #
    # test_tasks = load_task_catalogue('/home/kamil/MT3/final_test.txt')
    #
    # for task in test_tasks.keys():
    #     try:
    #         retrieval.retrieve_relevant_tasks(task)
    #         # print(f'{task} - Num retrieved: {len(retrieval.retrieve_relevant_tasks(task))}')
    #     except Exception as e:
    #         skill_type, tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_folder_name(task)
    #
    #         task_tag = LanguageBasedRetrieval._create_task_tag(tgt_obj_class=tgt_obj_class,
    #                                                            other_obj_class=other_obj_class)
    #         print(f'\n{task} - tag: {task_tag}\n')
