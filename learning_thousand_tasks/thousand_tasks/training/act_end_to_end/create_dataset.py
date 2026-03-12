import argparse

from thousand_tasks.core.globals import TASKS_DIR
from thousand_tasks.training.act_end_to_end.config import mt_act_args_dict
from thousand_tasks.training.act_end_to_end.dataset import BCDataset

if __name__ == '__main__':
    # --- Arg parser ---
    parser = argparse.ArgumentParser(description="MT-ACT+ end-to-end dataset processing script")
    parser.add_argument('-dir', '--dir', type=str, required=False,
                        help='Dataset directory (default: uses TASKS_DIR from globals.py)',
                        default=None)
    parser.add_argument('-name', '--name', type=str, required=False,
                        help='Suffix for processed folder name', default='')
    parser.add_argument('-cam', '--camera', type=str, required=False,
                        help="Whether to use the 'external' or 'head' camera", default='head')
    parser.add_argument('-act_mode', '--action_mode', type=str, required=False,
                        help="Whether to predict delta poses ('delta') or absolute delta poses ('abs_delta')",
                        default=None)
    args = parser.parse_args()

    # Use provided directory or default to TASKS_DIR
    dataset_dir = args.dir if args.dir is not None else str(TASKS_DIR)

    args.name = '_' + args.name if args.name != '' else ''

    if args.action_mode is not None:
        mt_act_args_dict['action_mode'] = args.action_mode

    print(f'\nCreating MT-ACT+ end-to-end training dataset from: {dataset_dir}')
    print('Generating end-to-end trajectories from demonstrations...\n')

    dataset_train = BCDataset(dataset_dir,
                              mt_act_args_dict,
                              reprocess=True,
                              processed_dir=f'processed{args.name}',
                              which_camera=args.camera)

    print(f'\nDataset creation complete! Processed data saved to:')
    print(f'{dataset_dir}/processed{args.name}/')
