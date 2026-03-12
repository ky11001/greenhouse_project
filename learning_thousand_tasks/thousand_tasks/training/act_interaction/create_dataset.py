import argparse

from thousand_tasks.core.globals import TASKS_DIR
from thousand_tasks.training.act_interaction.config import mt_act_args_dict
from thousand_tasks.training.act_interaction.dataset import BCDataset

if __name__ == '__main__':
    # --- Arg parser ---
    parser = argparse.ArgumentParser(description="BC interaction dataset processing script")
    parser.add_argument('-dir', '--dir', type=str, required=False,
                        help='Dataset directory (default: uses TASKS_DIR from globals.py)',
                        default=None)
    parser.add_argument('-demo_dir', '--demonstration_data_dir', type=str, required=False,
                        help='Separate demonstration directory (optional)', default=None)
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

    demo_dset_dir = args.demonstration_data_dir

    args.name = '_' + args.name if args.name != '' else ''

    if args.action_mode is not None:
        mt_act_args_dict['action_mode'] = args.action_mode

    print(f'\nCreating BC interaction training dataset from: {dataset_dir}')
    print('Generating interaction trajectories from demonstrations...\n')

    dataset_train = BCDataset(dataset_dir,
                              mt_act_args_dict,
                              reprocess=True,
                              processed_dir=f'interaction_processed{args.name}',
                              which_camera=args.camera,
                              separate_data_dir=demo_dset_dir)

    print(f'\nDataset creation complete! Processed data saved to:')
    print(f'{dataset_dir}/interaction_processed{args.name}/')
