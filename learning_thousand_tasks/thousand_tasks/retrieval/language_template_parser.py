import re

from thousand_tasks.data.utils import remove_demo_number_from_task_folder_name

# How to add new language template:
# 1. Add it below to templates.
# 2. Implement a parser for it in this file
# 3. Add it to classify_skill() function
# 4. Update parse_skill() function

SKILL_TEMPLATES = ['pick_up_<>',
                   'pick_up_<>_from_<>',  # Both
                   'place_<>_on/in/next_to_<>',  # Both
                   'pour_from_<>_into_<>',  # Both
                   'insert_<>_into_<>',  # Both
                   'scoop_<>_from_<>',  # Both
                   'open_<>',  # target
                   'close_<>',  # Target
                   'stir_<>_in_<>',  # Both
                   'fold_<>',  # Target
                   'unfold_<>',  # Target
                   'iron_<>',  # Target
                   'swipe_<>_on_<>',  # Both
                   'turn_on_<>',  # Target
                   'wipe_<>_with_<>',  # Both
                   'shake_<>',  # Target
                   'dust_<>_with_<>',  # Both
                   'hang_<>_on_<>',  # Both
                   'zip_<>',  # Target
                   'unzip_<>',
                   'light_<>',
                   'push_<>',
                   'pull_<>',
                   'twist-clockwise_<>',
                   'twist-anticlockwise_<>',
                   'press_<>',
                   'turn-left_<>',
                   'turn-right_<>',
                   'turn-on_<>',
                   'turn_off_<>',
                   'remove_<>_from_<>',  # Both
                   'wash_<>_with_<>',  # Both
                   'fill_<>_with_<>',
                   'sharpen_<>_with_<>',
                   'stack_<>',
                   'regrasp_<>_on_<>'
                   ]

SKILL_NAMES = [skill_template.replace('_<>', '') for skill_template in SKILL_TEMPLATES]


def get_skill_type_from_task_name(task_name):
    if re.search('^pick_up_.+_from_.+', task_name) is not None:
        skill_type = 'pick_up_from'

    elif re.search('^pick_up_.+', task_name) is not None:
        skill_type = 'pick_up'

    elif re.search('^place_.+_.+_.+', task_name) is not None:
        skill_type = 'place_on/in/next_to'

    elif re.search('^pour_.+_from_.+_into_.+', task_name) is not None:
        skill_type = 'pour_from_into'

    elif re.search('^insert_.+_into_.+', task_name) is not None:
        skill_type = 'insert_into'

    elif re.search('^scoop_.+_from_.+', task_name) is not None:
        skill_type = 'scoop_from'

    elif re.search('^open_.+', task_name) is not None:
        skill_type = 'open'

    elif re.search('^close_.+', task_name) is not None:
        skill_type = 'close'

    elif re.search('^stir_.+_in_.+', task_name) is not None:
        skill_type = 'stir_in'

    elif re.search('^fold_.*', task_name) is not None:
        skill_type = 'fold'

    elif re.search('^iron_.*', task_name) is not None:
        skill_type = 'iron'

    elif re.search('^swipe_.*_on_.+', task_name) is not None:
        skill_type = 'swipe_on'

    elif re.search('^turn-on_.*', task_name) is not None:
        skill_type = 'turn_on'

    elif re.search('^turn-off_.*', task_name) is not None:
        skill_type = 'turn_off'

    elif re.search('^wipe_.*_with_.*', task_name) is not None:
        skill_type = 'wipe_with'

    elif re.search('^shake_.*', task_name) is not None:
        skill_type = 'shake'

    elif re.search('^dust_.*_with_.+', task_name) is not None:
        skill_type = 'dust_with'

    elif re.search('^unfold_.*', task_name) is not None:
        skill_type = 'unfold'

    elif re.search('^hang_.*_on_.+', task_name) is not None:
        skill_type = 'hang_on'

    elif re.search('^zip_.*', task_name) is not None:
        skill_type = 'zip'

    elif re.search('^unzip_.*', task_name) is not None:
        skill_type = 'unzip'

    elif re.search('^light_.*', task_name) is not None:
        skill_type = 'light'

    elif re.search('^push_.*', task_name) is not None:
        skill_type = 'push'

    elif re.search('^pull_.*', task_name) is not None:
        skill_type = 'pull'

    elif re.search('^twist-clockwise_.*', task_name) is not None:
        skill_type = 'twist-clockwise'

    elif re.search('^twist-anticlockwise_.*', task_name) is not None:
        skill_type = 'twist-anticlockwise'

    elif re.search('^press_.*', task_name) is not None:
        skill_type = 'press'

    elif re.search('^turn-left_.*', task_name) is not None:
        skill_type = 'turn-left'

    elif re.search('^turn-right_.*', task_name) is not None:
        skill_type = 'turn-right'

    elif re.search('^stack_.*', task_name) is not None:
        skill_type = 'stack'

    elif re.search('^remove_.*_from_', task_name) is not None:
        skill_type = 'remove_from'

    elif re.search('^wash_.*_with_.+', task_name) is not None:
        skill_type = 'wash_with'

    elif re.search('^fill_.*_with_.+', task_name) is not None:
        skill_type = 'fill_with'

    elif re.search('^sharpen_.*_with_.+', task_name) is not None:
        skill_type = 'sharpen_with'

    elif re.search('^regrasp_.*_on_.+', task_name) is not None:
        skill_type = 'regrasp_on'

    else:
        error_message = f'\n\nTask \'{task_name}\' can\'t be classified. To add this skill:\n\n'
        error_message += '- Add the skill template to SKILL_TEMPLATES at the top of this file\n'
        error_message += '- Add a clause above that returns the new skill type\n'
        error_message += '- Add a corresponding function below to parse this skill and return a target and other object\n'
        error_message += '- Add a clause to the parse_skill function to parse this new skill\n'
        raise Exception(error_message)

    return skill_type


def parse_task_name(task_name, template_prefix):
    task_folder_name = remove_demo_number_from_task_folder_name(task_name)

    task_folder_name = task_folder_name.split(template_prefix + '_')[-1].replace('_', ' ')
    last_space_idx = task_folder_name.rfind(" ")

    target_obj_class = task_folder_name[last_space_idx + 1:]
    if last_space_idx == -1:
        target_obj_description = None
    else:
        target_obj_description = task_folder_name[:last_space_idx]

    other_obj_class = None
    other_obj_description = None

    return target_obj_class, target_obj_description, other_obj_class, other_obj_description


def parse_task_name_involving_grasped_obj(task_name, template_prefix, template_connector,
                                          reverse_tgt_and_grasped=False):
    task_folder_name = remove_demo_number_from_task_folder_name(task_name)

    # decompose
    grasped_obj_part = re.search('^' + template_prefix + '_.+_' + template_connector, task_folder_name).group()
    tgt_obj_part = re.search(template_connector + '_.+', task_folder_name).group()

    grasped_obj_part = grasped_obj_part.split(template_prefix + '_', )[-1].split('_' + template_connector)[0].replace(
        '_', ' ')
    dust_last_space_idx = grasped_obj_part.rfind(" ")
    other_obj_class = grasped_obj_part[dust_last_space_idx + 1:]
    if dust_last_space_idx == -1:
        other_obj_description = None
    else:
        other_obj_description = grasped_obj_part[:dust_last_space_idx].replace(' ', '_')

    tgt_obj_part = tgt_obj_part.split(template_connector + '_')[-1].replace('_', ' ')
    from_last_space_idx = tgt_obj_part.rfind(" ")
    target_obj_class = tgt_obj_part[from_last_space_idx + 1:]
    if from_last_space_idx == -1:
        target_obj_description = None
    else:
        target_obj_description = tgt_obj_part[:from_last_space_idx].replace(' ', '_')

    if reverse_tgt_and_grasped:
        target_obj_class, target_obj_description, other_obj_class, other_obj_description = other_obj_class, other_obj_description, target_obj_class, target_obj_description

    return target_obj_class, target_obj_description, other_obj_class, other_obj_description


def parse_task_name_involving_secondary_target_obj(task_name, template_prefix, template_connector,
                                          reverse_tgt_and_grasped=False):
    task_folder_name = remove_demo_number_from_task_folder_name(task_name)

    # decompose
    grasped_obj_part = re.search('^' + template_prefix + '_.+_' + template_connector, task_folder_name).group()
    tgt_obj_part = re.search(template_connector + '_.+', task_folder_name).group()

    grasped_obj_part = grasped_obj_part.split(template_prefix + '_', )[-1].split('_' + template_connector)[0].replace(
        '_', ' ')
    dust_last_space_idx = grasped_obj_part.rfind(" ")
    other_obj_class = grasped_obj_part[dust_last_space_idx + 1:]
    if dust_last_space_idx == -1:
        other_obj_description = None
    else:
        other_obj_description = grasped_obj_part[:dust_last_space_idx]

    tgt_obj_part = tgt_obj_part.split(template_connector + '_')[-1].replace('_', ' ')
    from_last_space_idx = tgt_obj_part.rfind(" ")
    target_obj_class = tgt_obj_part[from_last_space_idx + 1:]
    if from_last_space_idx == -1:
        target_obj_description = None
    else:
        target_obj_description = tgt_obj_part[:from_last_space_idx]

    if reverse_tgt_and_grasped:
        target_obj_class, target_obj_description, other_obj_class, other_obj_description = other_obj_class, other_obj_description, target_obj_class, target_obj_description

    return target_obj_class, target_obj_description, other_obj_class, other_obj_description

def parse_place_task_folder_name(task_folder_name):
    task_folder_name = remove_demo_number_from_task_folder_name(task_folder_name)

    # decompose
    if re.search('_on_', task_folder_name) is not None:
        place_part = re.search('^place_.+_on', task_folder_name).group()
        on_part = re.search('on_.+', task_folder_name).group()
        split_arg1 = '_on'
        split_arg2 = 'on_'

    elif re.search('_next_to_', task_folder_name) is not None:
        place_part = re.search('^place_.+_next_to', task_folder_name).group()
        on_part = re.search('next_to_.+', task_folder_name).group()
        split_arg1 = '_next_to'
        split_arg2 = 'next_to_'

    elif re.search('_in_', task_folder_name) is not None:
        place_part = re.search('^place_.+_in', task_folder_name).group()
        on_part = re.search('in_.+', task_folder_name).group()
        split_arg1 = '_in'
        split_arg2 = 'in_'

    place_part = place_part.split('place_')[-1].split(split_arg1)[0].replace('_', ' ')
    pick_last_space_idx = place_part.rfind(" ")
    other_obj_class = place_part[pick_last_space_idx + 1:]
    if pick_last_space_idx == -1:
        other_obj_description = None
    else:
        other_obj_description = place_part[:pick_last_space_idx]

    on_part = on_part.split(split_arg2)[-1].replace('_', ' ')
    from_last_space_idx = on_part.rfind(" ")
    target_obj_class = on_part[from_last_space_idx + 1:]
    if from_last_space_idx == -1:
        target_obj_description = None
    else:
        target_obj_description = on_part[:from_last_space_idx]

    return target_obj_class, target_obj_description, other_obj_class, other_obj_description


def parse_pour_from_into_task_folder_name(task_folder_name):
    task_folder_name = remove_demo_number_from_task_folder_name(task_folder_name)

    # decompose
    pour_from_part = re.search('^pour_.+_from_.+_into', task_folder_name).group()
    pour_into_part = re.search('into_.+', task_folder_name).group()

    pour_from_part = pour_from_part.split('_from_')[-1].split('_into')[0].replace('_', ' ')
    pour_from_last_space_idx = pour_from_part.rfind(" ")
    other_obj_class = pour_from_part[pour_from_last_space_idx + 1:]
    if pour_from_last_space_idx == -1:
        other_obj_description = None
    else:
        other_obj_description = pour_from_part[:pour_from_last_space_idx]

    pour_into_part = pour_into_part.split('into_')[-1].replace('_', ' ')
    into_last_space_idx = pour_into_part.rfind(" ")
    target_obj_class = pour_into_part[into_last_space_idx + 1:]
    if into_last_space_idx == -1:
        target_obj_description = None
    else:
        target_obj_description = pour_into_part[:into_last_space_idx]

    return target_obj_class, target_obj_description, other_obj_class, other_obj_description

def parse_task_folder_name(task_folder_name):
    skill_type = get_skill_type_from_task_name(task_folder_name)

    if skill_type == 'pick_up_from':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name_involving_grasped_obj(
            task_folder_name, 'pick_up', 'from')
    elif skill_type == 'pick_up':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'pick_up')
    elif skill_type == 'place_on/in/next_to':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_place_task_folder_name(task_folder_name)
    elif skill_type == 'pour_from_into':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_pour_from_into_task_folder_name(
            task_folder_name)
    elif skill_type == 'insert_into':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name_involving_grasped_obj(
            task_folder_name, 'insert', 'into')
    elif skill_type == 'scoop_from':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name_involving_grasped_obj(
            task_folder_name, 'scoop', 'from')
    elif skill_type == 'open':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'open')
    elif skill_type == 'close':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'close')
    elif skill_type == 'stir_in':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name_involving_grasped_obj(
            task_folder_name, 'stir', 'in')
    elif skill_type == 'fold':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'fold')
    elif skill_type == 'unfold':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'unfold')
    elif skill_type == 'iron':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'iron')
    elif skill_type == 'swipe_on':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name_involving_grasped_obj(
            task_folder_name, 'swipe', 'on')
    elif skill_type == 'turn_on':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'turn-on')
    elif skill_type == 'turn_off':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'turn-off')
    elif skill_type == 'wipe_with':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name_involving_grasped_obj(
            task_folder_name, 'wipe', 'with', True)
    elif skill_type == 'shake':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'shake')
    elif skill_type == 'dust_with':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name_involving_grasped_obj(
            task_folder_name, 'dust', 'with', True)
    elif skill_type == 'hang_on':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name_involving_grasped_obj(
            task_folder_name, 'hang', 'on')
    elif skill_type == 'zip':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'zip')
    elif skill_type == 'unzip':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'unzip')
    elif skill_type == 'light':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'light')
    elif skill_type == 'push':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'push')
    elif skill_type == 'pull':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'pull')
    elif skill_type == 'twist-clockwise':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name,
                                                                                     'twist-clockwise')
    elif skill_type == 'twist-anticlockwise':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name,
                                                                                     'twist-anticlockwise')
    elif skill_type == 'press':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'press')
    elif skill_type == 'turn-left':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'turn-left')
    elif skill_type == 'turn-right':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'turn-right')
    elif skill_type == 'stack':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name(task_folder_name, 'stack')
    elif skill_type == 'remove_from':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name_involving_grasped_obj(
            task_folder_name, 'remove', 'from', reverse_tgt_and_grasped=True)
    elif skill_type == 'wash_with':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name_involving_grasped_obj(
            task_folder_name, 'wash', 'with', reverse_tgt_and_grasped=True)
    elif skill_type == 'fill_with':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name_involving_grasped_obj(
            task_folder_name, 'fill', 'with', reverse_tgt_and_grasped=True)
    elif skill_type == 'sharpen_with':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name_involving_grasped_obj(
            task_folder_name, 'sharpen', 'with')
    elif skill_type == 'regrasp_on':
        tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des = parse_task_name_involving_grasped_obj(
            task_folder_name, 'regrasp', 'on')

    else:
        error_message = f'\n\nTask name: \'{task_folder_name}\'. Parser for skill \'{skill_type}\' does not exist. To fully add this skill\n\n'
        error_message += '- Add the skill template to SKILL_TEMPLATES at the top of this file\n'
        error_message += '- Add a function below to parse this skill and return a target and other object\n'
        error_message += '- Add a clause above that calls this function\n'
        raise Exception(error_message)

    return skill_type, tgt_obj_class, tgt_obj_des, other_obj_class, other_obj_des
