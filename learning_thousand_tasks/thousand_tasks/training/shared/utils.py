import numpy as np

from os.path import join

from scipy.spatial.transform import Slerp, Rotation as Rot
import inspect


def printarr(*arrs, float_width=6):
    """
    Print a pretty table giving name, shape, dtype, type, and content information for input tensors or scalars.
    Call like: printarr(my_arr, some_other_arr, maybe_a_scalar). Accepts a variable number of arguments.
    Inputs can be:
        - Numpy tensor arrays
        - Pytorch tensor arrays
        - Jax tensor arrays
        - Python ints / floats
        - None
    It may also work with other array-like types, but they have not been tested.
    Use the `float_width` option specify the precision to which floating point types are printed.
    Author: Nicholas Sharp (nmwsharp.com)
    Canonical source: https://gist.github.com/nmwsharp/54d04af87872a4988809f128e1a1d233
    License: This snippet may be used under an MIT license, and it is also released into the public domain.
             Please retain this docstring as a reference.
    """

    frame = inspect.currentframe().f_back
    default_name = "[temporary]"

    ## helpers to gather data about each array
    def name_from_outer_scope(a):
        if a is None:
            return '[None]'
        name = default_name
        for k, v in frame.f_locals.items():
            if v is a:
                name = k
                break
        return name

    def dtype_str(a):
        if a is None:
            return 'None'
        if isinstance(a, int):
            return 'int'
        if isinstance(a, float):
            return 'float'
        return str(a.dtype)

    def shape_str(a):
        if a is None:
            return 'N/A'
        if isinstance(a, int):
            return 'scalar'
        if isinstance(a, float):
            return 'scalar'
        return str(list(a.shape))

    def type_str(a):
        return str(type(a))[8:-2]  # TODO this is is weird... what's the better way?

    def device_str(a):
        if hasattr(a, 'device'):
            device_str = str(a.device)
            if len(device_str) < 10:
                # heuristic: jax returns some goofy long string we don't want, ignore it
                return device_str
        return ""

    def format_float(x):
        return f"{x:{float_width}g}"

    def minmaxmean_str(a):
        if a is None:
            return ('N/A', 'N/A', 'N/A')
        if isinstance(a, int) or isinstance(a, float):
            return (format_float(a), format_float(a), format_float(a))

        # compute min/max/mean. if anything goes wrong, just print 'N/A'
        min_str = "N/A"
        try:
            min_str = format_float(a.min())
        except:
            pass
        max_str = "N/A"
        try:
            max_str = format_float(a.max())
        except:
            pass
        mean_str = "N/A"
        try:
            mean_str = format_float(a.mean())
        except:
            pass

        return (min_str, max_str, mean_str)

    try:

        props = ['name', 'dtype', 'shape', 'type', 'device', 'min', 'max', 'mean']

        # precompute all of the properties for each input
        str_props = []
        for a in arrs:
            minmaxmean = minmaxmean_str(a)
            str_props.append({
                'name': name_from_outer_scope(a),
                'dtype': dtype_str(a),
                'shape': shape_str(a),
                'type': type_str(a),
                'device': device_str(a),
                'min': minmaxmean[0],
                'max': minmaxmean[1],
                'mean': minmaxmean[2],
            })

        # for each property, compute its length
        maxlen = {}
        for p in props: maxlen[p] = 0
        for sp in str_props:
            for p in props:
                maxlen[p] = max(maxlen[p], len(sp[p]))

        # if any property got all empty strings, don't bother printing it, remove if from the list
        props = [p for p in props if maxlen[p] > 0]

        # Added by Alex.
        # Account for possibility that header is longer than any of the values
        maxlen = {p: max(maxlen[p], len(p)) for p in props}

        # print a header
        header_str = ""
        for p in props:
            prefix = "" if p == 'name' else " | "
            fmt_key = ">" if p == 'name' else "<"
            header_str += f"{prefix}{p:{fmt_key}{maxlen[p]}}"
        print(header_str)
        print("-" * len(header_str))

        # now print the acual arrays
        for strp in str_props:
            for p in props:
                prefix = "" if p == 'name' else " | "
                fmt_key = ">" if p == 'name' else "<"
                print(f"{prefix}{strp[p]:{fmt_key}{maxlen[p]}}", end='')
            print("")

    finally:
        del frame

def interpolate_poses(starting_pose, target_pose, num_interpolations):
    trans_actions = np.linspace(starting_pose[:3, 3], target_pose[:3, 3], num_interpolations)
    slerp = Slerp([0, 1], Rot.from_matrix(np.stack((starting_pose[:3, :3], target_pose[:3, :3]))))
    rot_actions = slerp(np.linspace(0, 1, num_interpolations)).as_rotvec()

    pose_actions = [np.eye(4) for _ in range(num_interpolations)]
    for k in range(num_interpolations):
        pose_actions[k][:3, 3] = trans_actions[k]
        pose_actions[k][:3, :3] = Rot.from_rotvec(rot_actions[k]).as_matrix()

    return pose_actions


def get_task_name(task_dir):
    with open(join(task_dir, "task_name.txt"), "r") as f:
        task_name = f.read()
    return task_name

def get_skill_name(task_dir):
    with open(join(task_dir, "skill_name.txt"), "r") as f:
        task_name = f.read()
    return task_name
