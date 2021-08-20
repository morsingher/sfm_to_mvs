import numpy as np
import os
import json

def get_scene_json(main_path, ddad_path, split, scene):

    split = str(int(split != 'train'))
    with open(main_path) as f:
        data = json.load(f)
    
    filenames = data['scene_splits'][split]['filenames']

    selected_path = ''
    for filename in filenames:
        if filename[0:6] == scene:
            selected_path = filename 

    if selected_path:
        out_path = os.path.join(ddad_path, scene + '.json')
        with open(out_path, 'w') as f:
            json.dump({'scene_splits': {split : {'filenames': [selected_path]}}}, f)
        return out_path
    else:
        print('Failed to find the requested scene!')

def write_camera_file(idx, calib, pose, path, d_min, d_max):

    f = open(os.path.join(path, 'cameras/{:0>8d}.txt'.format(idx)), 'w')
    
    f.write('extrinsic\n')
    R = pose['rotation']
    t = pose['translation']
    for j in range(3):
        for k in range(3):
            f.write(str(R[j, k]) + ' ')
        f.write(str(t[j]) + ' \n')
    f.write('0.0 0.0 0.0 1.0 \n')
    
    f.write('\nintrinsic\n')
    for j in range(3):
        for k in range(3):
            f.write(str(calib[j, k]) + ' ')
        f.write('\n')

    depth_num = 192
    interval_scale = 1
    depth_interval = (d_max - d_min) / (depth_num - 1) / interval_scale
    f.write('\n{} {} {} {}'.format(0.1 * d_min, depth_interval, depth_num, d_max))

    f.close()

def get_same_view_range(idx, bound, dim):

    if idx < dim / 2:
        low = 0
        high = dim + 1
    elif idx >= bound - dim / 2:
        low = bound - dim - 1
        high = bound
    else:
        low = int(idx - dim / 2)
        high = int(idx + dim / 2 + 1)

    first_half = np.arange(low, idx)
    second_half = np.arange(idx + 1, high)

    return np.concatenate((first_half, second_half))

def get_lateral_view_range(idx, bound, dim):

    if idx < 0 or idx >= bound:
        return []
    else:
        if idx < dim / 2:
            low = 0
            high = dim + 1
        elif idx >= bound - dim / 2:
            low = bound - dim
            high = bound + 1
        else:
            low = int(idx - dim / 2)
            high = int(idx + dim / 2 + 1)

        first_half = np.arange(low, idx)
        second_half = np.arange(idx, high - 1)

        return np.concatenate((first_half, second_half))