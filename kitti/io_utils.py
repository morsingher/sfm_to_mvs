import numpy as np
import os

def compute_pose(pose, calib):

    data = pose.reshape((3, 4))
    
    R = data[:3, :3]
    R = np.linalg.inv(R)
    
    t = data[:, 3]
    offset = np.matmul(np.linalg.inv(calib[:3, :3]), calib[:, 3])
    t = t + offset

    t = np.matmul(R, -t)
    return R, t

def write_camera_file(idx, calib, pose, path, d_min, d_max):

    f = open(os.path.join(path, 'cameras/{:0>8d}.txt'.format(idx)), 'w')
    
    calib = np.asarray(calib[1:], dtype = 'float')
    calib = calib.reshape((3, 4))

    R, t = compute_pose(pose, calib)

    f.write('extrinsic\n')
    for j in range(3):
        for k in range(3):
            f.write(str(R[j, k]) + ' ')
        f.write(str(t[j]) + ' \n')
    f.write('0.0 0.0 0.0 1.0 \n')

    intrinsics = calib[:3, :3]
    f.write('\nintrinsic\n')
    for j in range(3):
        for k in range(3):
            f.write(str(intrinsics[j, k]) + ' ')
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
