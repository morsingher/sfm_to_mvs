from PIL import Image
import cv2
import numpy as np
import os
import shutil
import argparse
import matplotlib.pyplot as plt

from io_utils import *

def sfm_to_mvs(args):

    neighbors_out = {}
    count = 0

    x, y, z = [], [], []

    begin = int(args.begin)
    end = int(args.end)
    step = int(args.step)

    for i in range(begin, end + 1, step):

        print('Processing image {}...'.format(i))

        # Step 1: compute calibration and pose

        calib_filename = os.path.join(args.kitti_path, 'calib/calib_cam_to_cam.txt')
        with open(calib_filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
            intrinsics = lines[25].split()
            
        if (args.use_dvso_poses):
            pose_filename = os.path.join(args.kitti_path, 'poses_dvso.txt')
        else:
            pose_filename = os.path.join(args.kitti_path, 'poses.txt')

        with open(pose_filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
            pose = lines[i].split()
            pose = np.asarray(pose, dtype = 'float')
            x.append(pose[3])
            y.append(pose[7])
            z.append(pose[11])

        # Step 2: copy images

        in_img_filename = os.path.join(args.kitti_path, 'images/{:0>10d}.png'.format(i))
        img = cv2.imread(in_img_filename)
        out_img_filename = os.path.join(args.output_folder, 'images/{:0>8d}.jpg'.format(count))
        cv2.imwrite(out_img_filename, img)

        # Step 3: get keypoints from LiDAR scan or DVSO keypoints

        if args.use_dvso_points:
            depth_filename = os.path.join(args.kitti_path, 'keypoints/{:0>6d}.png'.format(i))
            depth = read_keypoints(depth_filename, img, float(intrinsics[1]))
        else:
            depth_filename = os.path.join(args.kitti_path, 'lidar/{:0>10d}.png'.format(i))
            depth = read_lidar_data(depth_filename)

        cols, rows = np.where(depth > 0.0)
        values = depth[cols, rows]
        points_filename = os.path.join(args.output_folder, 'points/{:0>8d}.txt'.format(count))
        np.savetxt(points_filename, np.vstack((rows, cols, values)).T, delimiter = ' ', fmt = '%f')

        depth_min = max(np.min(values), 0.5)
        depth_max = min(np.max(values), 80.0)

        write_camera_file(count, intrinsics, pose, args.output_folder, depth_min, depth_max)

        # Step 4: generate view selection file

        num_neighbors = 4

        same_ids = get_same_view_range(count, int((end - begin) / step) + 1, num_neighbors)
        score = [num_neighbors - abs(n - count) + 1 for n in same_ids]
        
        neighbors = []
        for n, s in zip(same_ids, score):
            neighbors.append((n, s))
        neighbors = sorted(neighbors, key = lambda x: x[1], reverse = True)
        neighbors_out[str(count)] = neighbors

        count += 1

    f = open(os.path.join(args.output_folder, 'pair.txt'), 'w')
    f.write('{}'.format(int((end - begin) / step) + 1))
    for i in range(count):
        neighbors = neighbors_out[str(i)]
        f.write('\n{}\n{} '.format(i, len(neighbors)))
        for n, s in neighbors:
            f.write('{} {} '.format(n, s))

    # plt.plot(x, z)
    # plt.axis('equal')
    # plt.grid()
    # plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert KITTI SFM output to MVS input')
    parser.add_argument('--kitti_path', required = True)
    parser.add_argument('--output_folder', required = True)
    parser.add_argument('--begin', default = 5)
    parser.add_argument('--end', default = 625)
    parser.add_argument('--step', default = 1)
    parser.add_argument('--use_dvso_poses', default = False)
    parser.add_argument('--use_dvso_points', default = False)
    args = parser.parse_args()

    if os.path.exists(args.output_folder):
        shutil.rmtree(args.output_folder)
    os.makedirs(args.output_folder)

    os.makedirs(os.path.join(args.output_folder, 'cameras'))
    os.makedirs(os.path.join(args.output_folder, 'points'))
    os.makedirs(os.path.join(args.output_folder, 'images'))

    sfm_to_mvs(args)
