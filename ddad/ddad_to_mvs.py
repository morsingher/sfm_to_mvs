import cv2
import numpy as np
import os
import shutil
import argparse

from io_utils import *
from math_utils import *

from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset

def sfm_to_mvs(args):

    main_json_path = os.path.join(args.ddad_path, 'ddad.json')
    scene_json_path = get_scene_json(main_json_path, args.ddad_path, args.split, args.scene)

    cam_list = [1, 5, 6, 7, 8, 9]
    datums = ['lidar'] + ['CAMERA_%02d' % idx for idx in [1, 5, 6, 7, 8, 9]]
    num_cameras = 6

    ddad_data = SynchronizedSceneDataset(
        os.path.join(args.ddad_path, scene_json_path),
        split='train',
        datum_names=datums,
        generate_depth_from_datum='lidar'
    )

    num_frames = len(ddad_data)
    print('Loaded DDAD split containing {} samples'.format(num_frames))

    f = open(os.path.join(args.output_folder, 'pair.txt'), 'w')
    f.write('{}'.format(num_cameras * num_frames))
    neighbors_out = {}

    for i in range(num_frames):

        sample = ddad_data[i] # sample[0] - lidar, sample[1:] - camera datums
        sample_datum_names = [datum['datum_name'] for datum in sample]
        print('Loaded sample {} with datums {}'.format(i, sample_datum_names))

        for j, img in enumerate(sample[1:]):
            
            idx = j * num_frames + i

            # Step 1: get keypoints from LiDAR scan

            depth = img['depth']
            cols, rows = np.where(depth > 0.0)
            values = depth[cols, rows]
            points_filename = os.path.join(args.output_folder, 'points/{:0>8d}.txt'.format(idx))
            np.savetxt(points_filename, np.vstack((rows, cols, values)).T, delimiter = ' ', fmt = '%f')

            depth_min = np.min(values)
            depth_max = np.max(values)

            gt_depth_filename = os.path.join(args.output_folder, 'gt_depth/{:0>8d}.npz'.format(idx))
            np.savez(gt_depth_filename, depth)
            gt_depth_filename = os.path.join(args.output_folder, 'gt_depth/{:0>8d}.png'.format(idx))
            cv2.imwrite(gt_depth_filename, depth)

            # Step 2: write calibration and pose file

            calib = img['intrinsics']
            pose = build_pose(img['pose'])
            write_camera_file(idx, calib, pose, args.output_folder, depth_min, depth_max)

            # Step 3: write images (with optional masks)

            if args.mask:
                mask_path = os.path.join(args.ddad_path, args.scene, 'masks/%02d_mask.png' % cam_list[j])
                mask = cv2.imread(mask_path).astype(np.uint8)
                image = cv2.cvtColor(np.array(img['rgb']), cv2.COLOR_RGB2BGR)
                masked_image = cv2.bitwise_and(image, image, mask = mask[:, :, 1])
                cv2.imwrite(os.path.join(args.output_folder, 'images/{:0>8d}.jpg'.format(idx)), masked_image)
            else:
                img['rgb'].save(os.path.join(args.output_folder, 'images/{:0>8d}.jpg'.format(idx)))

            # Step 4: generate view selection files

            ''' 
            Multi-view selection rules:
            1) frontal frames have two kinds of neighbors:
                a) num_same from the same camera
                b) num_lat from each close lateral camera
            2) lateral frames have three kinds of neighbors:
                a) num_same from the same camera
                b) num_front from close central cameras
                c) num_offset from lateral cameras of the same side, with offset +- k
            '''

            num_same = 12
            num_lat = 4
            num_front = 4
            num_offset = 4
            offset = 10

            close_views = {'0': [1, 2], '1': [0, 3], '2': [0, 4], '3': [5, 1], '4': [5, 2], '5': [3, 4]}
            offsets = {'1': offset, '2': offset, '3': - offset, '4': - offset}

            # Compute frame indices from the same camera

            same_ids = get_same_view_range(i, num_frames, num_same)
            score = [num_same - abs(n - i) + 1 for n in same_ids]
            
            neighbors = []
            for n, s in zip(same_ids, score):
                neighbors.append((j * num_frames + n, s))

            # Compute lateral indices for forward-facing cameras

            if j == 0 or j == 5:

                lat_ids = get_lateral_view_range(i, num_frames, num_lat)
                score = [(num_lat - abs(n - i) + 1) for n in lat_ids]

                for n, s in zip(lat_ids, score):
                    for view in close_views[str(j)]:
                        neighbors.append((view * num_frames + n, s))

            # Compute other indices for lateral cameras

            else:
                front_ids = get_lateral_view_range(i, num_frames, num_front)
                score = [(num_front - abs(n - i) + 1) for n in front_ids]
                for n, s in zip(front_ids, score):
                    neighbors.append((close_views[str(j)][0] * num_frames + n, s))

                offset_ids = get_lateral_view_range(i + offsets[str(j)], num_frames, num_offset)
                score = [(num_offset - abs(n - (i + offsets[str(j)])) + 1) for n in offset_ids]
                for n, s in zip(offset_ids, score):
                    neighbors.append((close_views[str(j)][1] * num_frames + n, s))

            neighbors = sorted(neighbors, key = lambda x: x[1], reverse = True)
            neighbors_out[str(idx)] = neighbors

    for i in range(num_frames * num_cameras):
        neighbors = neighbors_out[str(i)]
        f.write('\n{}\n{} '.format(i, len(neighbors)))
        for n, s in neighbors:
            f.write('{} {} '.format(n, s))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert DDAD SFM output to MVS input')
    parser.add_argument('--ddad_path', required = True)
    parser.add_argument('--split', default = 'train')
    parser.add_argument('--scene', required = True)
    parser.add_argument('--output_folder', required = True)
    parser.add_argument('--mask', default = False)
    args = parser.parse_args()

    if os.path.exists(args.output_folder):
        shutil.rmtree(args.output_folder)
    os.makedirs(args.output_folder)

    os.makedirs(os.path.join(args.output_folder, 'cameras'))
    os.makedirs(os.path.join(args.output_folder, 'points'))
    os.makedirs(os.path.join(args.output_folder, 'images'))
    os.makedirs(os.path.join(args.output_folder, 'gt_depth'))

    sfm_to_mvs(args)
