#!/usr/bin/env python

from io_utils import *
from math_utils import *

import collections
import struct
import numpy as np
import multiprocessing as mp
from functools import partial
import os
import argparse
import shutil
import cv2

def sfm_to_mvs(args):

    image_dir = os.path.join(args.input_folder, 'images')
    model_dir = os.path.join(args.input_folder, 'sparse')
    cam_dir = os.path.join(args.output_folder, 'cameras')
    image_converted_dir = os.path.join(args.output_folder, 'images')
    points_dir = os.path.join(args.output_folder, 'points')

    if os.path.exists(image_converted_dir):
        print("remove:{}".format(image_converted_dir))
        shutil.rmtree(image_converted_dir)
    os.makedirs(image_converted_dir)
    if os.path.exists(cam_dir):
        print("remove:{}".format(cam_dir))
        shutil.rmtree(cam_dir)

    if os.path.exists(points_dir):
        print("remove:{}".format(points_dir))
        shutil.rmtree(points_dir)
    os.makedirs(points_dir)  

    cameras, images, points3d = read_model(model_dir, args.model_ext)
    num_images = len(list(images.items()))

    param_type = {
        'SIMPLE_PINHOLE': ['f', 'cx', 'cy'],
        'PINHOLE': ['fx', 'fy', 'cx', 'cy'],
        'SIMPLE_RADIAL': ['f', 'cx', 'cy', 'k'],
        'SIMPLE_RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k'],
        'RADIAL': ['f', 'cx', 'cy', 'k1', 'k2'],
        'RADIAL_FISHEYE': ['f', 'cx', 'cy', 'k1', 'k2'],
        'OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2'],
        'OPENCV_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'k3', 'k4'],
        'FULL_OPENCV': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6'],
        'FOV': ['fx', 'fy', 'cx', 'cy', 'omega'],
        'THIN_PRISM_FISHEYE': ['fx', 'fy', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'sx1', 'sy1']
    }

    # Intrinsic parameters

    print('Reading intrinsic paramters...')

    intrinsic = {}
    for camera_id, cam in cameras.items():
        params_dict = {key: value for key, value in zip(param_type[cam.model], cam.params)}
        if 'f' in param_type[cam.model]:
            params_dict['fx'] = params_dict['f']
            params_dict['fy'] = params_dict['f']
        i = np.array([
            [params_dict['fx'], 0, params_dict['cx']],
            [0, params_dict['fy'], params_dict['cy']],
            [0, 0, 1]
        ])
        intrinsic[camera_id] = i

    print('Done!')

    new_images = {}
    for i, image_id in enumerate(sorted(images.keys())):
        new_images[i+1] = images[image_id]
    images = new_images

    # Extrinsic paramters

    print('Reading extrinsic parameters...')

    extrinsic = {}
    for image_id, image in images.items():
        e = np.zeros((4, 4))
        e[:3, :3] = qvec2rotmat(image.qvec)
        e[:3, 3] = image.tvec
        e[3, 3] = 1
        extrinsic[image_id] = e
    
    print('Done!')

    # Keypoints and depth range

    print('Reading keypoints and computing depth range...')

    depth_ranges = {}
    for i in range(num_images):

        fx = intrinsic[images[i+1].camera_id][0][0]
        fy = intrinsic[images[i+1].camera_id][1][1]
        cx = intrinsic[images[i+1].camera_id][0][2]
        cy = intrinsic[images[i+1].camera_id][1][2]
        xs, ys, zs = [], [], []
        
        for p3d_id in images[i+1].point3D_ids:
            if p3d_id == -1 or points3d[p3d_id].error > 1.0 or len(points3d[p3d_id].image_ids) < 3:
                continue
            transformed = np.matmul(extrinsic[i+1], [points3d[p3d_id].xyz[0], points3d[p3d_id].xyz[1], points3d[p3d_id].xyz[2], 1])
            xs.append((fx * (transformed[0] / transformed[2]) + cx).item())
            ys.append((fy * (transformed[1] / transformed[2]) + cy).item())
            zs.append(transformed[2].item())
        zs_sorted = sorted(zs)
        
        depth_min = zs_sorted[int(len(zs) * .01)] * 0.75
        depth_max = zs_sorted[int(len(zs) * .99)] * 1.25

        # Write keypoints

        with open(os.path.join(points_dir, '%08d.txt' % i), 'w') as f:
            for u, v, d in zip(xs, ys, zs):
                f.write('%f %f %f \n' % (u, v, d))

        # Determine depth number

        if args.max_d == 0:
            image_int = intrinsic[images[i+1].camera_id]
            image_ext = extrinsic[i+1]
            image_r = image_ext[0:3, 0:3]
            image_t = image_ext[0:3, 3]
            p1 = [image_int[0, 2], image_int[1, 2], 1]
            p2 = [image_int[0, 2] + 1, image_int[1, 2], 1]
            P1 = np.matmul(np.linalg.inv(image_int), p1) * depth_min
            P1 = np.matmul(np.linalg.inv(image_r), (P1 - image_t))
            P2 = np.matmul(np.linalg.inv(image_int), p2) * depth_min
            P2 = np.matmul(np.linalg.inv(image_r), (P2 - image_t))
            depth_num = (1 / depth_min - 1 / depth_max) / (1 / depth_min - 1 / (depth_min + np.linalg.norm(P2 - P1)))
        else:
            depth_num = args.max_d
        depth_interval = (depth_max - depth_min) / (depth_num - 1) / args.interval_scale
        depth_ranges[i+1] = (depth_min, depth_interval, depth_num, depth_max)

    print('Done!')

    # View selection
    
    print('Computing neighbors with view selection (this might take a while)...')

    score = np.zeros((len(images), len(images)))
    queue = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            queue.append((i, j))

    p = mp.Pool(processes=mp.cpu_count())
    func = partial(calc_score, images=images, points3d=points3d, args=args, extrinsic=extrinsic)
    result = p.map(func, queue)
    for i, j, s in result:
        score[i, j] = s
        score[j, i] = s
    view_sel = []
    num_view = min(20, len(images) - 1)
    for i in range(len(images)):
        sorted_score = np.argsort(score[i])[::-1]
        view_sel.append([(k, score[i, k]) for k in sorted_score[:num_view]])
    
    print('Done!')

    # Generate cameras files

    print('Generating cameras files...')

    try:
        os.makedirs(cam_dir)
    except os.error:
        print(cam_dir + ' already exist.')
    for i in range(num_images):
        with open(os.path.join(cam_dir, '%08d.txt' % i), 'w') as f:
            f.write('extrinsic\n')
            for j in range(4):
                for k in range(4):
                    f.write(str(extrinsic[i+1][j, k]) + ' ')
                f.write('\n')
            f.write('\nintrinsic\n')
            for j in range(3):
                for k in range(3):
                    f.write(str(intrinsic[images[i+1].camera_id][j, k]) + ' ')
                f.write('\n')
            f.write('\n%f %f %f %f\n' % (depth_ranges[i+1][0], depth_ranges[i+1][1], depth_ranges[i+1][2], depth_ranges[i+1][3]))
    with open(os.path.join(args.output_folder, 'pair.txt'), 'w') as f:
        f.write('%d\n' % len(images))
        for i, sorted_score in enumerate(view_sel):
            f.write('%d\n%d ' % (i, len(sorted_score)))
            for image_id, s in sorted_score:
                f.write('%d %d ' % (image_id, s))
            f.write('\n')

    print('Done!')

    # Convert images to jpg

    print('Converting images to jpg (this might take a while)...')

    for i in range(num_images):
        img_path = os.path.join(image_dir, images[i + 1].name)
        if not img_path.endswith(".jpg"):
            print(img_path)
            img = cv2.imread(img_path)
            cv2.imwrite(os.path.join(image_converted_dir, '%08d.jpg' % i), img)
        else:
            shutil.copyfile(os.path.join(image_dir, images[i+1].name), os.path.join(image_converted_dir, '%08d.jpg' % i))

    print('Done!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Convert COLMAP SFM output to MVS input')

    parser.add_argument('--input_folder', required=True, type=str, help='input_folder.')
    parser.add_argument('--output_folder', required=True, type=str, help='output_folder.')
    parser.add_argument('--max_d', type=int, default=192)
    parser.add_argument('--interval_scale', type=float, default=1)
    parser.add_argument('--model_ext', type=str, default=".txt",  choices=[".txt", ".bin"], help='sparse model ext')

    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_folder), exist_ok=True)
    
    sfm_to_mvs(args)