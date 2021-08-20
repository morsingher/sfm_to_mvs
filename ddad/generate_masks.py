import numpy as np
import cv2
import os
import argparse
import shutil
import glob

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--ddad_path', required = True)
	parser.add_argument('--scene', required = True)
	args = parser.parse_args()

	scene_path = os.path.join(args.ddad_path, args.scene)
	masks_dir = os.path.join(scene_path, 'masks')

	if os.path.exists(masks_dir):
		shutil.rmtree(masks_dir)
	os.makedirs(masks_dir)  

	contours = {}
	contours['05'] = np.array([[0, 1216], [1035, 1035], [1115, 1045], [1236, 1053], [1320, 1068], [1407, 1106], [1572, 1140], [1766, 1053], [1881, 1046], [1936, 1053], [1936, 1216]], dtype=np.int32)
	contours['06'] = np.array([[0, 1216], [0, 1090], [102, 1090], [270, 1168], [842, 1050], [1700, 1216]], dtype=np.int32)
	contours['07'] = np.array([[0, 0], [560, 0], [1, 254], [1, 754], [572, 840], [1260, 998], [1936, 1158], [1936, 1216], [0, 1216]], dtype=np.int32)
	contours['08'] = np.array([[1196, 0], [1936, 0], [1936, 1216], [0, 1216], [0, 1186], [848, 970], [1468, 836], [1935, 756], [1935, 344], [1816, 252]], dtype=np.int32)
	contours['09'] = np.array([[0, 1216], [0, 1088], [356, 986], [638, 962], [1066, 964], [1300, 986], [1550, 1076], [1734, 1150], [1935, 1178], [1935, 88], [844, 0], [1936, 0], [1936, 1216]], dtype=np.int32)

	# Camera 01 has no self-occluded areas, I generate a white mask just for consistency

	cam_path = os.path.join(scene_path, 'rgb/CAMERA_01')
	img_list = glob.glob(os.path.join(cam_path, '*.png'))
	img = cv2.imread(img_list[0])
	mask = np.ones((img.shape)) * 255
	mask_path = os.path.join(masks_dir, '01_mask.png')
	cv2.imwrite(mask_path, mask)

	for i in range(5, 10):

		cam_path = os.path.join(scene_path, 'rgb/CAMERA_0{}'.format(i))
		img_list = glob.glob(os.path.join(cam_path, '*.png'))

		img = cv2.imread(img_list[0])

		mask = np.ones((img.shape)) * 255
		cv2.fillPoly(mask, [contours['0{}'.format(i)]], (0,0,0))
		mask = mask.astype(np.uint8)

		mask_path = os.path.join(masks_dir, '0{}_mask.png'.format(i))
		cv2.imwrite(mask_path, mask)
