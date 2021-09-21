# SFM-to-MVS

This is a collection of Python scripts for converting Structure-From-Motion (SFM) output into Multi-View Stereo (MVS) input. 

### COLMAP

The raw COLMAP (https://colmap.github.io/) output is supported. If you have custom data, get a sparse reconstruction from COLMAP and then simply run:

```
python3 colmap/colmap_to_mvs.py --input_folder <data_in> --output_folder <data_out> --model_ext <.txt/.bin>
```

If you want to test with ETH3D data, then:

- Download the undistorted version of a scene from https://www.eth3d.net/datasets
- Extract the archive and rename the calibration folder to `sparse`
- Run the command above

The difference with other scripts processing COLMAP output is that I also generate a keypoint file for each image, which contains the pixel coordinates and the depth of each keypoint seen by at least 3 cameras with reprojection error lower than 1 pixel.

### DDAD

The DDAD dataset has been recently released by Toyota (https://github.com/TRI-ML/DDAD). It contains a lot of stuff, including images, camera-aligned LiDAR scans, ground truth poses and other data you can check out in their repo. In order to generate MVS input for a single scene, just run:

```
python3 ddad/ddad_to_mvs.py --ddad_path <path> --scene <num> --output_folder <data_out>
```

This script makes use of the official Toyota guidelines for loading data (learn more here: https://github.com/TRI-ML/dgp). The main obscure thing when reading the code might be view selection. Here's the logic:

- Forward-facing front and back cameras have shared visibility with themselves and with their immediate neighbors.
- Lateral-facing cameras also have shared visibility with their respectively opposite camera, if shifted by +- k time instants. For example, the front-left camera at time 0 will see something similar to the back-left camera at time 10, and viceversa.

The depth range is deduced from LiDAR and each recorded point in the scan is considered as a keypoint, as for ETH3D.

NOTE: I also provide the possibility to generate static self-occlusion masks and use them for masking out self-occluded areas in the images. This is still a bit experimental and disabled by default. If you want to try, use the `ddad/generate_masks.py` script and pass the flag `--mask = True` to the other script above.

### KITTI

I support KITTI odometry sequences, since ground truth pose is easily available, as well as sparse LiDAR scans (at least for most of them). These are the required steps:

- Choose a KITTI sequence. The mapping between odometry sequences and raw data is the following:
```
mapping = {
    "2011_10_03_drive_0027": "00",
    "2011_10_03_drive_0042": "01",
    "2011_10_03_drive_0034": "02",
    "2011_09_26_drive_0067": "03",
    "2011_09_30_drive_0016": "04",
    "2011_09_30_drive_0018": "05",
    "2011_09_30_drive_0020": "06",
    "2011_09_30_drive_0027": "07",
    "2011_09_30_drive_0028": "08",
    "2011_09_30_drive_0033": "09",
    "2011_09_30_drive_0034": "10"
}
```
- Download the raw data from http://www.cvlibs.net/datasets/kitti/raw_data.php. Specifically, download the "synced+rectified" version, as well as the calibration. 
- Extract both of them in a directory called e.g. `kitti` as `kitti/images` and `kitti/calib`.
- Download the raw LiDAR scans from http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction and extract the corresponding folder into something like `kitti/lidar`.
- Download the ground truth poses from http://www.cvlibs.net/datasets/kitti/eval_odometry.php and extract the corresponding folder into something like `kitti/poses.txt`.
- Run the script:
```
python3 kitti/kitti_to_mvs --kitti_path <path> --output_folder <data_out>
```

You can optionally pass a lower and upper bound on frames if you want to select a subset of the whole sequence.

### Contributing

Feel free to add support for more algorithms and dataset (or to suggest meaningful modifications to existing ones). Ideally, produce a script called `<method>_to_mvs.py` and generate data in the required format. 

### Acknowledgements

The COLMAP script is only slightly adapted from https://github.com/GhiXu/ACMMP, all the credits to the authors.