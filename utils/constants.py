"""
We create a superset of joints containing the OpenPose joints together with the ones that each dataset provides.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are the following:
"""

# Folder configuration
DATA_DIR = 'data/CMU_Panoptic_Dataset_2.0'
SMPL_DIR = 'data/body_models'
KEYPOINTS_FLDR = 'OpenPose3D'
IMU_FLDR = 'MC10_IMU'
METADATA = 'metadata.json'
SMPL_REGRESSOR = 'SMPLCOCORegressor.npy'
SMPL_MEAN_PARAMS = 'smpl_mean_params.npz'

CAMERA_CALIB_FILENAME = 'calib_norm.json'

# IMU sensor list
IMU_PARTS = {'sacrum': 0, 'chest': 9, 'head' :15, 
             'lbicep': 16, 'lfoot': 7, 'lforearm': 18, 'lhand': 20, 'lshank': 4, 'lthigh': 1,
             'rbicep': 17, 'rfoot': 8, 'rforearm': 19, 'rhand': 21, 'rshank': 5, 'rthigh': 2}

OP26_TO_OP25 = [1, 0, 9, 10, 11, 3, 4, 5, 2, 12, 13, 14, 6, 7, 8, 17, 15, 18, 16, 20, 19, 21, 22, 23, 24]
SMPL_TO_OP25 = [39, 37, 33, 32, 31, 34, 35, 36, 38, 27, 26, 25, 28, 29, 30, 41, 40, 43, 42, 19, 20, 21, 22, 23, 24]
OP25_TO_OP26 = [OP26_TO_OP25.index(i) for i in range(len(OP26_TO_OP25))]

# Dict containing the joints in numerical order
# JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}


CONNECTIVITY = {'face': [17, 15, 0, 16, 18, 'crimson'],
                'back': [0, 1, 8, 'maroon'],
                'rarm': [4, 3, 2, 1, 'forestgreen'],
                'larm': [7, 6, 5, 1, 'orange'],
                'rleg': [11, 10, 9, 8, 'darkblue'],
                'lleg': [14, 13, 12, 8, 'seagreen'],
                'rfoot': [23, 22, 11, 24, 'mediumblue'],
                'lfoot': [20, 19, 14, 21, 'mediumseagreen']}


SEGMENT_DICT = {'larm1': [7,6],
                'larm2': [6,5],
                'rarm1': [4, 3],
                'rarm2': [3, 2],
                'lleg1': [14,13],
                'lleg2': [13, 12],
                'rleg1': [11, 10],
                'rleg2': [10, 9],
                'back': [9, 12, 1]
                }
