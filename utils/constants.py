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
