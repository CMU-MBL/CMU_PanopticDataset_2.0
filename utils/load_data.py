from utils.conversion import rotation_matrix_to_angle_axis, rot6d_to_rotmat

import torch
import numpy as np
import pandas as pd
import os, sys
import os.path as osp
import csv, json

from tqdm import tqdm


def loadJson(file_name):
    """Load keypoints from json file
    Input : File name
    Output : index, 26 joints location
    """
    ids, joints = [[], []]
    with open(file_name) as jsonFile:
        data = json.load(jsonFile)
        for body in data["bodies"]:
            ids.append(body['id'])
            joints.append(np.array(body['joints26']).reshape(1,-1,4))
    try: 
        joints = np.vstack(joints)
    except:
        pass

    return ids, joints


def loadKeypoints(path, num_file=-1):
    """Load 3D openpose joints after fixing
    Input : Experiment path
            num_file is not -1 only if trying to load fewer data
    Output : index, 26 joints location for entire frames
    """
    
    _, _, files_ = next(os.walk(path))
    files_ = [file_ for file_ in files_ if (file_[0] == 'b' and file_[-1] == 'n')]
    files_.sort()
    joints = []
    n_file = len(files_) if num_file == -1 else num_file
    with tqdm(total=n_file, desc='Loading OP26 data...', leave=False) as prog_bar:
        for file_ in files_[:n_file]:
            ids, cur_joints = loadJson(os.path.join(path, file_))
            
            if cur_joints.shape[0] == 1 and len(ids) != 0 and type(ids[0]) == str:
                # Processed data
                cur_joints = [cur_joints[0]]
                ids = [0]

            for id in ids:
                if len(joints) == 0:
                    joints.append(cur_joints[0][None])
                elif len(joints) == 1 and len(cur_joints) == 2:
                    joints.append(cur_joints[1][None])
                else:
                    joints[id] = np.vstack([joints[id], cur_joints[id][None]])
            prog_bar.update(1)
            prog_bar.refresh()
    
    ids = [i for i in range(len(joints))]
    
    # Some experiments record openpose first
    if path.split('/')[-2] == '190607' and path[-2:] == '13':
        joints[0] = joints[0][150:]
        joints[1] = joints[1][150:]
    
    return joints, ids


def loadIMU(path, imuParts, seqLength):
    index = 'Timestamp (microseconds)'
    
    gyros, accels = [], []
    for imuPart in imuParts:
        gyro = pd.read_csv(osp.join(path, '%s_gyro.csv'%imuPart), index_col=index)
        gyro = np.array(gyro)[:5 * seqLength].reshape(-1, 5, 3).mean(axis=1, keepdims=True)[:-1]
        accel = pd.read_csv(osp.join(path, '%s_accel.csv'%imuPart), index_col=index)
        accel = np.array(accel)[:5 * seqLength].reshape(-1, 5, 3).mean(axis=1, keepdims=True)

        gyros += [gyro]
        accels += [accel]

    gyros, accels = np.hstack(gyros), np.hstack(accels)

    return gyros, accels


def loadMetaData(path):
    metadata = {}
    with open(path) as jsonFile:
        data = json.load(jsonFile)
        for key, value in data.items():
            metadata[key] = value['value']
    
    return metadata


def loadMeanParams(path):
    meanParams = np.load(path)
    initPose = torch.from_numpy(meanParams['pose'])
    initPose = rotation_matrix_to_angle_axis(rot6d_to_rotmat(initPose)).unsqueeze(0).reshape(1, 72)
    initPose, initOrient = initPose[:, 3:], initPose[:, :3]
    initBetas = torch.from_numpy(meanParams['shape']).unsqueeze(0)
  
    return initPose, initBetas, initOrient


def loadCalibration(filename, cam_type='hd'):
    camera_infos = dict()

    with open(filename) as json_file:
        calib = json.load(json_file)
        datasource = calib['calibDataSource']
        cameras = calib['cameras']
        for camera in cameras:
            if camera['type'] != cam_type:
                continue

            camera_info = dict()
            camera_name = '_'.join((camera['type'], camera['name']))
            camera_info['camera_name'] = camera_name
            camera_info['camera_dist'] = np.array(camera['distCoef'])
            camera_info['camera_pose'] = np.array(camera['R'])
            camera_info['camera_intrinsics'] = np.array(camera['K'])
            camera_info['camera_transl'] = np.array(camera['t'])
            camera_info['resolution'] = camera['resolution']

            camera_infos[camera_name] = camera_info

    return camera_infos
