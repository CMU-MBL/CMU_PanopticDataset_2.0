import torch
import numpy as np

import smplx
from smplx import SMPL as _SMPL
from smplx.utils import SMPLOutput as ModelOutput 
from smplx.lbs import vertices2joints


# Map joints to SMPL joints
JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
}


JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose', 'OP Neck', 'OP RShoulder', 'OP RElbow',
'OP RWrist', 'OP LShoulder', 'OP LElbow', 'OP LWrist',
'OP MidHip', 'OP RHip', 'OP RKnee', 'OP RAnkle',
'OP LHip', 'OP LKnee', 'OP LAnkle', 'OP REye',
'OP LEye', 'OP REar', 'OP LEar', 'OP LBigToe',
'OP LSmallToe', 'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
]


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, joint_regressor=None, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        if joint_regressor is not None:
            J_regressor_extra = np.load(joint_regressor)
            self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        base_joints = smpl_output.joints.clone()[:, self.joint_map]
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([base_joints, extra_joints], dim=1)
        
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output


def buildBodyModel(body_model_folder, J_Regressor, batch_size, device, gender):
    
    body_model = SMPL(J_Regressor,
                      body_model_folder,
                      gender=gender,
                      batch_size=batch_size,
                      create_transl=False).to(device)

    return body_model
