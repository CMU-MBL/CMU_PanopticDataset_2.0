from fitting.loss import ObjectiveFunction
from fitting.floop import FittingLoop

from utils import constants as _C
from utils.fitting_options import ArgsOptions
from utils.load_data import loadKeypoints, loadIMU, loadMetaData, loadMeanParams, loadCalibration
from utils.smpl import buildBodyModel
from utils.viz import generateVideo

import torch
import numpy as np

from tqdm import tqdm
import argparse
import os
import os.path as osp

# This setting is required to use pyrender with SSH access
os.environ['PYOPENGL_PLATFORM'] = 'egl'


def RunFitting(args, gtKeypoints, gtGyros, bodyModel):

    # Setup
    seqLen = gtKeypoints.shape[0]

    # Load Initialized SMPL Mean Parameters for Speeding-Up
    initPose, initBetas, initOrient = loadMeanParams(osp.join(_C.SMPL_DIR, _C.SMPL_MEAN_PARAMS))
    initPose, initBetas, initOrient = [torch.cat(seqLen * [tensor]).float().to(args.device) for tensor in [initPose, initBetas, initOrient]]

    # Prepare Weights of Loss Terms
    optWeightDict = {}
    for key, value in vars(args).items():
        if 'lw' in key:
            optWeightDict[key] = [w for w in value]

    keys = optWeightDict.keys()
    optWeights = [dict(zip(keys, vals)) for vals in zip(*(optWeightDict[k] for k in keys if optWeightDict[k] is not None))]
    
    for optWeight in optWeights:
        for key in optWeight:
            optWeight[key] = torch.tensor(optWeight[key]).float().to(args.device)

    # Prepare Ground-Truth Data
    gtKeypoints = gtKeypoints[..., _C.OP26_TO_OP25, :]
    gtKeypoints, gtKeypointsConf = gtKeypoints[..., :-1], gtKeypoints[..., -1:]
    gtKeypoints *= 1e-2     # Dimension matching (cm to m)
    
    # Build Objective Function
    objFunction = ObjectiveFunction(args)

    # Begin Fitting
    with FittingLoop(maxiters=args.maxiters, ftol=args.ftol, gtol=args.gtol) as floop:
        # Reset SMPL body model with Initialized Parameters
        bodyModel.reset_params(body_pose=initPose, betas=initBetas, global_orient=initOrient)
        
        # Create Optimizer
        optimParams = [bodyModel.body_pose, bodyModel.betas, bodyModel.global_orient]
        optimParams = list(filter(lambda x: x.requires_grad, optimParams))
        optimizer = torch.optim.Adam(optimParams, lr=args.lr, betas=args.betas)

        for optIdx, optWeight in enumerate(optWeights):
            # Adjust Learning Rate by Steps
            for g in optimizer.param_groups:
                if optIdx < 4:
                    g['lr'] = args.lr * (args.lr_decay_step ** optIdx)
                else:
                    g['lr'] = args.lr * (0.5 ** optIdx)

            optimizer.zero_grad()
            objFunction.resetOptWeights(optWeight)

            closure = floop.createClosure(optimizer, bodyModel, gtKeypoints, gtKeypointsConf, gtGyros, objFunction)

            if optIdx == args.flag1:
                objFunction.flag = 1
            if optIdx == args.flag2:
                objFunction.flag = 2

            loss = floop.forward(optimizer, closure, optimParams, stage=optIdx)

    # Get Fitted Model Output
    bodyModelOutput = bodyModel(body_pose=bodyModel.body_pose, betas=bodyModel.betas, global_orient=bodyModel.global_orient)

    # Visualize Optimization Results    
    if args.viz_results:
        if args.viz_cam_calib is not None:
            calibrations = loadCalibration(args.viz_cam_calib)
        
        # One sample view, you can modify this if you want
        calibration = calibrations['hd_00_11']

        vidName = osp.join(args.viz_dir, f'{args.subject}_{args.activity}.mp4')
        generateVideo(args, vidName, bodyModel, bodyModelOutput, gtKeypoints, gtKeypointsConf, calibration=None) 
    

def main():
    
    # Load Keypoints and IMU data
    dataDir = _C.DATA_DIR
    keypointsDir = osp.join(dataDir, args.subject, args.activity, _C.KEYPOINTS_FLDR)
    imuDir = osp.join(dataDir, args.subject, args.activity, _C.IMU_FLDR)
    gtKeypoints = loadKeypoints(keypointsDir)[0][0]
    gtGyros, _ = loadIMU(imuDir, args.imu_parts, gtKeypoints.shape[0])

    # Convert Numpy Array to Torch Tensor
    gtKeypoints = torch.from_numpy(gtKeypoints).float().to(args.device)
    gtGyros = torch.from_numpy(gtGyros).float().to(args.device)

    # Load Subject Info
    metadataDir = osp.join(dataDir, args.subject, _C.METADATA)
    metadata = loadMetaData(metadataDir)
    gender = 'male' if metadata['sex'] == 'M' else 'female'

    # Load SMPL body model
    SMPLDir = _C.SMPL_DIR
    modelDir = osp.join(SMPLDir, 'smpl')
    SMPLRegressor = osp.join(SMPLDir, _C.SMPL_REGRESSOR)
    bodyModel = buildBodyModel(modelDir, SMPLRegressor, gtKeypoints.shape[0], args.device, gender)

    RunFitting(args, gtKeypoints, gtGyros, bodyModel)


if __name__ == '__main__':
    # Load Fitting Options
    parser = ArgsOptions()
    args = parser.parse_args()

    assert args.subject is not None and args.activity is not None, \
            "Parse subject and activity information !"

    print(f"Run fitting for Subject {args.subject} | Activity {args.activity} \n\n")
    
    main()
