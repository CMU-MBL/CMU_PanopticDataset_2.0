from fitting.prior import MaxMixturePrior, L2Prior
from fitting.floop import SensorCalibration

from utils import constants as _C
from utils.conversion import angle_axis_to_rotation_matrix as Rodrigues
from utils.conversion import rotation_matrix_to_angle_axis as invRodrigues
from utils.conversion import deg2rad

import torch
from torch import nn
import numpy as np


def matchSensorAxis(synSensor, IMUParts):
    """Match synthetic IMU coordinate system to real IMU

    Args:
        synSensor: Synthetic IMU data
        IMUParts: name list of IMU used for optimization

    Return:
        synSensor: Calibrated (axis-matched) synthetic IMU
    """
    
    mapping = {'lbicep': 'la', 'lforearm': 'la', 'rbicep': 'ra', 'rforearm': 'ra'}
    device = synSensor.device
    matLA = torch.Tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).to(device)
    matRA = torch.Tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).to(device)
    matLL = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]]).to(device)
    matDict = {'la': matLA, 'ra': matRA, 'll': matLL}

    for idx, partName in enumerate(IMUParts):
        if not partName in mapping.keys():
            continue
        synSensor[:, idx] = synSensor[:, idx] @ matDict[mapping[partName]]

    return synSensor


def prepareTargetGyro(gyro, calibMatrix, fps=29.97):
    """Calculate groundtruth IMU data as rotation matrix

    Args:
        gyro: IMU gyroscope data in angle axis
        fps: frame per second of IMU data

    Return:
        rotation matrix
    """
    
    nSeq, nSen = gyro.shape[:2]
    device, dtype = gyro.device, gyro.dtype
    gyro_ = gyro * torch.Tensor([1, 1, 1]).to(device=device, dtype=dtype)
    gyro_ = deg2rad(gyro_) / fps

    gyroRotMat = Rodrigues(gyro_.view(-1, 3)).view(nSeq, nSen, 4, 4)[:, :, :3, :3]
    gyroRotMat = torch.transpose(calibMatrix, 1, 2) @ gyroRotMat @ calibMatrix
    gyroAngle = invRodrigues(gyroRotMat.view(-1, 3, 3)).view(nSeq, nSen, 3)

    return gyroRotMat, gyroAngle


def getGlobalOrientation(poses, parents):
    """Recover global orientation of body joints using kinematic tree

    Args:
        poses: pose parameters of SMPL model
        parents: kinematic tree defined by SMPL

    Return:
        pose in global orientation
    """

    results = []
    rootPose = Rodrigues(poses[:, 0])
    results += [rootPose]

    for i in range(1, parents.size(0)):
        locPose = Rodrigues(poses[:, i])
        globPose = results[parents[i]] @ locPose
        results += [globPose]

    results = [result.unsqueeze(1) for result in results]
    return torch.cat(results, dim=1)


def calculateSynGyro(output, bodyModel, IMUMap):
    """Calculate synthetic gyroscope data from given multi-frames SMPL parameters

    Args:
        output: SMPL model output with given parameters
        bodyModel: SMPL body model

    Return:
        synthetic IMU gyroscope data
    """
    
    poses = torch.cat((output.global_orient, output.body_pose), dim=-1)
    poses = poses.view(-1, 24, 3)

    globOrient = getGlobalOrientation(poses, bodyModel.parents)
    results = []
    for i in range(globOrient.size(1)):
        currGyro = torch.transpose(globOrient[:-1, i], 1, 2) @ globOrient[1:, i]
        results += [currGyro.unsqueeze(1)]
    
    gyroRotMat = torch.cat(results, dim=1)[:, IMUMap]
    gyroAngle = invRodrigues(gyroRotMat.view(-1, 4, 4)[:, :3]).view(-1, len(IMUMap), 3)

    return gyroRotMat, gyroAngle


def alignKeypoints(gtKeypoints, predKeypoints):
    """Aligning two set of 3D joints to their pelvis"""

    def centeringKeypoints(joints):
        """Rigid body translation to centering to pelvis jointss

        Args:
            joints: 3D keypoints
        
        Return:
            joints_: pelvis-centered joints

        """

        lhip, rhip = joints[:, 12].clone(), joints[:, 9].clone()
        pelv = (lhip + rhip)/2

        joints_ = joints - pelv.unsqueeze(1)
        return joints_


    gtKeypoints = centeringKeypoints(gtKeypoints)
    predKeypoints = centeringKeypoints(predKeypoints)

    return gtKeypoints, predKeypoints



class ObjectiveFunction(nn.Module):
    def __init__(self, args, **kwargs):
        super(ObjectiveFunction, self).__init__()

        self.rho = args.rho
        self.ignKeypointsIdx = args.ign_keypoints_idx
        self.IMUParts = args.imu_parts
        self.IMUMap = [_C.IMU_PARTS[part] for part in self.IMUParts]

        self.posePrior = MaxMixturePrior(dtype=torch.float, device=args.device)
        self.shapePrior = L2Prior(dtype=torch.float)

        self.register_buffer('lw_keypoints', torch.tensor(args.lw_keypoints, dtype=torch.float))
        self.register_buffer('lw_pprior', torch.tensor(args.lw_pprior, dtype=torch.float))
        self.register_buffer('lw_sprior', torch.tensor(args.lw_sprior, dtype=torch.float))
        self.register_buffer('lw_ptemp', torch.tensor(args.lw_ptemp, dtype=torch.float))
        self.register_buffer('lw_stemp', torch.tensor(args.lw_stemp, dtype=torch.float))
        self.register_buffer('lw_gyro', torch.tensor(args.lw_gyro, dtype=torch.float))

        self.device = args.device
        self.to(args.device)
        
        self.flag = 0
        self.calibMatrix = torch.cat([torch.eye(3).unsqueeze(0) for _ in self.IMUParts], dim=0).float().to(args.device)


    def resetOptWeights(self, optWeightDict):
        for key in optWeightDict:
            if hasattr(self, key):
                weightTensor = getattr(self, key)
                if 'torch.Tensor' in str(type(optWeightDict[key])):
                    weightTensor = optWeightDict[key].clone().detach()
                else:
                    weightTensor = torch.tensor(optWeightDict[key],
                                                 dtype=weightTensor.dtype,
                                                 device=weightTensor.device)
                setattr(self, key, weightTensor)


    def forward(self, bodyModel, bodyModelOutput, gtKeypoints, gtKeypointsConf, gtGyros, **kwargs):
        
        # Loss 1: Keypoints Distance Loss
        lossKeypoints, invalidMask, MPJPE = self.getKeypointsLoss(bodyModelOutput, gtKeypoints, gtKeypointsConf)

        # Loss 2-3: Pose and Shape Prior Loss
        lossPrior = self.getPriorLoss(bodyModelOutput, invalidMask)
        
        # Loss 4-5: Pose and Shape Temporal Loss
        lossTemp = self.getTemporalLoss(bodyModelOutput)
        
        # Loss 6: IMU-Video Fusion Loss
        lossGyro = self.getGyroLoss(gtGyros, bodyModelOutput, bodyModel, gtKeypointsConf)
       
        # Add All Losses
        lossTotal = lossKeypoints + lossPrior + lossTemp + lossGyro
        
        return lossTotal, MPJPE


    def getKeypointsLoss(self, bodyModelOutput, gtKeypoints, gtKeypointsConf):
        def robustifier(value):
            dist = torch.div(value**2, value**2 + self.rho ** 2)
            return self.rho ** 2 * dist
        
        # Clear Negative Confidence
        gtKeypointsConf[gtKeypointsConf < 0] = 0
        invalidMask = gtKeypointsConf.sum((-1, -2)) == 0

        # Regress Predicted Keypoints
        predKeypoints = bodyModelOutput.joints[:, _C.SMPL_TO_OP25]
        
        # Calculate Keypoints Loss
        gtKeypoints, predKeypoints = alignKeypoints(gtKeypoints, predKeypoints)
        distKeypoints = robustifier(gtKeypoints - predKeypoints)
        distKeypoints[:, self.ignKeypointsIdx] *= 0
        lossKeypoints = torch.sum(distKeypoints * gtKeypointsConf ** 2, dim=(1, 2))
        lossKeypoints = lossKeypoints.sum() * self.lw_keypoints

        # Calculate MPJPE (cm) for Results Visualization
        MPJPE = self.getMeanPerJointPositionError(gtKeypoints, predKeypoints, gtKeypointsConf)

        return lossKeypoints, invalidMask, MPJPE


    def getPriorLoss(self, bodyModelOutput, invalidMask):
        
        # Calculate Pose Prior
        lossPPrior = self.posePrior(bodyModelOutput.body_pose, bodyModelOutput.betas)
        lossPPrior[invalidMask] *= 0
        lossPPrior = lossPPrior.sum() * self.lw_pprior ** 2

        # Calculate Shape Prior
        lossSPrior = torch.sum(self.shapePrior(bodyModelOutput.betas)) * self.lw_sprior ** 2

        # Add Two Prior Losses
        lossPrior = lossPPrior + lossSPrior
        
        return lossPrior


    def getTemporalLoss(self, bodyModelOutput, **kwargs):
        
        # Calculate Pose Continuous Loss
        if self.flag > 0:
            rmatOrient = Rodrigues(bodyModelOutput.global_orient)
            diffOrient = rmatOrient[:-1].transpose(1, 2) @ rmatOrient[1:]
            diffOrient = invRodrigues(diffOrient[:, :3])
            diffOrient = torch.sum(diffOrient ** 2)

            rmatPose = Rodrigues(bodyModelOutput.body_pose.view(-1, 23, 3).view(-1, 3)).view(-1, 23, 4, 4)
            diffPose = rmatPose[:-1].transpose(2, 3) @ rmatPose[1:]
            diffPose = invRodrigues(diffPose[:, :, :3].view(-1, 3, 4)).view(-1, 23, 3)
            diffPose = torch.sum(diffPose ** 2)

            lossPTemp = (diffOrient + diffPose) * self.lw_ptemp
        
        else:
            # This loss is applied after initial few steps fitting
            lossPTemp = 0

        # Calculate Shape Consistency Loss
        lossSTemp = torch.sum(bodyModelOutput.betas.std(dim=0)) * self.lw_stemp
        
        # Add Two Temporal Losses
        lossTemp = lossPTemp + lossSTemp
        
        return lossTemp


    def getGyroLoss(self, gtGyros, bodyModelOutput, bodyModel, gtKeypointsConf):
       
        if self.lw_gyro == 0 or self.flag == 0:
            return 0

        # Prepare Real and Virtual IMUs
        realGyrosRotMat, realGyrosAngle = prepareTargetGyro(gtGyros, self.calibMatrix)
        synGyrosRotMat, synGyrosAngle = calculateSynGyro(bodyModelOutput, bodyModel, self.IMUMap)
        synGyrosAngle = matchSensorAxis(synGyrosAngle, self.IMUParts)
        synGyrosRotMat = Rodrigues(synGyrosAngle.view(-1, 3))[:, :3, :3].view(*realGyrosRotMat.shape)

        lossGyro = 0
        # Calculate Gyro Loss Only After Calibration
        if self.flag == 3:
            lossGyroCont = torch.sum((synGyrosAngle[1:] - synGyrosAngle[:-1]) ** 2)
            lossGyroDiff = torch.sum((synGyrosAngle - realGyrosAngle) ** 2, dim=-1)
            lossGyro = (lossGyroDiff.sum() + lossGyroCont) * self.lw_gyro
        
        if self.flag == 2:
            # Optimize Calibration Matrix
            confThresh = 6
            validIdx = gtKeypointsConf.squeeze(-1).sum(1) > confThresh
            validIdx = torch.logical_and(validIdx[1:], validIdx[:-1])
            
            CalibrationLoop = SensorCalibration(device=self.device)
            calibAngle = CalibrationLoop(realGyrosRotMat.clone()[validIdx], synGyrosRotMat.clone()[validIdx])
            calibMatrix = Rodrigues(calibAngle)[:, :3, :3]
            self.calibMatrix = calibMatrix.clone()
            
            # Change flag value so that do not repeat calibration
            self.flag = 3

        return lossGyro


    def getMeanPerJointPositionError(self, gtKeypoints, predKeypoints, gtKeypointsConf):
        validMask = (gtKeypointsConf > 1e-3)[:, :15, 0]
        validMaskWeight = torch.ones_like(validMask).float()
        validMaskWeight[~validMask] = 0
        MPJPE = torch.sqrt(((gtKeypoints[:, :15] - predKeypoints[:, :15]) ** 2).sum(-1)) * validMaskWeight * 1e2
        MPJPE = MPJPE.mean().item() 
        
        return MPJPE
