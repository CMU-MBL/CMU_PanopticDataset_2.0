from utils.conversion import angle_axis_to_rotation_matrix as Rodrigues

import torch
from torch import nn
import numpy as np
from tqdm import tqdm


class FittingLoop(object):
    def __init__(self, summary_steps=1, maxiters=100, ftol=2e-09, gtol=1e-07, **kwargs):
        super(FittingLoop, self).__init__()
        
        self.summary_steps = summary_steps
        self.maxiters = maxiters
        self.ftol = ftol
        self.gtol = gtol

    def __enter__(self):
        self.steps = 0
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def forward(self, optimizer, closure, params, stage=None, **kwargs):

        def rel_loss_change(prev_loss, curr_loss):
            return (prev_loss - curr_loss) / max(np.abs(prev_loss), np.abs(curr_loss), 1)
        
        with tqdm(total=self.maxiters, desc=f'Stage {stage+1}', leave=False) as progBar:
            for n in range(self.maxiters):
                loss, MPJPE = optimizer.step(closure)
                if (torch.isnan(loss).sum() > 0 and 
                    torch.isinf(loss).sum() > 0):
                    print("Inappropriate loss value, break the loop !")
                    break
                
                # Firt convergence criterion
                if self.ftol > 0 and n > 0:
                    relLossChange = rel_loss_change(prevLoss, loss.item())
                    if relLossChange < self.ftol:
                        break
                
                # Second convergence criterion
                if all([torch.abs(var.grad.view(-1).max()).item() < self.gtol
                        for var in params if var.grad is not None]):
                    break

                prevLoss = loss.item()
                progBar.update(1)
                progBar.set_postfix_str('MPJPE: %.1f cm'%MPJPE)

        return prevLoss

    def createClosure(self, optimizer, bodyModel, gtKeypoints, gtKeypointsConf, gtGyros, objFunction, **kwargs):
        
        def closure(backward=True):
            if backward:
                optimizer.zero_grad()
            
            bodyModelOutput = bodyModel(return_verts=True,
                                        body_pose=None,
                                        return_full_pose=True)
            
            loss, MPJPE = objFunction(bodyModel, bodyModelOutput, gtKeypoints, gtKeypointsConf, gtGyros, **kwargs)            
            if backward:
                loss.backward()
            
            self.steps += 1

            return loss, MPJPE

        return closure


class SensorCalibration(nn.Module):
    def __init__(self, lw_calib=[10, 10, 10], 
                 lr=1e-2, maxiters=150,
                 dtype=torch.float, device='cuda', **kwargs):

        super(SensorCalibration, self).__init__()

        self.lw_calib = lw_calib
        self.lr = lr
        self.maxiters = maxiters
        self.device = device
        self.dtype = dtype


    def lossFunction(self, orgE, trgE, calibAngle, lw):
        totalLoss = 0
        calibE = Rodrigues(calibAngle).unsqueeze(0)[:, :, :3, :3]
        for idx in range(orgE.shape[1]):
            predE = torch.transpose(calibE[:, idx], 1, 2) @ orgE[:, idx] @ calibE[:, idx]
            loss = torch.sum((predE - trgE[:, idx])**2) * lw
            totalLoss += loss

        return totalLoss


    def forward(self, orgGyros, trgGyros):

        optWeightDict = {'calib_loss': [w for w in self.lw_calib]}
        calibAngle = torch.rand((orgGyros.shape[1], 3), device=self.device, dtype=self.dtype)
        calibAngle.requires_grad = True 
        optimParams = [calibAngle]
        optimizer = torch.optim.Adam(optimParams, lr=self.lr)

        with torch.autograd.enable_grad():
            for step in tqdm(range(len(optWeightDict['calib_loss'])), leave=False, desc='Calibrating sensor...'):
                with tqdm(range(self.maxiters), leave=False) as progBar:
                    for _ in range(self.maxiters):
                        optimizer.zero_grad()
                        loss = self.lossFunction(orgGyros, trgGyros, calibAngle, optWeightDict['calib_loss'][step])
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        msg = 'Loss : %.2f'%(loss.item())
                        progBar.set_postfix_str(msg)
                        progBar.update(1); progBar.refresh()

                for g in optimizer.param_groups:
                    curr_lr = g['lr']
                    g['lr'] = curr_lr * 0.9

        return calibAngle.detach()
