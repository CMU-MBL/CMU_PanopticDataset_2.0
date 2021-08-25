import torch
from utils import constants as _C

import argparse

class ArgsOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Optimization fitting options')

        # Target options
        self.parser.add_argument('--subject', type=str, default=None, help='Target subject to fit')
        self.parser.add_argument('--activity', type=str, default=None, help='Target activity for the subject')
        
        # Optimization options
        self.parser.add_argument('--device', type=str, default='cuda', help='Device for the computation')
        self.parser.add_argument('--maxiters', type=int, default=100, help='Maximum iterations per step')
        self.parser.add_argument('--ftol', type=float, default=0, help='Convergence criterion 1')
        self.parser.add_argument('--gtol', type=float, default=0, help='Convergence criterion 2')
        self.parser.add_argument('--lr', type=float, default=5e-2, help='Learning rate')
        self.parser.add_argument('--betas', default=(0.9, 0.999), help='Betas for Adam optimizer')
        self.parser.add_argument('--lr-decay-step', default=0.8, help='Learning rate decay by each step')
        
        # Fitting options
        self.parser.add_argument('--rho', type=float, default=1e2, help='Weight for robustifier')
        self.parser.add_argument('--ign-keypoints-idx', nargs='+', type=int, default=[])
        self.parser.add_argument('--flag1', type=int, default=1, help='Step number to initiate 1st optim flag')
        self.parser.add_argument('--flag2', type=int, default=3, help='Step number to initiate 2nd optim flag')

        # Weight options
        self.parser.add_argument('--lw-keypoints', nargs='+', type=float, default=[1e4, 1e4, 1e4, 1e4, 1e4])
        self.parser.add_argument('--lw-pprior', nargs='+', type=float, default=[10, 5, 2, 2, 1])
        self.parser.add_argument('--lw-sprior', nargs='+', type=float, default=[100, 20, 10, 10, 5])
        self.parser.add_argument('--lw-ptemp', nargs='+', type=float, default=[0, 0, 5e2, 5e2, 5e2])
        self.parser.add_argument('--lw-stemp', nargs='+', type=float, default=[2e4, 2e4, 1e4, 1e4, 1e4])
        self.parser.add_argument('--lw-gyro', nargs='+', type=float, default=[0, 0, 0, 0, 2e4])

        # Data options
        self.parser.add_argument('--imu-parts', nargs='+', type=str, 
                                 default=['lbicep', 'lfoot', 'lforearm', 'lhand', 'lshank', 'lthigh', 
                                           'rbicep', 'rfoot', 'rforearm', 'rhand', 'rshank', 'rthigh'])

        # Post analysis options
        self.parser.add_argument('--viz-results', default=False, action='store_true')
        self.parser.add_argument('--viz-dir', type=str, default='output', help='Folder directory for output')
        self.parser.add_argument('--viz-type', type=str, default='smpl', choices=['smpl', 'gt-keypoints', 'pred-keypoints'], help='Visualization type')
        self.parser.add_argument('--viz-res', nargs=2, type=int, default=(720, 1080), help='Resolution of visualization')
        self.parser.add_argument('--viz-cam-calib', type=str, default=None, help='Camera calibration for visualization')


    def parse_args(self, attrDict=None, **kwargs):
        self.args = self.parser.parse_args()

        if attrDict is not None:
            for key, value in attrDict.items():
                setattr(self.args, key, value)

        if self.args.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA is unavailable at your system ! Change device to CPU")
            self.args.device = 'cpu'

        opts = vars(self.args)
        print('\n\n' + "#" * 20+ " Optimization Configurations " + "#" * 20)
        for key, value in opts.items():
            if key in ['subject', 'activity']:
                continue
            print(f"{key.upper()}: {value}")
        print("#" * 69 + "\n\n")

        return self.args

