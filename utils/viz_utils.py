import numpy as np
import cv2
import matplotlib.pyplot as plt


"""Utility functions and constants for visualizing data"""


CONNECTIVITY_SET = [(18, 17), (17, 1), (1, 15), (15, 16),  # face
             (0, 1), (0, 2),    # body
             (5, 4), (4, 3), (3, 0),    # left arm
             (11, 10), (10, 9), (9, 0),     # right arm
             (8, 7), (7, 6), (6, 2),    # left leg
             (14, 13), (13, 12), (12, 2),   # right leg
             (19, 20), (19, 8), (8, 21),    # left foot
             (22, 23), (23, 14), (14, 24)]  # right foot


COLOR_SET = [(220,20,60), (220,20,60), (220,20,60), (220,20,60),  # face
        (153, 0, 0), (153, 0, 0),  # body
        (153, 153, 0), (153, 153, 0), (153, 102, 0),   # left arm
        (0, 153, 0), (0, 153, 0), (51, 153, 0),   # right arm
        (0, 153, 102), (0, 153, 153), (0, 153, 153),  # left leg
        (0, 51, 153), (0, 0, 153), (0, 0, 153),  # right leg
        (0,0,205), (0,0,205), (0,0,205),    # left foot
        (60,179,113), (60,179,113), (60,179,113)    # right foot
    ]



def projectKeypoints(x3d, img_res, K, R, t, conf=None, dist=None):
    """
    x3d : F * N * 3 numpy array
    K : 3 * 3 numpy array
    R : 3 * 3 numpy array
    t : 3 * 1 numpy array
    conf : F * N * 1 numpy array
    dist : 5 * 1 numpy array
    """

    imh, imw = img_res

    if conf is not None:
        confMask = conf > 1e-4
    else:
        confMask = np.ones_like(conf).astype(np.bool)

    x2d = np.zeros_like(x3d[:, :, :2])
    R2_criterion = np.zeros_like(x3d[:, :, 0])

    for J in range(x3d.shape[1]):
        """ J is joint index """
        x = np.dot(R, x3d[:, J].T) + t
        xp = x[:2] / x[2]

        if dist is not None:
            X2 = xp[0] * xp[0]
            Y2 = xp[1] * xp[1]
            XY = X2 * Y2
            R2 = X2 + Y2
            R4 = R2 * R2
            R6 = R4 * R2
            R2_criterion[:, J] = R2

            radial = 1.0 + dist[0] * R2 + dist[1] * R4 + dist[4] * R6
            tan_x = 2.0 * dist[2] * XY + dist[3] * (R2 + 2.0 * X2)
            tan_y = 2.0 * dist[3] * XY + dist[2] * (R2 + 2.0 * Y2)

            xp[0, :] = radial * xp[0, :] + tan_x
            xp[1, :] = radial * xp[1, :] + tan_y

        pt = np.dot(K[:2, :2], xp) + K[:2, 2:]
        x2d[:, J, :] = pt.T

    x2d = x2d.astype('int32')
    x_visible = np.logical_and(x2d[:, :, 0] >= 0, x2d[:, :, 0] < imw)
    y_visible = np.logical_and(x2d[:, :, 1] >= 0, x2d[:, :, 1] < imh)
    visible = np.logical_and(x_visible, y_visible)
    visMask = np.logical_and(visible, R2_criterion < 1.)
    
    mask = np.logical_and(confMask, visMask)

    return x2d, mask


def projectSMPL(bodyModel, bodyModelOutput, calibration):
    from scipy.spatial.transform import Rotation as _R

    device = bodyModel.device
    bp = torch.from_numpy(_R.from_rotvec(bodyModelOutput.body_pose.detach().cpu().numpy().reshape(-1, 3)).as_matrix()).float().to(device)
    bs = bodyModelOutput.betas
    go = _R.from_rotvec(bodyModelOutput.global_orient.detach().cpu().numpy).as_matrix()
    go = calibration['R'] @ go
    go = torch.from_numpy(go).float().to(device)

    bodyModelOutput = bodyModel(body_pose=bp, betas=bs, global_orient=go)
    
    return bodyModelOutput
