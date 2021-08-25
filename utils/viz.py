from utils import constants as _C
from utils.viz_utils import CONNECTIVITY_SET, COLOR_SET, projectKeypoints

import numpy as np
import torch

from tqdm import tqdm, trange
import cv2
import trimesh
import pyrender

import os
import os.path as osp


# # Draw Skeleton

def drawSkeleton(x, y, img):
    """ Draw 2D skeleton model """

    for idx, index_set in enumerate(CONNECTIVITY_SET):
        xs, ys = [], []
        for index in index_set:
            if (x[index] > 1e-5 and y[index] > 1e-5):
                xs.append(x[index])
                ys.append(y[index])

        if len(xs) == 2:
            # Draw line
            start = (xs[0], ys[0])
            end = (xs[1], ys[1])
            img = cv2.line(img, start, end, COLOR_SET[idx], 5)
        
    return img


# # Draw SMPL

_renderer = None
def getRenderer(viewpoint_width, viewpoint_height, point_size):
    global _renderer
    if _renderer is None or viewpoint_width != _renderer.viewpoint_width \
            or viewpoint_height != _renderer.viewpoint_height or point_size != _renderer.point_size:
         renderer = pyrender.OffscreenRenderer(
                 viewport_width=viewpoint_width, viewport_height=viewpoint_height, point_size=point_size)

    return renderer


def renderSMPL(vertices, faces, image, intrinsics, pose, transl, alpha=1.0):
    
    img_h, img_w = image.shape[:2]
    material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.5, 0.5, 0.5, 1.0)
            )

    # Generate SMPL vertices mesh
    mesh = trimesh.Trimesh(vertices, faces)

    # Default rotation of SMPL body model
    rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
    mesh.apply_transform(rot)

    mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

    scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
    scene.add(mesh, 'mesh')

    camera_pose = np.eye(4)
    camera_pose[:3, :3] = pose
    camera_pose[:3, 3] = transl
    camera = pyrender.IntrinsicsCamera(fx=intrinsics[0, 0], fy=intrinsics[1, 1],
                                       cx=intrinsics[0, 2], cy=intrinsics[1, 2])
    scene.add(camera, pose=camera_pose)

    # Light information
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
    light._generate_shadow_texture()
    light_pose = np.eye(4)

    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)

    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)

    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    renderer = getRenderer(img_w, img_h, 1.0)

    color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    valid_mask = (rend_depth > 0)[:,:,None]
    
    color = color.astype(np.float32) / 255.0
    valid_mask = (rend_depth > 0)[:,:,None]
   
    smpl_img = color[:, :, :3] * valid_mask * alpha
    
    output_img = smpl_img +\
                 valid_mask * image / 255 * (1-alpha) + (1 - valid_mask) * image / 255

    return (255 * output_img).astype(np.int16)


def generateVideo(args, vidName, bodyModel, bodyModelOutput, gtKeypoints, gtKeypointsConf, calibration=None):
    """Automatically generate output video of Fitting Results

    Args:
        vidName: Output video filename
        bodyModel: SMPL model
        bodyModelOutput: Fitting result SMPL body
        gtKeypoints: Ground-truth Keypoints
        calibration: Camera calibration matrices (optional)

    """

    gtKeypoints = gtKeypoints[:, _C.OP25_TO_OP26].detach().cpu()
    if gtKeypointsConf is not None:
        gtKeypointsConf = gtKeypointsConf[:, _C.OP25_TO_OP26].detach().cpu().squeeze(-1).numpy()

    outDir = args.viz_dir; os.makedirs(outDir, exist_ok=True)
    tmpDir = osp.join(outDir, 'images'); os.system(f'rm -rf {tmpDir}'); os.makedirs(tmpDir)
    
    if calibration is not None:
        camR, camT, camK, imgRes = calibration['camera_pose'], calibration['camera_transl'].T / 1e2, \
                                   calibration['camera_intrinsics'], calibration['resolution'][::-1]
        if args.viz_type == 'smpl':
            camT[0, 0] *= -1
    else:
        imgRes = args.viz_res
        camR , camT, camK = np.eye(3), np.array([[0, 1, 30]]), np.array([[5e3, 0, imgRes[1]/2], [0, 5e3, imgRes[0]/2], [0, 0, 1]])
        imgRes = args.viz_res

    imgBG = np.ones([*imgRes, 3]) * 255
    
    predKeypoints = bodyModelOutput.joints[:, _C.SMPL_TO_OP25][:, _C.OP25_TO_OP26].detach().cpu()
    predVertices = bodyModelOutput.vertices.detach().cpu()
    
    # Change camera coordinate to world coordinate
    gtKeypoints = (gtKeypoints @ camR.T)
    predVertices = (predVertices @ camR.T)
    predKeypoints = (predKeypoints @ camR.T)

    # Align Fitting results with Ground-Truth
    gtPelvis = gtKeypoints[:, [6, 12]].mean(1, keepdims=True)
    predPelvis = predKeypoints[:, [6, 12]].mean(1, keepdims=True)
    diffPelvis = gtPelvis - predPelvis
    
    predVertices += diffPelvis
    predKeypoints += diffPelvis

    # Project Keypoints onto Image
    predX2d, predMask = projectKeypoints(predKeypoints, imgRes, camK, np.eye(3), camT.T)
    gtX2d, gtMask = projectKeypoints(gtKeypoints, imgRes, camK, np.eye(3), camT.T, conf=gtKeypointsConf)
    predX2d[~predMask] = 0; gtX2d[~gtMask] = 0

    faces = bodyModel.faces
    seqLen = predVertices.shape[0]

    for frameIdx in tqdm(range(seqLen), desc='Generating videos ...', leave=False):
        imgName = osp.join(tmpDir, 'smpl_%05d.png'%frameIdx)
        
        if not 'keypoints' in args.viz_type:
            vertices = predVertices[frameIdx]
            img = renderSMPL(vertices, faces, imgBG.copy(), camK, np.eye(3), camT)

        else:
            x2d = gtX2d[frameIdx] if args.viz_type == 'gt-keypoints' else predX2d[frameIdx]
            img = drawSkeleton(x2d[:, 0], x2d[:, 1], imgBG.copy())
        
        cv2.imwrite(imgName, img.astype(np.int16))

    # if seqLength > 100:
    os.system(f'ffmpeg -y -hide_banner -loglevel error -framerate 30 -pattern_type glob -i "{tmpDir}/*.png" -c:v libx264 -pix_fmt yuv420p {vidName}')
    os.system(f'rm -rf {tmpDir}')
