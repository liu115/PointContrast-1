import numpy as np
import os
import cv2


def load_pointcloud(data_root, scene_name, frame_id):

    # depth_intrinsic = np.loadtxt(opt.input_path + '/intrinsic/intrinsic_depth.txt')
    depth_intrinsic = np.loadtxt(os.path.join(data_root, scene_name, 'intrinsic_depth.txt'))

    pose_fn = os.path.join(data_root, scene_name, 'pose', f"{frame_id}.txt")
    depth_fn = os.path.join(data_root, scene_name, 'depth', f"{frame_id}.png")
    # for ind, (pose, depth) in enumerate(zip(poses, depths)):
    # name = os.path.basename(pose).split('.')[0]
    # print('='*50, ': {}'.format(pose))
    depth_img = cv2.imread(depth_fn, -1)  # read 16bit grayscale image
    pose = np.loadtxt(pose_fn)
    # print('Camera pose: ')
    # print(pose)

    depth_shift = 1000.0
    x,y = np.meshgrid(np.linspace(0,depth_img.shape[1]-1,depth_img.shape[1]), np.linspace(0,depth_img.shape[0]-1,depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:,:,0] = x
    uv_depth[:,:,1] = y
    uv_depth[:,:,2] = depth_img/depth_shift
    uv_depth = np.reshape(uv_depth, [-1,3])
    uv_depth = uv_depth[np.where(uv_depth[:,2]!=0),:].squeeze()

    intrinsic_inv = np.linalg.inv(depth_intrinsic)
    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = depth_intrinsic[0,3]
    by = depth_intrinsic[1,3]
    point_list = []
    n = uv_depth.shape[0]
    points = np.ones((n,4))
    X = (uv_depth[:,0]-cx)*uv_depth[:,2]/fx + bx
    Y = (uv_depth[:,1]-cy)*uv_depth[:,2]/fy + by
    points[:,0] = X
    points[:,1] = Y
    points[:,2] = uv_depth[:,2]
    points_world = np.dot(points, np.transpose(pose))

    return points_world[:, :3]

