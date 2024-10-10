import os
import json
import numpy as np
from copy import deepcopy

import matplotlib.pyplot as plt 

def parse_map(fname):
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
    aruco_dict = {}
    object_dict = {}

    for key in gt_dict:
        # read SLAM map
        if key.startswith('aruco'):
            aruco_num = int(key.strip('aruco')[:-2])
            aruco_dict[aruco_num] = np.reshape([gt_dict[key]['x'], gt_dict[key]['y']], (2, 1))
        
        # read object map
        else:
            object_type = key.split('_')[0]
            if object_type not in object_dict:
                object_dict[object_type] = np.array([[gt_dict[key]['x'], gt_dict[key]['y']]])
            else:
                object_dict[object_type] = np.append(object_dict[object_type], [[gt_dict[key]['x'], gt_dict[key]['y']]], axis=0)
    
    return aruco_dict, object_dict

# for SLAM evaluation
def match_aruco_points(aruco0: dict, aruco1: dict):
    points0 = []
    points1 = []
    keys = []
    for key in aruco0:
        if not key in aruco1:
            continue

        points0.append(aruco0[key])
        points1.append(aruco1[key])
        keys.append(key)
        
    return keys, np.hstack(points0), np.hstack(points1)


def solve_umeyama2d(points1, points2):
    # Solve the optimal transform such that
    # R(theta) * p1_i + t = p2_i

    assert (points1.shape[0] == 2)
    assert (points1.shape[0] == points2.shape[0])
    assert (points1.shape[1] == points2.shape[1])

    # Compute relevant variables
    num_points = points1.shape[1]
    mu1 = 1 / num_points * np.reshape(np.sum(points1, axis=1), (2, -1))
    mu2 = 1 / num_points * np.reshape(np.sum(points2, axis=1), (2, -1))
    sig1sq = 1 / num_points * np.sum((points1 - mu1) ** 2.0)
    sig2sq = 1 / num_points * np.sum((points2 - mu2) ** 2.0)
    Sig12 = 1 / num_points * (points2 - mu2) @ (points1 - mu1).T

    # Use the SVD for the rotation
    U, d, Vh = np.linalg.svd(Sig12)
    S = np.eye(2)
    if np.linalg.det(Sig12) < 0:
        S[-1, -1] = -1

    # Return the result as an angle and a 2x1 vector
    R = U @ S @ Vh
    theta = np.arctan2(R[1, 0], R[0, 0])
    x = mu2 - R @ mu1

    return theta, x

def apply_transform(theta, x, points):
    # Apply an SE(2) transform to a set of 2D points
    assert (points.shape[0] == 2)

    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    points_transformed = R @ points + x
    
    return points_transformed

def compute_slam_rmse(points1, points2):
    # Compute the RMSE between two matched sets of 2D points.
    assert (points1.shape[0] == 2)
    assert (points1.shape[0] == points2.shape[0])
    assert (points1.shape[1] == points2.shape[1])
    num_points = points1.shape[1]
    residual = (points1 - points2).ravel()
    MSE = 1.0 / num_points * np.sum(residual ** 2)

    return np.sqrt(MSE)

# for object pose estimation evaluation
def compute_object_est_error(gt_list, est_list):
    """Compute the object target pose estimation error based on Euclidean distance

    If there are more estimations than the number of targets (e.g. only 1 target orange, but detected 2),
        then take the average error of the 2 detections

    if there are fewer estimations than the number of targets (e.g. 2 target oranges, but only detected 1),
        then return [MAX_ERROR, error with the closest target]

    @param gt_list: target ground truth list
    @param est_list: estimation list
    @return: error of all the objects
    """

    MAX_ERROR = 1

    object_errors = {}

    for object_type in gt_list:
        n_gt = len(gt_list[object_type])  # number of targets in this fruit type

        type_errors = []
        for i, gt in enumerate(gt_list[object_type]):
            dist = []
            try:
                for est in est_list[object_type]:
                    dist.append(np.linalg.norm(gt - est))  # compute Euclidean distance
    
                n_est = len(est_list[object_type])
    
                # if this fruit type has been detected
                if len(dist) > 0:
                    if n_est > n_gt:    # if more estimation than target, take the mean error
                        object_errors[object_type + '_{}'.format(i)] = np.round(np.mean(dist), 5)
                    elif n_est < n_gt:  # see below
                        type_errors.append(np.min(dist))
                    else:   # for normal cases, n_est == n_gt, take the min error
                        object_errors[object_type + '_{}'.format(i)] = np.round(np.min(dist), 5)
            except:   # if there is no estimation for this fruit type
                for j in range(n_gt):
                    object_errors[object_type + '_{}'.format(j)] = MAX_ERROR

        if len(type_errors) > 0:    # for the n_est < n_gt scenario
            type_errors = np.sort(type_errors)
            for i in range(len(type_errors) - 1):
                object_errors[object_type + '_{}'.format(i+1)] = np.round(type_errors[i], 5)
            object_errors[object_type + '_{}'.format(0)] = MAX_ERROR

    return object_errors

def align_object_poses(theta, x, objects_est):
    objects = deepcopy(objects_est)

    for object in objects:
        poses = []
        for pos in objects[object]:
            pos = np.reshape(pos, (2, 1))
            pos = apply_transform(theta, x, pos)
            pos = np.reshape(pos, (1, 2))[0]

            poses.append(pos)

        objects[object] = poses

    return objects

####################################
# main loop
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Matching the estimated map and the true map')
    parser.add_argument('--truemap', type=str, default='truemap.txt')
    parser.add_argument('--slam-est', type=str, default='lab_output/slam.txt')
    parser.add_argument('--object-est', type=str, default='lab_output/objects.txt')
    args, _ = parser.parse_known_args()

    aruco_gt, object_gt = parse_map(args.truemap)
    
    slam_only, object_only = False, False
    if os.path.exists(args.slam_est):
        aruco_est, _ = parse_map(args.slam_est)
        if len(aruco_est) == 0:
            object_only = True
    else:
        object_only = True
    
    if os.path.exists(args.object_est):
        _, object_est = parse_map(args.object_est)
        if len(object_est) == 0:
            slam_only = True
    else:
        slam_only = True


    if slam_only:
        # only evaluate SLAM
        taglist, slam_est_vec, slam_gt_vec = match_aruco_points(aruco_est, aruco_gt)
        theta, x = solve_umeyama2d(slam_est_vec, slam_gt_vec)
        slam_est_vec_aligned = apply_transform(theta, x, slam_est_vec)
        
        slam_rmse_raw = compute_slam_rmse(slam_est_vec, slam_gt_vec)
        slam_rmse_aligned = compute_slam_rmse(slam_est_vec_aligned, slam_gt_vec)

        print(f'The SLAM RMSE before alignment = {np.round(slam_rmse_raw, 5)}')
        print(f'The SLAM RMSE after alignment = {np.round(slam_rmse_aligned, 5)}')

    elif object_only:
        object_est_errors = compute_object_est_error(object_gt, object_est)
        print('Object pose estimation errors:')
        print(json.dumps(object_est_errors, indent=4))
    
    else:
        # evaluate SLAM
        taglist, slam_est_vec, slam_gt_vec = match_aruco_points(aruco_est, aruco_gt)
        theta, x = solve_umeyama2d(slam_est_vec, slam_gt_vec)
        slam_est_vec_aligned = apply_transform(theta, x, slam_est_vec)


        # TODO fuck around by dra, this should be added into the eval slam only 
        idx = np.argsort(taglist)
        gt_vec = slam_gt_vec[:, idx]
        us_vec = slam_est_vec[:, idx]
        us_vec_aligned = slam_est_vec_aligned

        slam_rmse_raw = compute_slam_rmse(slam_est_vec, slam_gt_vec)
        slam_rmse_aligned = compute_slam_rmse(slam_est_vec_aligned, slam_gt_vec)

        # TODO added by dra
        print()
        print("The following parameters optimally transform the estimated points to the ground truth.")
        print("Rotation Angle: {}".format(theta))
        print("Translation Vector: ({}, {})".format(x[0,0], x[1,0]))

        print(f'The SLAM RMSE before alignment = {np.round(slam_rmse_raw, 5)}')
        print(f'The SLAM RMSE after alignment = {np.round(slam_rmse_aligned, 5)}')

        print('----------------------------------------------')
        
        # evaluate object pose estimation errors

        # align the object poses using the transform computed from SLAM
        object_est_aligned = align_object_poses(theta, x, object_est)

        object_est_errors_raw = compute_object_est_error(object_gt, object_est)
        object_est_errors_aligned = compute_object_est_error(object_gt, object_est_aligned)

        print('Object pose estimation errors before alignment:')
        print(json.dumps(object_est_errors_raw, indent=4))
        print('Object pose estimation errors after alignment:')
        print(json.dumps(object_est_errors_aligned, indent=4))
        
        err_lst, err_lst_aligned = [], []
        for object_err, object_err_aligned in zip(object_est_errors_raw, object_est_errors_aligned):
            err = object_est_errors_raw[object_err]
            err_aligned = object_est_errors_aligned[object_err_aligned]
            err_lst.append(err)
            err_lst_aligned.append(err_aligned)
        
        print(f'Average object pose estimation error before alignment: {np.mean(err_lst)}')
        print(f'Average object pose estimation error after alignment: {np.mean(err_lst_aligned)}')


        # TODO sandra fucks around here
        # this works! 
        ax = plt.gca()
        ax.scatter(gt_vec[0,:], gt_vec[1,:], marker='o', color='C0', s=100)
        ax.scatter(us_vec_aligned[0,:], us_vec_aligned[1,:], marker='x', color='C1', s=100)
        for i in range(len(taglist)):
            ax.text(gt_vec[0,i]+0.05, gt_vec[1,i]+0.05, taglist[i], color='C0', size=12)
            ax.text(us_vec_aligned[0,i]+0.05, us_vec_aligned[1,i]+0.05, taglist[i], color='C1', size=12)
        plt.title('Arena')
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.set_xticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        ax.set_yticks([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
        plt.legend(['Real','Pred'])
        plt.grid()
        plt.show()