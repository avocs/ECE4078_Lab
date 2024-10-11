# measure performance of target detection and pose estimation
import ast
import json
import numpy as np

# read in the object poses
def parse_map(fname: str) -> dict:
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline()) 
        redapple, greenapple, orange, mango, capsicum = [], [], [], [], []

        # remove unique id of targets of the same type 
        for key in gt_dict:
            if key.startswith('redapple'):
                redapple.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('greenapple'):
                greenapple.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('orange'):
                orange.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('mango'):
                mango.append(np.array(list(gt_dict[key].values()), dtype=float))
            elif key.startswith('capsicum'):
                capsicum.append(np.array(list(gt_dict[key].values()), dtype=float))
    
    # if more than 1 estimation is given for a target type, only the first estimation will be used
    num_per_target = 1 # max number of units per target type. We are only use 1 unit per fruit type
    if len(redapple) > num_per_target:
        redapple = redapple[0:num_per_target]
    if len(greenapple) > num_per_target:
        greenapple = greenapple[0:num_per_target]
    if len(orange) > num_per_target:
        orange = orange[0:num_per_target]
    if len(mango) > num_per_target:
        mango = mango[0:num_per_target]
    if len(capsicum) > num_per_target:
        capsicum = capsicum[0:num_per_target]

    return redapple, greenapple, orange, mango, capsicum


# compute the Euclidean distance between each target and its closest estimation and returns the average over all targets
def compute_dist(gt_list, est_list):
    gt_list = gt_list
    est_list = est_list
    dist_av = 0
    dist_list = []
    dist = []
    for gt in gt_list:
        # find the closest estimation for each target
        for est in est_list:
            dist.append(np.linalg.norm(gt-est)) # compute Euclidean distance
        dist.sort()
        dist_list.append(dist[0]) # distance between the target and its closest estimation
        dist = []
    dist_av = sum(dist_list)/len(dist_list) # average distance
    return dist_av

# main program
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Matching the estimated map and the true map")
    parser.add_argument("truth", type=str, help="The ground truth file name.")
    parser.add_argument("est", type=str, help="The estimate file name.")
    args, _ = parser.parse_known_args()

    # read in ground truth and estimations
    redapple_gt, greenapple_gt, orange_gt, mango_gt, capsicum_gt = parse_map(args.truth)
    redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = parse_map(args.est)

    # #TODO added this for checks 
    # print(f'gt: redapple: {redapple_gt}, greenapple: {greenapple_gt}, orange: {orange_gt}, mango: {mango_gt}, capsicum: {capsicum_gt}')
    # print(f'est: redapple: {redapple_est}, greenapple: {greenapple_est}, orange: {orange_est}, mango: {mango_est}, capsicum: {capsicum_est}')
    
  ##########################################################################  
    # TODO printing is done for the sake of visualisation cuz i am going insane
    # Print the header
    object_labels = ["redapple", "greenapple", "orange", "mango", "capsicum"]
    object_data = [
        (redapple_gt, redapple_est),
        (greenapple_gt, greenapple_est),
        (orange_gt, orange_est),
        (mango_gt, mango_est),
        (capsicum_gt, capsicum_est)
    ]
    object_dict = {label: data for label, data in zip(object_labels, object_data)}
    print("\n##############################################################")
    print("    Object  |  Ground Truth  |    Estimated    | Average Distance")
    print("------------|----------------|-----------------|------------------")

    # Print the data for each object
    for label, (gt, est) in object_dict.items():
        # print(label)
        if gt and est:
            distance = np.linalg.norm(np.array(gt) - np.array(est))
            print(f"{label:11} | {gt[0][0]:6} {gt[0][1]:6}  | {est[0][0]:6.2f} {est[0][1]:6.2f} | {distance:.4f}")
        else: 
            print(f"no data found for  {label:11}")
    print("\n##############################################################")

#########################################################################

    # compute average distance between a target and its closest estimation
    redapple_dist = compute_dist(redapple_gt,redapple_est)
    greenapple_dist = compute_dist(greenapple_gt,greenapple_est)
    orange_dist = compute_dist(orange_gt, orange_est)
    mango_dist = compute_dist(mango_gt, mango_est)
    capsicum_dist = compute_dist(capsicum_gt, capsicum_est)
    
    av_dist = (redapple_dist+greenapple_dist+orange_dist+mango_dist+capsicum_dist)/5
    
    print("Average distances between the targets and the closest estimations:")
    print("redapple = {}, greenapple = {}, orange = {}, mango = {}, capsicum = {}".format(redapple_dist,greenapple_dist,orange_dist,mango_dist,capsicum_dist))
    print("estimation error: ", av_dist)     