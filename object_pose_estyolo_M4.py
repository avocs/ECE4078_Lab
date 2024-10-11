# estimate the pose of target objects detected
import numpy as np
import json
import os
import ast
import cv2
from YOLOv8.detector import ObjectDetector


# list of object fruits, assigned to labels
# OBJECT_TYPES = ['1', '2', '3', '4', '5']
OBJECT_TYPES = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']

# File repurposed to estimate object pose by directly obtaining the bounding boxes and robot poses as saved in the pred.txt and pred_n.png

# new estimate pose
def estimate_pose(camera_matrix, obj_info, robot_pose):    
    """
    input:
        camera_matrix: list, the intrinsic matrix computed from camera calibration (read from 'param/intrinsic.txt')
            |f_x, s,   c_x|
            |0,   f_y, c_y|
            |0,   0,   1  |
            (f_x, f_y): focal length in pixels
            (c_x, c_y): optical centre in pixels
            s: skew coefficient
        obj_info: list, an individual bounding box in an image (generated by get_bounding_box, [label,[x,y,width,height]])
        robot_pose: list, pose of robot corresponding to the image (read from 'lab_output/pred.txt', [x,y,theta])
    output:
        object_pose: dict, prediction of object pose
    """
    
    # TODO: actual sizes of objects [width, depth, height] in a dictionary instead of a list
    # --- this is taken from original object_pose_est.py,
    # object_dimensions_dict = {'1': [0.074,0.074,0.087], '2': [0.081, 0.081, 0.067], 
    #                           '3': [0.075, 0.075, 0.072], '4': [0.113, 0.067, 0.058], 
    #                           '5': [0.073, 0.073, 0.088]}
    object_dimensions_dict = {'1': [0.074,0.074,0.087], '2': [0.077, 0.077, 0.067], 
                              '3': [0.075, 0.075, 0.072], '4': [0.113, 0.067, 0.058], 
                              '5': [0.073, 0.073, 0.088]}

    #####################################################################
    # estimate object pose using bounding box and robot pose
    object_class = obj_info[0]     # get predicted object label of the box
    object_box = obj_info[1]       # get bounding box measures: [x,y,width,height]
    true_height = object_dimensions_dict[object_class][2]   # true height of the fruit object from the class label
    object_pose_rejected = False
    
    # camera parameters 
    focal_length_x = camera_matrix[0][0]  # focal length x
    f_y = camera_matrix[1][1]  # focal length y 
    principal_pt_x = camera_matrix[0][2]  # principal point x
    c_y = camera_matrix[1][2]  # principal point y

        
    ######### Replace with your codes #########
    # Compute pose of the object based on bounding box info and robot's pose
    # Use the height of the bounding box to estimate the distance Z_c
    box_width, box_height = object_box[2], object_box[3]
    
    # object depth as seen by camera, calculate wrt true height
    object_measured_depth = (true_height * focal_length_x) / box_height 
    
    # Compute the center of the bounding box 
    bbox_centre_x = object_box[0]

    # deviation of fruit in the x-axis with respect to the camera centreline
    X_c = (bbox_centre_x*object_measured_depth - principal_pt_x*object_measured_depth)/focal_length_x

    # Compute the angle of the fruit_object with respect to camera
    theta_cam = np.arctan(X_c/object_measured_depth)
    
    # Orientation of the robot
    theta_r = robot_pose[2][0]
    X_r = robot_pose[0][0]
    Y_r = robot_pose[1][0]
    
    # Compute the position of fruit in world frame
    theta_rc = theta_r - theta_cam
    diagonal = np.sqrt(object_measured_depth**2 + X_c**2)
    fruit_world_x = X_r + diagonal*np.cos(theta_rc)
    fruit_world_y = Y_r + diagonal*np.sin(theta_rc)
    
    # Save the estimated pose as a dictionary
    object_pose = {'x': fruit_world_x, 'y': fruit_world_y}
    print(f'Object_pose class {object_class}: {object_pose} || robot pose{X_r:.4f} {Y_r:.4f} {theta_r:.4f}')
    print(f"""
    Box Width: {box_width:.4f}  | Box Height: {box_height:.4f}  | objdepth_c: {object_measured_depth:.4f} | bboxx: {bbox_centre_x:.4f}
    X_fruitcam: {X_c:.4f}        | theta_fruitcam: {theta_cam:.4f}       | theta_fruitworld: {theta_rc:.4f}
    """)
    # object_pose_dict[object_list[object_num-1]] = object_pose
    ###########################################
    
    # TODO object depth filtering 
    object_depth_min_threshold = 0
    object_depth_max_threshold = 4
    if object_measured_depth < object_depth_min_threshold or object_measured_depth > object_depth_max_threshold:
        object_pose_rejected = True

    return object_pose, object_pose_rejected


# merge the estimations of the objects so that there are at most 1 estimate for each object type
# -- the default estimation was to take the first estimation of each object type 

def merge_estimations(object_map):
    object_est = {}
    fruit_est = {}
    num_per_object = 1  # max number of units per object type. We are only using 1 unit per fruit type
    outlier_threshold1 = 0.6
    outlier_threshold2 = 0.38
    filter_enabled = True

   # If the object_map is not empty
    if object_map:
        for fruit_number in range(1, len(OBJECT_TYPES) + 1):
            fruit_label = str(fruit_number)
            fruit_estimations = [object_map[key] for key in object_map if key.startswith(fruit_label)]

            # Check if there are estimations for this fruit type
            if fruit_estimations:
                # Extract x and y values from dictionaries
                x_values = [estimation['x'] for estimation in fruit_estimations]
                y_values = [estimation['y'] for estimation in fruit_estimations]

                # Calculate mean and standard deviation
                x_mean = np.mean(x_values)
                y_mean = np.mean(y_values)
                x_std = np.std(x_values)
                y_std = np.std(y_values)
                x_median = np.median(x_values)
                y_median = np.median(y_values)

                print(f'fruit {fruit_number}, std x :{x_std:.4f}, std y: {y_std:.4f}')
                fruit_est[f'{OBJECT_TYPES[fruit_number - 1]}'] = {'x': x_values, 'y': y_values}


                if filter_enabled:
                # Filter outliers
                    filtered_estimations1 = [estimation for estimation in fruit_estimations if
                                            abs(estimation['x'] - x_mean) <= outlier_threshold1 * x_std and
                                            abs(estimation['y'] - y_mean) <= outlier_threshold1 * y_std]
                    
                    # tighter filter for outliers 
                    filtered_estimations2 = [estimation for estimation in fruit_estimations if
                                            abs(estimation['x'] - x_mean) <= outlier_threshold2 * x_std and
                                            abs(estimation['y'] - y_mean) <= outlier_threshold2 * y_std]
                    print(f'filtered fruit {fruit_number}: {filtered_estimations2}')

                    # Calculate mean of filtered estimations
                    if filtered_estimations2:
                        print('estimations 2')
                        x_mean = np.mean([estimation['x'] for estimation in filtered_estimations2])
                        y_mean = np.mean([estimation['y'] for estimation in filtered_estimations2])
                        object_est[f'{OBJECT_TYPES[fruit_number - 1]}_fin'] = {'x': x_mean, 'y': y_mean}
                    elif filtered_estimations1:
                        print('estimations 1')
                        x_mean = np.mean([estimation['x'] for estimation in filtered_estimations1])
                        y_mean = np.mean([estimation['y'] for estimation in filtered_estimations1])
                        object_est[f'{OBJECT_TYPES[fruit_number - 1]}_fin'] = {'x': x_mean, 'y': y_mean}
                    else:
                        print('no filter..?')
                        object_est[f'{OBJECT_TYPES[fruit_number - 1]}_finnofilt'] = {'x': x_mean, 'y': y_mean}

                # default method, without filtering, to just round the mean to 1dp 
                else: 
                        print('filter disabled')
                        object_est[f'{OBJECT_TYPES[fruit_number - 1]}_finnofilt'] = {'x': x_mean, 'y': y_mean}
                        # object_est[f'{OBJECT_TYPES[fruit_number - 1]}_finnofilt'] = {'x': x_median, 'y': y_median}

    return object_est, fruit_est


# NOTE what the fuck is going on here
def live_fruit_pose_update(self):
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')    
    
    # a dictionary of all the saved detector outputs
    # image_poses should contain all the robot poses recorded and its corresponding image file taken
    image_poses = {}
    with open('lab_output/pred.txt', 'r') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line) # pose_dict = {pose, predfname}
            image_poses[pose_dict['predfname']] = pose_dict['pose'] # image_poses = {pred_n.png: robotpose}
    print(f'Reading data from {len(image_poses.keys())} images..')

    image_boundingboxes = {}
    with open('lab_output/bbox.txt', 'r') as fb:
        for line in fb.readlines():
            bbox_dict = ast.literal_eval(line)
            predfname =  bbox_dict['predfname']
            if predfname not in image_boundingboxes:
                image_boundingboxes[predfname] = []
            image_boundingboxes[predfname].append([bbox_dict['label'], bbox_dict['bbox']])


    # estimate pose of objects in each image

    # TODO variables labelled 'full' means all fruit bboxes are being used to estimate pose without filtering out too near/too far boxes
    model_path = os.path.join('YOLOv8', 'best_10k.pt')
    yolo = ObjectDetector(model_path)
    object_pose_dict = {}
    object_pose_full_dict = {}
    detected_type_list = []
    detected_type_full_list = []

    # for each image file, get the raw image's bounding boxes and robot pose info, estimate where the fruit is in the world frame, 
    # then add these estimations into an object_pose_dict.
    # object_pose_dict should collect the estimated poses of all objects across all pictures taken
    for image_path in image_poses.keys():
        input_image = cv2.imread(image_path)

        img = cv2.cvtColor(input_image,cv2.COLOR_RGB2BGR)

        # calls inference on the image and gets the bounding boxes coordinate results
        # bounding_boxes, bbox_img = yolo.detect_single_image(img)
        
        # TODO show bounding boxes and convert back to rgb for inferencing 
        # bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB)
        cv2.imshow('bbox', input_image)
        cv2.waitKey(0)

        # obtain robot pose and bounding boxes for the corresponding image
        robot_pose = image_poses[image_path]
        bounding_boxes = image_boundingboxes[image_path]

        # for each bounding box in the list of boxes obtainable from the image,
        for detection in bounding_boxes:
            # detection[0] is the class label
            label = detection[0]
            # occurence is a tally of how many times a label's bounding box has appeared across all images
            occurrence = detected_type_list.count(label)
            occurrence_full = detected_type_full_list.count(label)
            # for each bounding box that appears, estimate where it is on the map, then add the results as an entry to object_pose_dict
            object_pose_entry, object_pose_rejected = estimate_pose(camera_matrix, detection, robot_pose)
           
            # if the bounding box is not invalid, add this entry to the a filtered list to pass into merge
            # but this pose will be added to a general dictionary anyway
            if not object_pose_rejected:
                object_pose_dict[f'{label}_{occurrence}'] = object_pose_entry
                # adds the occurenece to the list, for tallying
                detected_type_list.append(label)
            
            object_pose_full_dict[f'{label}_{occurrence_full}'] = object_pose_entry
            detected_type_full_list.append(label)
            # print(f'detection0 {detection[0]}, occurence {occurrence}')

    # merge the estimations of the objects so that there are at most 1 estimations of each object type
    object_est = {}
    object_est, fruit_ests = merge_estimations(object_pose_dict)
    object_est_full, fruit_ests_full = merge_estimations(object_pose_full_dict)

    return object_est


def main(): 

    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')    
    
    # a dictionary of all the saved detector outputs
    # image_poses should contain all the robot poses recorded and its corresponding image file taken
    image_poses = {}
    with open('lab_output/pred.txt', 'r') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line) # pose_dict = {pose, predfname}
            image_poses[pose_dict['predfname']] = pose_dict['pose'] # image_poses = {pred_n.png: robotpose}
    print(f'Reading data from {len(image_poses.keys())} images..')

    image_boundingboxes = {}
    with open('lab_output/bbox.txt', 'r') as fb:
        for line in fb.readlines():
            bbox_dict = ast.literal_eval(line)
            predfname =  bbox_dict['predfname']
            if predfname not in image_boundingboxes:
                image_boundingboxes[predfname] = []
            image_boundingboxes[predfname].append([bbox_dict['label'], bbox_dict['bbox']])


    # estimate pose of objects in each image

    # TODO variables labelled 'full' means all fruit bboxes are being used to estimate pose without filtering out too near/too far boxes
    # model_path = os.path.join('YOLOv8', 'best_10k.pt')
    # yolo = ObjectDetector(model_path)
    object_pose_dict = {}
    object_pose_full_dict = {}
    detected_type_list = []
    detected_type_full_list = []

    # for each image file, get the raw image's bounding boxes and robot pose info, estimate where the fruit is in the world frame, 
    # then add these estimations into an object_pose_dict.
    # object_pose_dict should collect the estimated poses of all objects across all pictures taken
    for image_path in image_poses.keys():
        input_image = cv2.imread(image_path)

        img = cv2.cvtColor(input_image,cv2.COLOR_RGB2BGR)

        # calls inference on the image and gets the bounding boxes coordinate results
        # bounding_boxes, bbox_img = yolo.detect_single_image(img)
        
        # TODO show bounding boxes and convert back to rgb for inferencing 
        # bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB)
        cv2.imshow('bbox', input_image)
        cv2.waitKey(0)

        # obtain robot pose and bounding boxes for the corresponding image
        robot_pose = image_poses[image_path]
        bounding_boxes = image_boundingboxes[image_path]

        # for each bounding box in the list of boxes obtainable from the image,
        for detection in bounding_boxes:
            # detection[0] is the class label
            label = detection[0]
            # occurence is a tally of how many times a label's bounding box has appeared across all images
            occurrence = detected_type_list.count(label)
            occurrence_full = detected_type_full_list.count(label)
            # for each bounding box that appears, estimate where it is on the map, then add the results as an entry to object_pose_dict
            object_pose_entry, object_pose_rejected = estimate_pose(camera_matrix, detection, robot_pose)
           
            # if the bounding box is not invalid, add this entry to the a filtered list to pass into merge
            # but this pose will be added to a general dictionary anyway
            if not object_pose_rejected:
                object_pose_dict[f'{label}_{occurrence}'] = object_pose_entry
                # adds the occurenece to the list, for tallying
                detected_type_list.append(label)
            
            object_pose_full_dict[f'{label}_{occurrence_full}'] = object_pose_entry
            detected_type_full_list.append(label)
            # print(f'detection0 {detection[0]}, occurence {occurrence}')

    # merge the estimations of the objects so that there are at most 1 estimations of each object type
    object_est = {}
    object_est, fruit_ests = merge_estimations(object_pose_dict)
    object_est_full, fruit_ests_full = merge_estimations(object_pose_full_dict)

########################################################################################################
    # GENERATING OUTPUTS # 

    # print(fruit_ests)
    print("\nFruit Estimations:")
    print("----------------------------")
    for label, estimations in fruit_ests.items():
        print(f"{label:11} | Total: {len(estimations['x'])}", end="\n")
        print("========================")
        for x,y in zip(estimations['x'], estimations['y']):
            print(f"{x:7.4f}, {y:7.4f}\n", end="")
        print()
    
    print("\nObject Final Estimations:")
    print("----------------------------")
    for label, estimation in object_est.items():
        print(f"{label:20} | {estimation['x']:6.3f} {estimation['y']:6.3f}")

    # save object pose estimations into objects.txt
    objects_path = os.path.join('lab_output', 'objects.txt')
    with open(objects_path, 'w') as fo:
        json.dump(object_est, fo, indent=4)

    print('Estimations saved!')

    return object_est


# main loop
if __name__ == "__main__":


    # camera_matrix = np.ones((3,3))/2
    # this accesses the camera intrinsic matrix for estimation purposes

    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')    
    
    # a dictionary of all the saved detector outputs
    # image_poses should contain all the robot poses recorded and its corresponding image file taken
    image_poses = {}
    with open('lab_output/pred.txt', 'r') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line) # pose_dict = {pose, predfname}
            image_poses[pose_dict['predfname']] = pose_dict['pose'] # image_poses = {pred_n.png: robotpose}
    print(f'Reading data from {len(image_poses.keys())} images..')

    image_boundingboxes = {}
    with open('lab_output/bbox.txt', 'r') as fb:
        for line in fb.readlines():
            bbox_dict = ast.literal_eval(line)
            predfname =  bbox_dict['predfname']
            if predfname not in image_boundingboxes:
                image_boundingboxes[predfname] = []
            image_boundingboxes[predfname].append([bbox_dict['label'], bbox_dict['bbox']])


    # estimate pose of objects in each image

    # TODO variables labelled 'full' means all fruit bboxes are being used to estimate pose without filtering out too near/too far boxes
    # model_path = os.path.join('YOLOv8', 'best_10k.pt')
    # yolo = ObjectDetector(model_path)
    object_pose_dict = {}
    object_pose_full_dict = {}
    detected_type_list = []
    detected_type_full_list = []

    # for each image file, get the raw image's bounding boxes and robot pose info, estimate where the fruit is in the world frame, 
    # then add these estimations into an object_pose_dict.
    # object_pose_dict should collect the estimated poses of all objects across all pictures taken
    for image_path in image_poses.keys():
        input_image = cv2.imread(image_path)

        img = cv2.cvtColor(input_image,cv2.COLOR_RGB2BGR)

        # calls inference on the image and gets the bounding boxes coordinate results
        # bounding_boxes, bbox_img = yolo.detect_single_image(img)
        
        # TODO show bounding boxes and convert back to rgb for inferencing 
        # bbox_img = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB)
        cv2.imshow('bbox', input_image)
        cv2.waitKey(0)

        # obtain robot pose and bounding boxes for the corresponding image
        robot_pose = image_poses[image_path]
        bounding_boxes = image_boundingboxes[image_path]

        # for each bounding box in the list of boxes obtainable from the image,
        for detection in bounding_boxes:
            # detection[0] is the class label
            label = detection[0]
            # occurence is a tally of how many times a label's bounding box has appeared across all images
            occurrence = detected_type_list.count(label)
            occurrence_full = detected_type_full_list.count(label)
            # for each bounding box that appears, estimate where it is on the map, then add the results as an entry to object_pose_dict
            object_pose_entry, object_pose_rejected = estimate_pose(camera_matrix, detection, robot_pose)
           
            # if the bounding box is not invalid, add this entry to the a filtered list to pass into merge
            # but this pose will be added to a general dictionary anyway
            if not object_pose_rejected:
                object_pose_dict[f'{label}_{occurrence}'] = object_pose_entry
                # adds the occurenece to the list, for tallying
                detected_type_list.append(label)
            
            object_pose_full_dict[f'{label}_{occurrence_full}'] = object_pose_entry
            detected_type_full_list.append(label)
            # print(f'detection0 {detection[0]}, occurence {occurrence}')

    # merge the estimations of the objects so that there are at most 1 estimations of each object type
    object_est = {}
    object_est, fruit_ests = merge_estimations(object_pose_dict)
    object_est_full, fruit_ests_full = merge_estimations(object_pose_full_dict)

########################################################################################################
    # GENERATING OUTPUTS # 

    # print(fruit_ests)
    print("\nFruit Estimations:")
    print("----------------------------")
    for label, estimations in fruit_ests.items():
        print(f"{label:11} | Total: {len(estimations['x'])}", end="\n")
        print("========================")
        for x,y in zip(estimations['x'], estimations['y']):
            print(f"{x:7.4f}, {y:7.4f}\n", end="")
        print()
    
    print("\nObject Final Estimations:")
    print("----------------------------")
    for label, estimation in object_est.items():
        print(f"{label:20} | {estimation['x']:6.3f} {estimation['y']:6.3f}")

    # save object pose estimations into objects.txt
    objects_path = os.path.join('lab_output', 'objects.txt')
    with open(objects_path, 'w') as fo:
        json.dump(object_est, fo, indent=4)

    print('Estimations saved!')