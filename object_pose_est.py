# estimate the pose of a detected object
import os
import ast
import PIL
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from machinevisiontoolbox import Image


# use the machinevision toolbox to get the bounding box of the detected object(s) in an image
# -- for our version, this function is no longer needed, since model.predict can directly return the bounding boxes
def get_bounding_box(object_number, image_path):
    image = PIL.Image.open(image_path).resize((640,480), PIL.Image.Resampling.NEAREST)
    object = Image(image)==object_number
    blobs = object.blobs()
    [[u1,u2],[v1,v2]] = blobs[0].bbox # bounding box
    width = abs(u1-u2)
    height = abs(v1-v2)
    center = np.array(blobs[0].centroid).reshape(2,)
    box = [center[0], center[1], int(width), int(height)] # box=[x,y,width,height]
    return box

# read in the list of detection results with bounding boxes and their matching robot pose info
def get_image_info(file_path, image_poses):
    # there are at most five types of objects in each image
    object_lst_box = [[], [], [], [], []]
    object_lst_pose = [[], [], [], [], []]
    completed_img_dict = {}

    # add the bounding box info of each object in each image
    # object labels: 1 = redapple, 2 = greenapple, 3 = orange, 4 = mango, 5 = capsicum, 0 = not_a_object
    print(f'fpath {file_path}')
    # iterates through all unique pixel values in the grayscale version of the colour-map image
    img_vals = set(Image(file_path, grey=True).image.reshape(-1))
    for object_num in img_vals:
        # if the pixel is something other than 0 (either of the labels)
        if object_num > 0:
            try:
                # get the bounding box corresponding to the object label
                box = get_bounding_box(object_num, file_path) # [x,y,width,height]
                # get the pose of the robot
                pose = image_poses[file_path] # [x, y, theta]
                object_lst_box[object_num-1].append(box) # bounding box of object
                object_lst_pose[object_num-1].append(np.array(pose).reshape(3,)) # robot pose
            except ZeroDivisionError:
                pass

    # if there are more than one objects of the same type, combine them
    for i in range(5):
        if len(object_lst_box[i])>0:
            box = np.stack(object_lst_box[i], axis=1)
            pose = np.stack(object_lst_pose[i], axis=1)
            # added entry to dictionary on object's bounding box as viewed by camera and corresponding robot pose
            completed_img_dict[i+1] = {'object': box, 'robot': pose}
        
    return completed_img_dict

# estimate the pose of a object based on size and location of its bounding box in the robot's camera view and the robot's pose
def estimate_pose(camera_matrix, completed_img_dict):
    camera_matrix = camera_matrix
    focal_length = camera_matrix[0][0]
    
    # actual (approximate) sizes of objects
    object_dimensions = []
    redapple_dimensions = [0.074, 0.074, 0.087]
    object_dimensions.append(redapple_dimensions)
    greenapple_dimensions = [0.081, 0.081, 0.067]
    object_dimensions.append(greenapple_dimensions)
    orange_dimensions = [0.075, 0.075, 0.072]
    object_dimensions.append(orange_dimensions)
    mango_dimensions = [0.113, 0.067, 0.058] # measurements when laying down
    object_dimensions.append(mango_dimensions)
    capsicum_dimensions = [0.073, 0.073, 0.088]
    object_dimensions.append(capsicum_dimensions)

    object_list = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']

    object_pose_dict = {}
    # for each object in each detection output, estimate its pose
    for object_num in completed_img_dict.keys():
        # obtain bounding box, robot pose and object actual dimensions before performing estimations 
        box = completed_img_dict[object_num]['object'] # [[x],[y],[width],[height]]
        robot_pose = completed_img_dict[object_num]['robot'] # [[x], [y], [theta]]
        true_height = object_dimensions[object_num-1][2]
        
        ######### Replace with your codes #########
        # TODO: compute pose of the object based on bounding box info and robot's pose
        # This is the default code which estimates every pose to be (0,0)
        object_pose = {'x': 0.0, 'y': 0.0}

        # # Calculate object position in the camera frame 
        # object_center_x = box[0] + box[2]/2     # centre width
        # object_center_y = box[1] + box[3]/2     # centre height 
        # object_depth = true_height * focal_length / box[3]   

        # # Convert object position in camera frame to world frame
        # object_world_x = object_center_x / focal_length * object_depth + robot_pose[0]
        # object_world_y = object_center_y / focal_length * object_depth + robot_pose[1]

        # # Calculate object orientation (assuming upright)
        # # object_orientation = robot_pose[2]  # Assuming object is upright relative to robot
        # object_pose_dict[object_num] = {'x': object_world_x, 'y': object_world_y}

        # append result to a dictionary containing final estimations of the object pose in the world frame 
        object_pose_dict[object_list[object_num-1]] = object_pose
        ###########################################
    
    return object_pose_dict

# merge the estimations of the objects so that there are at most 1 estimate for each object type
def merge_estimations(object_map):

    # initialise estimation lists for each object
    redapple_est, greenapple_est, orange_est, mango_est, capsicum_est = [], [], [], [], []
    object_est = {}
    num_per_object = 1 # max number of units per object type. We are only use 1 unit per fruit type

    # combine the estimations from multiple detector outputs
    # for each file in object_map (dictionary), each file would have possibly more than one key for the fruits
    # if the object type (key) is detected in the file, the code appends the estimated pose as a numpy arr to the corresponding list
    for f in object_map:
        for key in object_map[f]:
            if key.startswith('redapple'):
                redapple_est.append(np.array(list(object_map[f][key].values()), dtype=float))
            elif key.startswith('greenapple'):
                greenapple_est.append(np.array(list(object_map[f][key].values()), dtype=float))
            elif key.startswith('orange'):
                orange_est.append(np.array(list(object_map[f][key].values()), dtype=float))
            elif key.startswith('mango'):
                mango_est.append(np.array(list(object_map[f][key].values()), dtype=float))
            elif key.startswith('capsicum'):
                capsicum_est.append(np.array(list(object_map[f][key].values()), dtype=float))

    ######### Replace with your codes #########
    # TODO: the operation below is the default solution, which simply takes the first estimation for each object type.
    # Replace it with a better merge solution.
    if len(redapple_est) > num_per_object:
        redapple_est = redapple_est[0:num_per_object]
    if len(greenapple_est) > num_per_object:
        greenapple_est = greenapple_est[0:num_per_object]
    if len(orange_est) > num_per_object:
        orange_est = orange_est[0:num_per_object]
    if len(mango_est) > num_per_object:
        mango_est = mango_est[0:num_per_object]
    if len(capsicum_est) > num_per_object:
        capsicum_est = capsicum_est[0:num_per_object]
    ###########################################
    
    # object_est below will take the merged estimation, and to be writtenas an output file in lab_output/objects.txt
    # the estimation default to (0,0) if the object(s) are unable to be detected.
    # object_est is the final estimation dictionary
    for i in range(num_per_object):
        try:
            object_est['redapple_'+str(i)] = {'x':redapple_est[i][0], 'y':redapple_est[i][1]}
        except:
            object_est['redapple_'+str(i)] = {'x': 0.0, 'y': 0.0}
        try:
            object_est['greenapple_'+str(i)] = {'x':greenapple_est[i][0], 'y':greenapple_est[i][1]}
        except:
            object_est['greenapple_'+str(i)] = {'x': 0.0, 'y': 0.0}
        try:
            object_est['orange_'+str(i)] = {'x':orange_est[i][0], 'y':orange_est[i][1]}
        except:
            object_est['orange_'+str(i)] = {'x': 0.0, 'y': 0.0}
        try:
            object_est['mango_'+str(i)] = {'x':mango_est[i][0], 'y':mango_est[i][1]}
        except:
            object_est['mango_'+str(i)] = {'x': 0.0, 'y': 0.0}
        try:
            object_est['capsicum_'+str(i)] = {'x':capsicum_est[i][0], 'y':capsicum_est[i][1]}
        except:
            object_est['capsicum_'+str(i)] = {'x': 0.0, 'y': 0.0}
           
    return object_est



if __name__ == "__main__":
    # camera_matrix = np.ones((3,3))/2
    # this accesses the camera intrinsic matrix for estimation purposes
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')    
    
    # a dictionary of all the saved detector outputs
    # image_poses should contain all the robot poses recorded and its corresponding image file taken
    image_poses = {}
    with open('lab_output/pred.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['predfname']] = pose_dict['pose']
    print(image_poses)

    # estimate pose of objects in each detector output
    object_map = {}        

    # for each image file, get the image's bounding boxes and robot pose info, estimate where the fruit is in the world frame, 
    # then add these estimations into an object_map.
    # object_map should collect the estimation poses of all objects across all pictures taken
    for file_path in image_poses.keys():
        completed_img_dict = get_image_info(file_path, image_poses)
        print(completed_img_dict)
        object_map[file_path] = estimate_pose(camera_matrix, completed_img_dict)

    # merge the estimations of the objects so that there are only one estimate for each object type
    # object_est finalises the estimation list by obtaining only one estimation value per fruit object
    object_est = merge_estimations(object_map)
                     
    # save object pose estimations
    with open('lab_output/objects.txt', 'w') as fo:
        json.dump(object_est, fo, indent=4)
    
    print('Estimations saved!')