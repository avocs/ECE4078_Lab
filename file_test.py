import os, ast
import json

with open('lab_output/pred.txt', 'r') as fp:
    # Check file size before reading
    initial_size = os.path.getsize('lab_output/pred.txt')
    print("Initial file size:", initial_size)

    image_poses = {}

    for line in fp.readlines():
        # Process the line as needed
        pose_dict = ast.literal_eval(line) # pose_dict = {pose, predfname}
        image_poses[pose_dict['imgfname']] = pose_dict['pose'] # image_poses = {pred_n.png: robotpose}

    print(image_poses)
    # Check file size after reading
    final_size = os.path.getsize('lab_output/pred.txt')
    print("Final file size:", final_size)

    if initial_size != final_size:
        print("File size has changed unexpectedly.")



