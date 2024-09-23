# M4 - Autonomous fruit searching
import os
import sys
import cv2
import ast
import json
import time
import argparse
import numpy as np
from pibot import PibotControl

# import SLAM components (M2)
# sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
# from slam.aruco_detector import ArucoDetector
from slam.aruco_sensor import ArucoSensor
from slam.aruco_sensor import Marker

# TODO
from operate import Operate
from pibot import Drive
import matplotlib.pyplot as plt
import math
import pygame

######
# Previous Name: auto_fruit_search_sandra.py
# Last Modified: 12th September 2024
# 6.00 PM
# Last Edited By: REEN
# CTRL K + 0 to collapse all


def read_true_map(fname):
    """
    Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search
    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['redapple', 'greenapple', 'orange']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as f:
        try:
            gt_dict = json.load(f)                   
        except ValueError as e:
            with open(fname, 'r') as f:
                gt_dict = ast.literal_eval(f.readline())   
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """
    Read the search order of the target fruits
    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """
    Print out the target fruits' pos in the search order
    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """
    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(3):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit, fruit, np.round(fruit_true_pos[i][0], 1), np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


###################################################################
# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
def drive_to_point(waypoint, robot_pose, operateObj):
    
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    '''
    One simple strategy is to first turn on the spot facing the waypoint, then drive straight to the way point
    Refer to wheel_calibration.py if you need the robot to move in a specified amount of time
    '''
    # pass

    '''
    DEFINITION OF VARIABLES
    - leftBool (Bool):     This is a boolean variable to identify if the car is turning left.
    - rightBool (Bool):    This is a boolean variable to identify if the car is turning right.

    - rotatation_speed (float):     This is the variable to tell the rotation speef of the car.
    - straight_speed (float):       This is the variable to tell the driving straight speed of the car.

    - rot_time (float):     This is the time taken for the car to rotate 360 degrees. Comes from the time.txt file.
    - drive_time (float):   This is the time taken for the car to drive straight 1m. Comes from the time.txt file.

    - dx (float):   This is the delta x distance between the current x-position and the waypoint x position.
    - dy (float):   This is the delta y distance between the current y-position and the waypoint y position.
    - distance_to_go (float):   This is the displacement needed to be travelled. (Pythogoras Theorem. Pls imagine TRIANGLE).

    - ANGLES:       Bear with me here. Pythogoras theorem measures angle from x-axis.
                    But because I want it to be measured from robot's Y-axis, that's why there's a "TARGET_ANGLE" and
                    "FINAL_ANGLE" variables.
                    Also, for my sanity, I changed it to DEGREES but CALCULATION uses RAD.

    - target_angle_rad (float):     This is the direct angle measured from the x-axis of the pythogoras theorem. RADIAN
    - target_angle (float):         Purely for Reen's sanity in DEGREES.
    - final_angle (float):          As our robot will always realign to become straight, so, this angle is the final angle
                                    it needs to turn from the y-axis to the waypoint. IN DEGREES
    - final_angle_rad (float):      For time calculation.

    - wheel_rot (array of float):   Basically like operate.py, where I specify the left wheel speed and right wheel speed.

    '''

    ## DEFINE VARIABLES
    leftBool = False    # Determines if the car should turn left
    rightBool = False   # Determines if the car should turn right
    rotation_speed = 0.4 # As per what was in operate.py
    straight_speed = 0.7 # As per what was in operate.py
    rot_time = 1.9 # TODO: This is to be changed to be obtained from the time.txt file
    drive_time = 0
    final_angle = 0
    final_angle_rad = 0
    target_angle_rad = 0
    target_angle = 0
    
    ## CALCULATION
    # Find the distance needed to travel in the x and y direction
    dx = waypoint[0] - robot_pose[0]
    dy = waypoint[1] - robot_pose[1]
    distance_to_go = np.sqrt((dx**2) + (dy**2)) # The displacement needed to be travelled (Pythogoras Theorem)
    
    # Find the angle needed to turn to face the waypoint
    target_angle_rad = np.arctan2(dy, dx)
    target_angle = target_angle_rad*180/(np.pi) # Converting it to degrees for my sanity

    ####################################################
    # TODO: Printing Statements for my sanity
    print("---------------------------------------------------------")
    print(f"Original Pose:\n\tX = {robot_pose[0]}\n\tY = {robot_pose[1]}")
    print(f"New Pose:\n\tX = {waypoint[0]}\n\tY = {waypoint[1]}")
    print(f"Distance:\n\tX = {dx}\n\tY = {dy}")
    print(f"Displacement:\n\t{distance_to_go}")
    print(f"Target Angle: {target_angle}")
    ####################################################

    ####################################################
    # Essentially, this is a very long algorithm that sort the angle between:
    # 1. RHP
    # 2. LHP
    # 3. Pure Straight
    # 4. Pure Backwards
    # 5. Pure Right
    # 6. Pure Left
    
    # Quadrant 1 and 4 (RHP)  ->   Right Half Plane
    if (0 < target_angle < 90) or (-90 < target_angle < 0):
        rightBool = True
        print("Turn Right!")

    # Quadrant 2 and 3 (LHP)  ->   Left Half Plane
    elif (90 < target_angle < 180) or (-180 < target_angle < -90):
        leftBool = True
        print("Turn Left!")

    # Go Pure Straight
    elif target_angle == 90 or target_angle == -270:
        final_angle = 0
        left_wheel_speed = straight_speed
        right_wheel_speed = straight_speed
    
    # Go Pure Backwards
    elif target_angle == -90 or target_angle == 270:
        final_angle = 180

        # Turning right so it turns along the right side to the back
        rightBool = True

        left_wheel_speed = straight_speed
        right_wheel_speed = straight_speed
    
    # Turn Pure Right
    elif target_angle == 0 or target_angle == 360:
        if dx == 0 and dy == 0:
            final_angle = 0
            left_wheel_speed = straight_speed
            right_wheel_speed = straight_speed
        else:
            final_angle = 90
            rightBool = True
        
    
    # Turn Pure Left
    elif target_angle == 180 or target_angle == -180:
        final_angle = 90
        leftBool = True
    ####################################################
        

    ####################################################
    # Essentially, this part of the code, is so that, after I determine if it turns right or left, I find the final angle.
    # Which as defined above, the final angle has to be measured from the robot's y-axis.

    # If TURN RIGHT
    if rightBool:

        # Set the wheel speeds
        left_wheel_speed = rotation_speed
        right_wheel_speed = -rotation_speed

        # Final sorting of angles
        if target_angle < 0:
            final_angle = abs(target_angle) + 90
        
        else:
            final_angle = 90 - target_angle

    # IF TURN LEFT
    elif leftBool:

        # Set the wheel speeds
        left_wheel_speed = -rotation_speed
        right_wheel_speed = rotation_speed
    
        # Final sorting of angles
        if target_angle < 0:
            final_angle = target_angle + 180 + 90
        
        else:
            final_angle = target_angle - 90
    ####################################################

    ####################################################
    # Set the wheel rotation speeds
    wheel_rot = [left_wheel_speed, right_wheel_speed]
    
    # Update Robot Pose
    # TODO had to change this to force index waypoints cuz i removed x,y as global variables
    robot_pose[0] = waypoint[0]
    robot_pose[1] = waypoint[1]

    # Converting the final angle to RADIAN
    final_angle_rad = (final_angle/180)*(np.pi)
    
    # IF turn RIGHT or turn LEFT
    if rightBool or leftBool:

        # Calculate Time according to ratio
        # TODO: Check this please
        turn_time = (final_angle_rad*rot_time)/(2*np.pi)
    
    # Else, GO STRAIGHT
    else:
        # No spinny spin
        turn_time = 0
    

    # Calculate the DRIVE STRAIGHT TIME
    # TODO: Check this also
    drive_time = distance_to_go/(scale*straight_speed)
    

    ####################################################
    # TODO: Printing Statements for my sanity
    print(f"Final Angle Degrees: {final_angle}")
    print(f"Final Angle Radian: {final_angle_rad}")
    print(f"Turning Time:\n\t{turn_time}")
    print(f"Driving Time:\n\t{drive_time}")
    ####################################################


    ####################################################
    # Basically stole this from wheel_calibration =]

    #########################
    # INITIAL MOVEMENT TO TURN ROBOT TO FACE WAYPOINT
    start = time.time()
    elapsed = 0
    while elapsed < turn_time:
        operateObj.pibot_control.set_velocity(wheel_rot)
        elapsed = time.time() - start
    pibot_control.set_velocity([0,0])
    lv = wheel_rot[0]
    rv = wheel_rot[1]
    turn_drive_meas = Drive(lv, rv, turn_time)

    # operateObj.command['wheel_speed'] = wheel_rot
    # print(operateObj.command['wheel_speed'])
    print("Movement 1")
    # drive_meas = operateObj.control()
    operateObj.update_slam(turn_drive_meas)
    time.sleep(0.5)

    #########################
    # SECOND MOVEMENT TO MOVE ROBOT STRAIGHT
    start = time.time()
    elapsed = 0
    while elapsed < drive_time:
        operateObj.pibot_control.set_velocity([straight_speed, straight_speed])
        elapsed = time.time() - start
    pibot_control.set_velocity([0,0])

    lv = straight_speed
    rv = straight_speed
    straight_drive_meas = Drive(lv, rv, drive_time)

    # operateObj.command['wheel_speed'] = straight_speed
    print("Movement 2")
    # drive_meas = operateObj.control()
    operateObj.update_slam(straight_drive_meas)
    time.sleep(0.5)

    #########################
    # THIRD MOVEMENT TO REORIENT ROBOT TO Y-AXIS 0
    start = time.time()
    elapsed = 0
    wheel_rot[0] = -wheel_rot[0]
    wheel_rot[1] = -wheel_rot[1]
    while elapsed < turn_time:
        operateObj.pibot_control.set_velocity(wheel_rot)
        elapsed = time.time() - start
    operateObj.command['wheel_speed'] = wheel_rot
    lv = wheel_rot[0]
    rv = wheel_rot[1]
    turn_drive_meas = Drive(lv, rv, turn_time)


    # drive_meas = operateObj.control()
    operateObj.update_slam(turn_drive_meas)
    print("Movement 3")
    pibot_control.set_velocity([0,0])
    

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
    # time.sleep(3)
    
    
###################################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here
    # update the robot pose [x,y,theta]
    # robot_pose = [0.0,0.0,0.0] # replace with your calculation
def get_robot_pose(operateObj):

    # TODO: Need to implement slam here. Right now I am just manually keying it in
    # robot_pose[0] = waypoint[0]
    # robot_pose[1] = waypoint[1]
    robot_pose = operateObj.ekf.robot.state.squeeze().tolist()
    robot_pose[0] = round(robot_pose[0], 3)
    robot_pose[1] = round(robot_pose[1], 3)
    robot_pose[2] = round(robot_pose[2], 3)

    print(f"Robot Pose: {robot_pose}\n")
    return robot_pose


###################################################################
###################################################################
# TODO: OWN CODE ADDED

###################################################################
# Helper function for waypoint selection 
def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))

# Function for handling user interaction and waypoint selection on the GUI
def enter_waypoint_on_click(event, fig, px, py, idx, waypoint_callback, operateObj):
    global waypoint
    space = np.array([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])

    x = round_nearest(event.xdata, 0.2)
    y = round_nearest(event.ydata, 0.2)
    bunch_of_functions(waypoint, robot_pose, operateObj)

    if event.button == 1:
        # Left click: add point
        px.append(x)
        py.append(y)
        idx += 1

        waypoint = [x,y]
        waypoint_callback(waypoint, robot_pose, operateObj)
    elif event.button == 3:
        # Right click: delete last point
        del px[-1]
        del py[-1]
        idx -= 1
        waypoint = None

    # Clear and redraw the plot with updated waypoints
    plt.clf()
    plt.plot(0, 0, 'rx')

    for i in range(len(px)):
        plt.scatter(px[i], py[i], color='C0')
        plt.text(px[i] + 0.05, py[i] + 0.05, i + 1, color='C0', size=12)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xticks(space); plt.yticks(space)
    plt.grid()
    fig.canvas.draw()

# Function to set up the initial GUI plot
def gui_setup():
    space = np.array([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])
    fig = plt.figure()
    plt.plot(0, 0, 'rx')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xticks(space); plt.yticks(space)
    plt.grid()
    return fig

# Function to launch the GUI and handle interaction
def generate_gui(fig, px, py, idx, waypoint_callback, operateObj):
    print("Specify waypoint on grid for robot to drive to")
    fig.canvas.mpl_connect('button_press_event', lambda event: enter_waypoint_on_click(event, fig, px, py, idx, waypoint_callback, operateObj))
    bunch_of_functions(waypoint, robot_pose, operateObj)
    plt.show()

# Calls to drive the robot upon detecting a new waypoint input, continuously does so until you close the figure
def waypoint_callback(waypoint, robot_pose, operateObj):
    operateObj.command['run_obj_detector'] = True
    operateObj.command['save_obj_detector'] = True
    operateObj.command['save_slam'] = True
    # take latest picture and update slam
    operateObj.take_pic()
    # pass
    # operateObj.take_pic()
    # # TODO: Fuck ard and find out
    # # operateObj.command['wheel_speed'] = [0,0]
    # pibot_control.set_velocity([0,0])
    # drive_meas = operateObj.control()
    # operateObj.update_slam(drive_meas)


    # print("Waypoint selected:", waypoint)
    
    drive_to_point(waypoint, robot_pose, operateObj)
    robot_pose = get_robot_pose(operateObj)
    print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
    operateObj.record_data()
    operateObj.save_image()
    operateObj.detect_object()
    operateObj.draw(canvas)
    pygame.display.update()
    # robot_pose = get_robot_pose(operateObj)
    # print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
    # time.sleep(3)   

def bunch_of_functions(waypoint, robot_pose, operateObj):
    operateObj.command['run_obj_detector'] = True
    operateObj.command['save_obj_detector'] = True
    operateObj.command['save_slam'] = True
    operateObj.take_pic()
    drive_to_point(waypoint, robot_pose, operateObj)
    robot_pose = get_robot_pose(operateObj)
    print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
    operateObj.record_data()
    operateObj.save_image()
    operateObj.detect_object()
    operateObj.draw(canvas)
    pygame.display.update()


###################################################################

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    # parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--map", type=str, default='m3set1.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=5000)
    
    # TODO: Added Arguments and Params because of the Operate Object
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--yoloV8", default='YOLOv8/best_10k.pt')
    args, _ = parser.parse_known_args()

    pibot_control = PibotControl(args.ip, args.port)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

    ###################################################################
    #TODO 头痛
    
    # robot = EKF.__init__(args.calib_dir, args.ip)
    # calib_dir = "calibration/param/"
    # fileK = os.path.join(calib_dir, 'intrinsic.txt')
    # camera_matrix = np.loadtxt(fileK, delimiter=',')
    # fileD = os.path.join(calib_dir, 'distCoeffs.txt')
    # dist_coeffs = np.loadtxt(fileD, delimiter=',')
    # fileS = os.path.join(calib_dir, 'scale.txt')
    # scale = np.loadtxt(fileS, delimiter=',')
    # fileB = os.path.join(calib_dir, 'baseline.txt')
    # baseline = np.loadtxt(fileB, delimiter=',')
    # robot = Robot(baseline, scale, camera_matrix, dist_coeffs)

    # EKFOBJECT = EKF(robot)
    operateObj = Operate(args)
    ekfObj = operateObj.ekf
    # operateObj.ekf.load_true_map(args.map)

    n_observed_markers = len(operateObj.ekf.taglist)
    if n_observed_markers == 0:
        if not operateObj.ekf_on:
            operateObj.notification = 'SLAM is running'
            operateObj.ekf_on = True
        else:
            operateObj.notification = '> 2 landmarks is required for pausing'
    elif n_observed_markers < 3:
        operateObj.notification = '> 2 landmarks is required for pausing'
    else:
        if not operateObj.ekf_on:
            operateObj.request_recover_robot = True
        operateObj.ekf_on = not operateObj.ekf_on
        if operateObj.ekf_on:
            operateObj.notification = 'SLAM is running'
        else:
            operateObj.notification = 'SLAM is paused'
        
        lms = []
    
    print(operateObj.notification)

    lms = []
    print(aruco_true_pos)
    print(fruits_true_pos)
    # all_true_pos = np.concatenate(aruco_true_pos, fruits_true_pos)
    # print(all_true_pos)
    for i,lm in enumerate(aruco_true_pos):
        measure_lm = Marker(np.array([[lm[0]],[lm[1]]]), i+1)
        lms.append(measure_lm)
    for i, lm in enumerate(fruits_true_pos):
        measure_lm = Marker(np.array([[lm[0]],[lm[1]]]), i+10)
        # print("i: ", i+10)
        # print("Measure: ", measure_lm.position)
        lms.append(measure_lm)
    operateObj.ekf.add_landmarks(lms) 
    operateObj.ekf.load_true_map(args.map)
    # print("LMS: ", lms)
    operateObj.command['save_slam'] = True
    operateObj.record_data()
    print(operateObj.notification)
    operateObj.ekf_on = True
    
    ###################################################################

    ###################################################################
    # Variables, p will contains clicked points, idx contains current point that is being selected
    px, py = [], []
    idx = 0

    # # TODO: forseeable issue
    # # Estimate the robot's pose
    # robot_pose = get_robot_pose(waypoint, operateObj)

    fig = gui_setup()
    # generate_gui(fig, px, py, idx, waypoint_callback, operateObj)


    pygame.font.init() 
    # TITLE_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 35)
    # TEXT_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 40)
    
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 Lab')
    pygame.display.set_icon(pygame.image.load('ui/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('ui/loading.png')
    pibot_animate = [pygame.image.load('ui/8bit/pibot1.png'),
                     pygame.image.load('ui/8bit/pibot2.png'),
                     pygame.image.load('ui/8bit/pibot3.png'),
                    pygame.image.load('ui/8bit/pibot4.png'),
                     pygame.image.load('ui/8bit/pibot5.png')]
    pygame.display.update()

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    # The following code is only a skeleton code the semi-auto fruit searching task
    while True:
        operateObj.command['run_obj_detector'] = True
        operateObj.command['save_obj_detector'] = True
        operateObj.command['save_slam'] = True
        # take latest picture and update slam
        operateObj.take_pic()
        
        # lv, rv = operateObj.pibot_control.set_velocity([0, 0])
        # drive_meas = Drive(lv, rv, 0.0)
        # operateObj.update_slam(drive_meas)
        
        
        # enter the waypoints
        # instead of manually enter waypoints in command line, you can get coordinates by clicking on a map (GUI input), see camera_calibration.py
        # x,y = 0.0,0.0
        # x = input("X coordinate of the waypoint: ")
        # try:
        #     x = float(x)
        # except ValueError:
        #     print("Please enter a number.")
        #     continue
        # y = input("Y coordinate of the waypoint: ")
        # try:
        #     y = float(y)
        # except ValueError:
        #     print("Please enter a number.")
        #     continue

        # estimate the robot's pose
        robot_pose = get_robot_pose(operateObj)

        # robot drives to the waypoint
        # waypoint = [x,y]
        generate_gui(fig, px, py, idx, waypoint_callback, operateObj)
        pygame.quit()
        sys.exit()



        # drive_to_point(waypoint,robot_pose,operateObj)
        robot_pose = get_robot_pose(operateObj)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))
        
        # # custom
        # lv, rv = operateObj.pibot_control.set_velocity([0, 0])
        # drive_meas = Drive(lv, rv, 0.0)
        # operateObj.update_slam(drive_meas)

        # # exit
        # operateObj.pibot_control.set_velocity([0, 0])
        operateObj.record_data()
        operateObj.save_image()
        operateObj.detect_object()
        operateObj.draw(canvas)
        pygame.display.update()
        # time.sleep(5)
        # uInput = input("Add a new waypoint? [Y/N]")
        # if uInput == 'N':
        #     break

    