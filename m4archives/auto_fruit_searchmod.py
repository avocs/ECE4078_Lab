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

# TODO
from operate import Operate
from pibot import Drive
import matplotlib.pyplot as plt
import math


def read_true_map(fname):
    """
    Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search
    @param fname: filename of the map
    @return:
        1) fruit_list: list of target fruits, e.g. ['redapple', 'greenapple', 'orange']
        2) fruit_true_pos: locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) aruco_true_pos: locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
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

# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
def drive_to_point(waypoint, robot_pose, operateThing):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    leftBool = False
    rightBool = False
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point
    # refer to wheel_calibration.py if you need the robot to move in a specified amount of time

    ## CALCULATIONS
    rotation_speed = 0.4
    straight_speed = 0.7
    total_time = 2.2
    dx = waypoint[0] - robot_pose[0]
    dy = waypoint[1] - robot_pose[1]
    distance_to_go = np.sqrt((dx**2) + (dy**2))
    target_angle = np.arctan2(dy, dx)
    target_angle = target_angle*180/(np.pi)

    ####################################
    print("---------------------------------------------------------")
    print(f"Original Pose:\n\tX = {robot_pose[0]}\n\tY = {robot_pose[1]}")
    print(f"New Pose:\n\tX = {waypoint[0]}\n\tY = {waypoint[1]}")
    print(f"Distance:\n\tX = {dx}\n\tY = {dy}")
    print(f"Displacement:\n\t{distance_to_go}")
    print(f"Target Angle: {target_angle}")
    #####################################

    if (0 < target_angle < 90) or (-90 < target_angle < 0): # Quadrant 1 and 4
        # Right Half Plane
        rightBool = True
        print("Turn Right!")

    elif (90 < target_angle < 180) or (-180 < target_angle < -90):
        # Left Half Plane
        leftBool = True
        print("Turn Left!")

    elif target_angle == 90 or target_angle == -270:
        # Go Straight
        final_angle = 0
        left_wheel_speed = straight_speed
        right_wheel_speed = straight_speed
        pass
        
    elif target_angle == -90 or target_angle == 270:
        # Go backwards
        final_angle = 180
        rightBool = True
        left_wheel_speed = straight_speed
        right_wheel_speed = straight_speed
        pass

    elif target_angle == 0 or target_angle == 360:
        # Turn Right
        rightBool = True
        final_angle = 90
    
    elif target_angle == 180 or target_angle == -180:
        # Turn Left
        leftBool = True
        final_angle = 90


    if rightBool:
        # Set the wheel speeds
        left_wheel_speed = rotation_speed
        right_wheel_speed = -rotation_speed
        # Final sorting of angles
        if target_angle < 0:
            final_angle = abs(target_angle) + 90
        else:
            final_angle = 90 - target_angle

    elif leftBool:
        # Set the wheel speeds
        left_wheel_speed = -rotation_speed
        right_wheel_speed = rotation_speed
    
        # Final sorting of angles
        if target_angle < 0:
            final_angle = target_angle + 180 + 90
        else:
            final_angle = target_angle - 90

    wheel_rot = [left_wheel_speed, right_wheel_speed]
    # Update Robot Pose
    # TODO had to change this to force index waypoints cuz i removed x,y as global variables
    robot_pose[0] = waypoint[0]
    robot_pose[1] = waypoint[1]

    print(f"Final Angle Degrees: {final_angle}")
    final_angle_rad = (final_angle/180)*(np.pi)
    print(f"Final Angle Radian: {final_angle_rad}")
    if rightBool or leftBool:
        # angular_vel = abs((right_wheel_speed - left_wheel_speed)/baseline)
        # print("Angular Velocity: ")
        # turn_time = abs(final_angle_rad*(baseline*np.pi/scale/30)/(2*np.pi)) + 0.025 #0.025

        # turn_time = abs(final_angle_rad*(baseline*np.pi/scale/30)/(2*np.pi)) + 0.025

        # angular_vel = abs((right_wheel_speed - left_wheel_speed)/baseline)
        # turn_time = (2*np.pi)*final_angle_rad/angular_vel
        turn_time = (final_angle_rad*total_time)/(2*np.pi)
    else:
        turn_time = 0
    print(f"Turning Time:\n\t{turn_time}")

    drive_time = distance_to_go/(scale*straight_speed)
    print(f"Driving Time:\n\t{drive_time}")

    start = time.time()
    elapsed = 0
    while elapsed < turn_time:
        pibot_control.set_velocity(wheel_rot)
        elapsed = time.time() - start
    pibot_control.set_velocity([0,0])

    start = time.time()
    elapsed = 0
    while elapsed < drive_time:
        pibot_control.set_velocity([straight_speed, straight_speed])
        elapsed = time.time() - start
    pibot_control.set_velocity([0,0])

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
    


def get_robot_pose(waypoint):
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here
    global robot_pose

    # update the robot pose [x,y,theta]
    # robot_pose = [0.0,0.0,0.0] # replace with your calculation
    # robot_pose = EKFOBJECT.get_state_vector()
    # robot_pose = ekfobject.get_state_vector()
    # print("This is the ekf object inside get robot pose", ekfobject)
    robot_pose[0] = waypoint[0]
    robot_pose[1] = waypoint[1]
    
    print(f"Robot Pose: {robot_pose}\n")
    ####################################################


    return robot_pose



###################################################################
# Helper function for waypoint selection 
def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))

# Function for handling user interaction and waypoint selection on the GUI
def enter_waypoint_on_click(event, fig, px, py, idx, waypoint_callback, robot_pose, thing):
    global waypoint
    space = np.array([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])

    x = round_nearest(event.xdata, 0.2)
    y = round_nearest(event.ydata, 0.2)

    if event.button == 1:
        # Left click: add point
        px.append(x)
        py.append(y)
        idx += 1

        waypoint = [x,y]
        waypoint_callback(waypoint, robot_pose, thing)
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

    plt.xlabel("X");
    plt.ylabel("Y")
    plt.xticks(space); plt.yticks(space)
    plt.grid()
    fig.canvas.draw()

# Function to set up the initial GUI plot
def gui_setup():
    space = np.array([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])

    fig = plt.figure()
    plt.plot(0, 0, 'rx')
    plt.xlabel("X");
    plt.ylabel("Y")
    plt.xticks(space); plt.yticks(space)
    plt.grid()
    return fig

# Function to launch the GUI and handle interaction
def generate_gui(fig, px, py, idx, waypoint_callback, robot_pose, thing):
    print("Specify waypoint on grid for robot to drive to")
    fig.canvas.mpl_connect('button_press_event', lambda event: enter_waypoint_on_click(event, fig, px, py, idx, waypoint_callback, robot_pose, thing))
    plt.show()

# Calls to drive the robot upon detecting a new waypoint input, continuously does so until you close the figure
def waypoint_callback(waypoint, robot_pose, thing):
    print("Waypoint selected:", waypoint)
    robot_pose = get_robot_pose(waypoint)
    drive_to_point(waypoint, robot_pose, thing)
    print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))


###########################################################################

# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=5000)
    #TODO i ADDED THIS
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--yoloV8", default='YOLOv8/best_10k.pt')
    parser.add_argument("--mode", default=1)
    args, _ = parser.parse_known_args()

    pibot_control = PibotControl(args.ip, args.port)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

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
    thing = Operate(args)
    ekfObj = thing.ekf


    # # The following code is only a skeleton code the semi-auto fruit searching task
    # while True:
    #     # enter the waypoints
    #     # instead of manually enter waypoints in command line, you can get coordinates by clicking on a map (GUI input)
    #     x,y = 0.0, 0.0
    #     x = input("X coordinate of the waypoint: ")
    #     try:
    #         x = float(x)
    #     except ValueError:
    #         print("Please enter a number.")
    #         continue
    #     y = input("Y coordinate of the waypoint: ")
    #     try:
    #         y = float(y)
    #     except ValueError:
    #         print("Please enter a number.")
    #         continue

    # Variables, p will contains clicked points, idx contains current point that is being selected
    px, py = [], []
    idx = 0
    robot_pose = get_robot_pose(waypoint)

    fig = gui_setup()
    generate_gui(fig, px, py, idx, waypoint_callback, robot_pose, thing)
    
    # estimate the robot's pose
    # robot_pose = get_robot_pose(waypoint)

        # TODO
        # print("here is the roboting posing in main", robot_pose)

        # robot drives to the waypoint
        # waypoint = [x,y]
    # drive_to_point(waypoint, robot_pose, thing)
    # print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # # TODO
        # print("\n\n\n\n")

        # # exit
        # uInput = input("Add a new waypoint? [Y/N]")
        # if uInput.lower() == 'n':
        #     break