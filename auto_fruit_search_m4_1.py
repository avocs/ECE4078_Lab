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
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
from slam.aruco_sensor import ArucoSensor
from slam.aruco_sensor import Marker
from operate import Operate
import pygame
# import shutil

######
# Last Modified: 17th September 2024
# 11.00 PM
# Last Edited By: REEN
# CTRL K + 0 to collapse all

poseTurningTime = 150 # STEAL NUMBER FROM SANDRA
thresholdError = 0.1
straightTimeAdjustment = 1.20
offsetForCamera = 0.17 # in metres
maximumBestTurnAngle = 90
turnTimeAdditional = 0.13
offsetForCentreRobot = 0.23 
TURN_WAIT = 0.25

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


def normalise_theta():
    operateObj.ekf.robot.state[2, 0] = operateObj.ekf.robot.state[2, 0] % (2*np.pi)


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

    robot_pose = operateObj.ekf.robot.state

    # Find the distance needed to travel in the x and y direction
    dx = waypoint[0] - robot_pose[0]
    dy = waypoint[1] - robot_pose[1]
    dx = dx - offsetForCamera
    dy = dy - offsetForCamera
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

    
    final_angle_rad = clamp_angle(target_angle_rad)
    final_angle = final_angle_rad*180/(np.pi)
    # Determine the wheel speeds for turning
    turn_speeds = [-rotation_speed, rotation_speed] if final_angle_rad > 0 else [rotation_speed, -rotation_speed]
    # turn_speeds = [0.0, 0.0] if num_ticks != 0 else turn_speeds
    
    turn_time = float( (abs(final_angle_rad)*baseline) / (rotation_speed*scale) )

    

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
    print("Movement 1")
    # operateObj.command['wheel_speed'] = wheel_rot
    # drive_meas = operateObj.control(turn_time)
    # bunch_of_functions(operateObj, drive_meas, canvas)
    # time.sleep(turn_time)
    # operateObj.pibot_control.set_velocity([0, 0])

    # operateObj.command['wheel_speed'] = [0,0]
    # bunch_of_functions(operateObj)
    # time.sleep(0.5)

    angle_left_to_turn = final_angle_rad

    while abs(angle_left_to_turn) > maximumBestTurnAngle:
        
        print(f"Current turn_angle: {np.rad2deg(angle_left_to_turn)}, robot_pose: {robot_pose}")
        
        current_turn = maximumBestTurnAngle if angle_left_to_turn > 0 else -maximumBestTurnAngle

        turn_time = float((abs(current_turn) * baseline) / (rotation_speed * scale)) + turnTimeAdditional
        operateObj.command['wheel_speed'] = turn_speeds
        drive_meas = operateObj.control(turn_time)
        time.sleep(turn_time)
        operateObj.pibot_control.set_velocity([0, 0])
        # time.sleep(1)

        operateObj.take_pic()
        bunch_of_functions(operateObj, drive_meas, canvas)
        
        robot_pose = operateObj.ekf.robot.state
        
        print(f"Updated robot_pose: {robot_pose}")

        time.sleep(1.50)

        angle_left_to_turn -= current_turn  # Decrease turn_angle by the current turn amount
        # turn_angle = normalize_angle(turn_angle)  # Normalize again after the update
        print(f"Updated turn_angle: {np.rad2deg(angle_left_to_turn)}")

# Handle the final turn if turn_angle is less than max_turn_angle
    if abs(angle_left_to_turn) > 0:

        turn_speeds = [-rotation_speed, rotation_speed] if final_angle_rad > 0 else [rotation_speed, -rotation_speed]

        final_turn_time = float( (abs(angle_left_to_turn)*baseline) / (rotation_speed*scale) )+0.1
        operateObj.command['wheel_speed'] = turn_speeds
        drive_meas = operateObj.control(final_turn_time)
        time.sleep(final_turn_time)
        operateObj.pibot_control.set_velocity([0, 0])
        # time.sleep(1)

        operateObj.take_pic()
        bunch_of_functions(operateObj, drive_meas, canvas)

        robot_pose = operateObj.ekf.robot.state
        print(f"Updated robot_pose after final turn: {robot_pose}")
        print(f"Final turn_angle: {np.rad2deg(angle_left_to_turn)}")

    #########################
    # SECOND MOVEMENT TO MOVE ROBOT STRAIGHT
    time.sleep(1)

    print("Movement 2")

    # Calculate the DRIVE STRAIGHT TIME
    # TODO: Check this also
    drive_time = distance_to_go/(scale*straight_speed)
    drive_time*= straightTimeAdjustment


    operateObj.command['wheel_speed'] = [straight_speed, straight_speed]
    drive_meas = operateObj.control(drive_time)
    # operate.update_slam(drive_meas)
    # operate.draw(canvas)
    # pygame.display.update()
    time.sleep(drive_time)
    pibot_control.set_velocity([0,0])
    operateObj.take_pic()
    bunch_of_functions(operateObj)


    # time.sleep(0.5)

    # operateObj.command['wheel_speed'] = [0,0]
    # bunch_of_functions(operateObj)

    # #########################
    # # THIRD MOVEMENT TO REORIENT ROBOT TO Y-AXIS 
    # print("Movement 3")   
    # wheel_rot[0] = -wheel_rot[0]
    # wheel_rot[1] = -wheel_rot[1]
    # operateObj.command['wheel_speed'] = wheel_rot
    # bunch_of_functions(operateObj)
    # time.sleep(0.5)

    # operateObj.command['wheel_speed'] = [0,0]
    # bunch_of_functions(operateObj)
    # time.sleep(0.5)
    robot_pose = operateObj.ekf.robot.state
    

    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
    # time.sleep(3)
    

# clamps the angle in radians, to range of -pi to pi
def clamp_angle(turning_angle_raw):
    turning_angle = turning_angle_raw % (2*np.pi) # shortcut to while loop deducting 2pi from the angle
    # if angle more than 180 deg, make a negative angle 
    turning_angle = turning_angle - 2*np.pi if turning_angle > np.pi else turning_angle
    return turning_angle

###################################################################

def get_robot_pose():
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here
    # update the robot pose [x,y,theta]
    # robot_pose = [0.0,0.0,0.0] # replace with your calculation
# def get_robot_pose(operateObj):

    # TODO: Need to implement slam here. Right now I am just manually keying it in

    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')

    # turn_angle = np.deg2rad(22.5)  # 45 degrees to radians
    turn_speed = 0.4  # Speed for turning
    detection_timeout = 60  # Timeout in seconds for detecting markers
    markers_detected = []

    # aruco_sensor = ArucoSensor(robot)  # Initialize ArucoSensor with required parameters if needed
    
    start_time = time.time()  # Record start time
    current_angle = 0  # Initialize the current angle
    # image_id = 0 
    # folder = 'lab_output/test/'
    # if not os.path.exists(folder):
    #     os.makedirs(folder)
    # else:
    #     shutil.rmtree(folder)
    #     os.makedirs(folder)
        
    turn_angle = np.pi/12
    #how far the robot turns each time
    capture_time = TURN_WAIT #how logn the robot waits between turns. want to ake this as small as possible
    
    while True:  # 8 steps for 360 degrees
        
        # frame = pibot.get_image()
        # f_ = os.path.join(folder, f'{image_id}.png')

        # image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f_, image)
        
        normalise_theta()

        turn_time = (baseline * turn_angle) / (turn_speed * scale)+0.1  # Calculate turn time
        operateObj.command['wheel_speed'] = [-turn_speed, turn_speed]

        drive_meas = operateObj.control(turn_time)
        time.sleep(turn_time)

        operateObj.pibot_control.set_velocity([0, 0])
        
        operateObj.take_pic()
        bunch_of_functions(operateObj, drive_meas, canvas)
                
        robot_pose = operateObj.ekf.robot.state
        # robot_pose = [operateObj.ekf.robot.state[0, 0], operateObj.ekf.robot.state[1, 0],operate.ekf.robot.state[2, 0]/np.pi*180]
        print(robot_pose)

        operateObj.command['wheel_speed']=[0,0]
        drive_meas = operateObj.control(capture_time)
        time.sleep(capture_time)
        print(drive_meas.dt,drive_meas.left_speed,drive_meas.right_speed,drive_meas.left_cov,drive_meas.right_cov)
        operateObj.take_pic()
        
        measurements = operateObj.update_slam(drive_meas)
        operateObj.draw(canvas)
        pygame.display.update()
        print(measurements)
        
        if measurements:
            print(f"Detected markers at {current_angle} degrees:")            
            for m in measurements:
                # Assuming m has attributes `tag` and `position`
                print(f"  Marker ID: {m.tag}, Position: {m.position}")
            for m in measurements:
                if m.tag not in [marker.tag for marker in markers_detected]:
                    markers_detected.append(m)
                    
            if len(measurements) >=2:
                #appropriate number of measurements found can use markers to approximate location
                print("2 markers found")
                time.sleep(2)
                for i in range (poseTurningTime):
                    operateObj.take_pic()
                    measurements = operateObj.update_slam(drive_meas)
                    operateObj.draw(canvas)
                    pygame.display.update()
                    #check to see if robot pose is being updated
                    previous_pose = robot_pose
                    robot_pose = operateObj.ekf.robot.state
                    # robot_pose = [operateObj.ekf.robot.state[0, 0], operate.ekf.robot.state[1, 0],operate.ekf.robot.state[2, 0]/np.pi*180]
                    print(robot_pose)
                    if np.abs(np.mean(np.array(previous_pose)-np.array(robot_pose)))<=thresholdError:
                        break
                    time.sleep(0.01)
                
                
                # for _ in range(POSE_ESTIMATE_TIME):
                #     operate.take_pic()
                #     measurements = operate.update_slam(drive_meas)
                #     operate.draw(canvas)
                #     pygame.display.update()
                #     #check to see if robot pose is being updated
                #     robot_pose = [operate.ekf.robot.state[0, 0], operate.ekf.robot.state[1, 0],operate.ekf.robot.state[2, 0]/np.pi*180]
                #     print(robot_pose)
                #     time.sleep(0.01)
                #give it some time to update location accurately might need a longer wait time
                break

        # time.sleep(capture_time)

    if len(markers_detected) < 2:
        print("Timeout: Unable to detect enough markers.")
        
        robot_pose = operateObj.ekf.robot.state
        # robot_pose = [operate.ekf.robot.state[0, 0], operate.ekf.robot.state[1, 0],operate.ekf.robot.state[2, 0]]
        return robot_pose
    else:
        # Perform EKF update if enough markers are detected
        robot_pose = operateObj.ekf.robot.state
        # robot_pose = [operate.ekf.robot.state[0, 0], operate.ekf.robot.state[1, 0],operate.ekf.robot.state[2, 0]]
        return robot_pose


def bunch_of_functions(operateObj, drive_meas, canvas):
    # operateObj.take_pic()
    # drive_meas = operateObj.control()
    operateObj.update_slam(drive_meas)
    # operateObj.record_data()
    # operateObj.save_image()
    # operateObj.detect_object()
    operateObj.draw(canvas)
    pygame.display.update()


###################################################################

# main loop
if __name__ == "__main__":

    ############################
    '''
    ARGUMENTS

    '''
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=5000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--yoloV8", default='YOLOv8/best_10k.pt')
    args, _ = parser.parse_known_args()

    pibot_control = PibotControl(args.ip, args.port)

    operateObj = Operate(args)

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    print("Fruit List: ", fruits_list)
    print("Fruits True Pos: ", fruits_true_pos)
    print("Aruco True Pos: ", aruco_true_pos)
    print("Search List: ", search_list)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]

    # Intial value for drive measure
    drive_meas = operateObj.control()

    
    ###################################################################
    ###################################################################
    # TODO: PYGAME GUI
    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 40)
    
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
    ###################################################################
    ###################################################################

    
    
    ###################################################################
    ###################################################################
    # The following code is only a skeleton code the semi-auto fruit searching task
    # loopFlag = 0
    # while True:

    #     if loopFlag == 0:
    #         for i in range(5):
    #             operateObj.update_keyboard()
    #             operateObj.take_pic()
    #             drive_meas = operateObj.control()
    #             operateObj.update_slam(drive_meas)
    #             operateObj.record_data()
    #             operateObj.save_image()
    #             operateObj.detect_object()
    #             # visualise
    #             operateObj.draw(canvas)
    #             pygame.display.update()
    #             flag = 1
    

    while True:
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue
        

        
        # Estimate the robot's pose
        print(drive_meas)
        robot_pose = get_robot_pose()
        # robot_pose = [0,0,0]
        # Robot drives to the waypoint
        waypoint = [x, y]
        drive_to_point(waypoint, robot_pose)
        robot_pose = operateObj.ekf.robot.state
        # robot_pose = [operateObj.ekf.robot.state[0, 0], operateObj.ekf.robot.state[1, 0], operateObj.ekf.robot.state[2, 0]]
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint, robot_pose))

        # Exit
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput.lower() == 'n':
            break