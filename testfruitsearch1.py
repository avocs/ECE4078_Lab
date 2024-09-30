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
# from slam.aruco_detector import ArucoDetector
import matplotlib.pyplot as plt
import math
from slam.ekf import EKF 
from slam.robot import Robot 
from slam.aruco_sensor import ArucoSensor
from slam.aruco_sensor import Marker
import pygame 
from pibot import Drive

############################################# oh my fucking god im gonna make my own operate rn #######################################################
# First Fucked Around by: DRA
# File created: 22/09/2024 3:25am 
# Last Fuck Around Date & Time: 23/09/2024  1:00am 

# NOTE: 
''' 
1. calling this file no longer requires '--ip' argument since ive changed out the default argument to our robot's ip
2. ive basically yoinked operate and made a new operate class here --- cant be arsed about good OOP design anymore
    however im using this merely for ui purposes and i can no longer keep track of how operate calls to classes like EKF and Drive etc. so those are separated
3. removed the part where the robot reorients itself back to normal, seeing if this works tmr. 
4. current issue: this needs crazy time calibration.
5. 
'''


class Operate:
    def _init_(self): 

        # Initialise robot controller object
        self.pibot_control = PibotControl(args.ip, args.port)
        self.command = {'wheel_speed':[0, 0], # left wheel speed, right wheel speed
                        'save_slam': False,
                        'run_obj_detector': False,                       
                        'save_obj_detector': False,
                        'save_image': False,
                        'load_true_map': False}

        # Other auxiliary objects/variables      
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.obj_detector_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        self.count_down = 300 # 5 min timer
        self.start_time = time.time()
        self.control_clock = time.time()
        self.img = np.zeros([480,640,3], dtype=np.uint8)
        self.aruco_img = np.zeros([480,640,3], dtype=np.uint8)
        self.obj_detector_pred = np.zeros([480,640], dtype=np.uint8)        
        self.bg = pygame.image.load('ui/gui_mask.jpg')


    # paint the GUI            
    def draw(self, canvas):

        width, height = 900, 660
        canvas = pygame.display.set_mode((width, height))
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(520, 480+v_pad), not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, position=(h_pad, v_pad))

        # for object detector (M3)
        detector_view = cv2.resize(self.prediction_img, (320, 240), cv2.INTER_NEAREST)
        self.draw_pygame_window(canvas, detector_view, position=(h_pad, 240+2*v_pad))

        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad))
        self.put_caption(canvas, caption='Detector', position=(h_pad, 240+2*v_pad))
        self.put_caption(canvas, caption='Pi-Bot Cam', position=(h_pad, v_pad))

        notification = TEXT_FONT.render(self.notification, False, text_colour)
        canvas.blit(notification, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))


def boot_ui():
    '''
    This is the function you need to call to boot the pygame window.
    '''

    global TITLE_FONT, TEXT_FONT, start

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
    
    return None 

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
                    aruco_true_pos[marker_id-1][0] = x
                    aruco_true_pos[marker_id-1][1] = y
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
def robot_move_rotate(turning_angle=0, wheel_lin_speed=0.7, wheel_rot_speed=0.4):
    '''
    this function makes the robot turn a fixed angle automatically

    '''
    global robot_pose, baseline
    full_rotation_time = 3.0       # TODO whatever this is 
    # wheel_rot_speed = 0.4 

    ticks_per_revolution = 20
    wheel_diameter = 66e-3            # yoinked from cytron

    if abs(turning_angle) > 0:
        tick_offset = 8
    else: 
        tick_offset = 0

    # clamp angle between -180 to 180 deg 
    print(f'preturn {turning_angle}')
    turning_angle = turning_angle % (2*np.pi) # shortcut to while loop deducting 2pi 
    # if the angle is more than 180 deg, make this a negative angle instead 
    turning_angle = turning_angle - 2*np.pi if turning_angle > np.pi else turning_angle
    turning_angle_deg = turning_angle * 180 / np.pi

    # move the robot to perform rotation 
    # -- time to turn according to ratio 
    turning_time = abs(turning_angle) * full_rotation_time / (2 * np.pi) 

    # wheel circumference
    wheel_circum = np.pi * wheel_diameter
    #  If the robot pivoted 360°, the distance traveled by each wheel 
    # would be equal to the circumference of this pivot circle
    pivot_circum = np.pi * baseline 
    distance_per_wheel = abs(turning_angle / (2*np.pi)) * pivot_circum

    # distance each wheel must travel
    # distance_per_wheel = (baseline/2) * turning_angle
    turning_revolutions = distance_per_wheel / wheel_circum
    num_ticks = round(turning_revolutions * ticks_per_revolution) + tick_offset
    # manual override
    print(f'turning for {num_ticks} ticks to {turning_angle_deg}')
    print(f"turning for {turning_time}s to {turning_angle_deg}")




    # -- direction of wheels, depending on sign
    if turning_time != 0: # if the car is not going straight/has to turn
        if (turning_angle) > 0: # turn left 
            lv, rv = [-wheel_rot_speed, wheel_rot_speed]
        elif turning_angle < 0:
            lv, rv = [wheel_rot_speed, -wheel_rot_speed] 
    else: 
        lv, rv = [0.0, 0.0]
    
    turn_speeds = [lv, rv]

    # alt nyoom 
    initial_ticks = pibot_control.get_counter_values()
    ticks_travelled_left, ticks_travelled_right = 0,0
    pibot_control.set_velocity(turn_speeds)

    while True: 
        curr_ticks = pibot_control.get_counter_values()
        ticks_travelled_left = curr_ticks[0] - initial_ticks[0]
        ticks_travelled_right = curr_ticks[1] - initial_ticks[1]
        print(f"curr_ticks {curr_ticks}")

        if ticks_travelled_left >= num_ticks and ticks_travelled_right >= num_ticks:
            break
    pibot_control.set_velocity([0,0])
    

    # wheel_rotation_speeds = [lv, rv]

    # # nyoom 
    # start = time.time()
    # elapsed = 0
    # while elapsed < turning_time:
    #     pibot_control.set_velocity(wheel_rotation_speeds)
    #     elapsed = time.time() - start
    # pibot_control.set_velocity([0,0])


    turn_drive_meas = Drive(lv, rv, turning_time)
    # get_robot_pose(turn_drive_meas)
    robot_pose[2] += turning_angle

    return None 


def robot_move_straight(dist_to_waypt=0, wheel_lin_speed=0.5, wheel_rot_speed=0.4):
    '''
    this function makes the robot drive straight a certain time automatically 
    '''
    global robot_pose, scale

    ticks_per_revolution = 20
    wheel_diameter = 66e-3            # yoinked from cytron

    if dist_to_waypt > 0:
        tick_offset = 16
    else: 
        tick_offset = 0

    # wheel circumference
    wheel_circum = np.pi * wheel_diameter
    drive_revolutions = dist_to_waypt / wheel_circum
    num_ticks = drive_revolutions * ticks_per_revolution + tick_offset

    # time to drive straight for 
    drive_time = dist_to_waypt / (scale * wheel_lin_speed)
    lv, rv = wheel_lin_speed, wheel_lin_speed
    drive_speeds = [lv, rv] 

    # number of ticks to drive striaght for
    print(f'driving for {num_ticks} ticks')
    print(f"driving for {drive_time}s")


    # nyoom 
    # start = time.time()
    # elapsed = 0
    # while elapsed < drive_time:
    #     pibot_control.set_velocity(drive_speeds)
    #     elapsed = time.time() - start
    # pibot_control.set_velocity([0,0])

    # alt nyoom 
    initial_ticks = pibot_control.get_counter_values()
    ticks_travelled_left, ticks_travelled_right = 0,0
    pibot_control.set_velocity(drive_speeds)

    while True: 
        curr_ticks = pibot_control.get_counter_values()
        print(f"curr_ticks {curr_ticks}")

        ticks_travelled_left = curr_ticks[0] - initial_ticks[0]
        ticks_travelled_right = curr_ticks[1] - initial_ticks[1]
        
        if ticks_travelled_left >= num_ticks and ticks_travelled_right >= num_ticks:
            break
    
    pibot_control.set_velocity([0,0])

    straight_drive_meas = Drive(lv, rv, drive_time)
    # get_robot_pose(straight_drive_meas)
    robot_pose[0], robot_pose[1] = waypoint[0], waypoint[1]

    return None 


def drive_to_point(waypoint):
    global robot_pose

    # 1. Robot rotates, turning towards the waypoint
    # ===================================================
    y_dist_to_waypt = waypoint[1] - robot_pose[1]
    x_dist_to_waypt = waypoint[0] - robot_pose[0]
    angle_to_waypt = np.arctan2(y_dist_to_waypt, x_dist_to_waypt) # angle measured in rad, from theta = 0
    # this is the angle that the robot needs to turn, in radians. sign determines direction of turning
    turning_angle = angle_to_waypt - robot_pose[2] 

    print(f'curr orientation {robot_pose[2]}, angle_towaypt {angle_to_waypt}, turning_angle{turning_angle}')
    robot_move_rotate(turning_angle)
    print(f"Robot Pose: {robot_pose}\n")
    time.sleep(0.2)

    # 2. Robot drives straight towards waypt
    # ===============================================
    dist_to_waypt = math.hypot(x_dist_to_waypt, y_dist_to_waypt)
    robot_move_straight(dist_to_waypt)
    print(f"Robot Pose: {robot_pose}\n")
    time.sleep(0.2)

    # 3. Robot reorients itself to world y-axis
    # =============================================
    robot_move_rotate(-turning_angle)
    time.sleep(0.2)
    # 4. arrived at waypoint 
    # =================================================
    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))




def get_robot_pose(drive_meas):
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # this is uhhhh inaccurate, currently hardcoded to just take robot pose right away
    # obtain angle with respect to x-axis
    # robot_pose[2] = np.arctan2(waypoint[1]-robot_pose[1],waypoint[0]-robot_pose[0])
    # robot_pose[2] = (robot_pose[2] + 2*np.pi) if (robot_pose[2] < 0) else robot_pose[2] # limit from 0 to 360 degree

    robot_pose[0] = waypoint[0]
    robot_pose[1] = waypoint[1]
    # robot_pose[0] = round(robot_pose[0], 3)
    # robot_pose[1] = round(robot_pose[1], 3)
    # robot_pose[2] = round(robot_pose[2], 3)

    # update the robot pose [x,y,theta]
    print(f"Robot Pose: {robot_pose}\n")
    ####################################################

    return robot_pose


########################################################################################################################
# main loop
if __name__ == "__main__":

    # arguments for robot
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.0.104')
    parser.add_argument("--port", metavar='', type=int, default=5000)
    args, _ = parser.parse_known_args()

    # robot start up using arguments
    pibot_control = PibotControl(args.ip,args.port)

    # obtain robot settings and parameters (MAKE THEM GLOBAL FOR SANITY)
    global scale, baseline
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')    

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print("======= GT ==============")
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    print("=========================")

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0, np.pi/2]


######################################################################################################################

    px, py = [], []
    idx = 0

    # The following code is only a skeleton code the semi-auto fruit searching task
    # -------------------- LEVEL 1 CODE ------------------------------------------
    while True:
        # enter the waypoints
        # instead of manually enter waypoints in command line, you can get coordinates by clicking on a map (GUI input)
        x,y = 0.0, 0.0
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

        # estimate the robot's pose
        # robot_pose = get_robot_pose()

        # robot drives to the waypoint
        waypoint = [x,y]
        drive_to_point(waypoint)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput.lower() == 'n':
            break