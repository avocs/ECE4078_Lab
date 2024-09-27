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
import copy
import a_star_path_planning as astar

############################################# oh my fucking god im gonna make my own operate rn #######################################################
# First Fucked Around by: DRA
# File created: 24/09/2024 4.00pm 
# Last Fuck Around Date & Time: 24/09/2024 4pm

# NOTE: 
'''
0. As sandra continues fucking around and finding out, she found it wiser to just bring in the entire operate class
cuz of some functions that are quite useful and im sick of copy pasting 
hahaha im going insane. this file is gonna stay around for referencing though! 
1. added code to generate the waypoints from the path planning file
2. trying to find a sequence to search for fruits here
3. 

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
        # changed from default false to true 
        self.ekf_on = True 
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
    global robot_pose 
    full_rotation_time = 2.0       # TODO whatever this is 
    wheel_rot_speed = 0.4 

    # could have a calibration baseline to determine turning base speed

    # clamp angle between -180 to 180 deg 
    print(f'preturn {turning_angle}')
    turning_angle = turning_angle % (2*np.pi) # shortcut to while loop deducting 2pi 
    # if the angle is more than 180 deg, make this a negative angle instead 
    turning_angle = turning_angle - 2*np.pi if turning_angle > np.pi else turning_angle
    turning_angle_deg = turning_angle * 180 / np.pi

    # move the robot to perform rotation 
    # -- time to turn according to ratio 
    turning_time = abs(turning_angle) * full_rotation_time / (2 * np.pi) 

    # -- direction of wheels, depending on sign
    if turning_time != 0: # if the car is not going straight/has to turn
        if (turning_angle) > 0: # turn left 
            lv, rv = [-wheel_rot_speed, wheel_rot_speed]
        elif turning_angle < 0: # turn right
            lv, rv = [wheel_rot_speed, -wheel_rot_speed] 
    else: 
        lv, rv = [0.0, 0.0]

    print(f"turning for {turning_time}s to {turning_angle_deg}")
    wheel_rotation_speeds = [lv, rv]

    # nyoom 
    start = time.time()
    elapsed = 0
    while elapsed < turning_time:
        pibot_control.set_velocity(wheel_rotation_speeds)
        elapsed = time.time() - start
    pibot_control.set_velocity([0,0])

    turn_drive_meas = Drive(lv, rv, turning_time)
    # get_robot_pose(turn_drive_meas)
    robot_pose[2] += turning_angle

    return None 


def robot_move_straight(dist_to_waypt=0, wheel_lin_speed=0.7, wheel_rot_speed=0.4):
    '''
    this function makes the robot drive straight a certain time automatically 
    '''
    global robot_pose, scale

    # time to drive straight for 
    drive_time = dist_to_waypt / (scale * wheel_lin_speed)
    lv, rv = wheel_lin_speed, wheel_lin_speed
    drive_speeds = [lv, rv] 

    print(f"driving for {drive_time}s")

    # TODO dra mod 
    # pibot_control.set_target(100, 100)
    # pibot_control.set_velocity(drive_speeds)


    # nyoom 
    start = time.time()
    elapsed = 0
    while elapsed < drive_time:
        pibot_control.set_velocity(drive_speeds)
        elapsed = time.time() - start
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

    # 2. Robot drives straight towards waypt
    # ===============================================
    dist_to_waypt = math.hypot(x_dist_to_waypt, y_dist_to_waypt)
    robot_move_straight(dist_to_waypt)
    print(f"Robot Pose: {robot_pose}\n")
    # 3. Robot reorients itself to world y-axis
    # =============================================
    robot_move_rotate(-turning_angle)
    # 4. arrived at waypoint 
    # =================================================
    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))



def get_robot_pose(drive_meas):
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here

    # Plan A: hardcode it 
    # # obtain angle with respect to x-axis
    # # robot_pose[2] = np.arctan2(waypoint[1]-robot_pose[1],waypoint[0]-robot_pose[0])
    # # robot_pose[2] = (robot_pose[2] + 2*np.pi) if (robot_pose[2] < 0) else robot_pose[2] # limit from 0 to 360 degree

    # robot_pose[0] = waypoint[0]
    # robot_pose[1] = waypoint[1]
    # # robot_pose[0] = round(robot_pose[0], 3)
    # # robot_pose[1] = round(robot_pose[1], 3)
    # # robot_pose[2] = round(robot_pose[2], 3)

    # # update the robot pose [x,y,theta]
    # print(f"Robot Pose: {robot_pose}\n")
    ####################################################

    # Plan B: SLAM through EKF 
    # '''
    global robot_pose
    global landmarks # NEW Added
    landmarks, detector_output = capture_and_read_image()
    ekf.predict(drive_meas)
    # ekf.add_landmarks(landmarks) #TODO not sure if this is needed or not
    ekf.update(landmarks) 

    robot_pose = ekf.robot.state.reshape(-1)
    print(f"Pred. Robot Pose: [{robot_pose[0]},{robot_pose[1]},{robot_pose[2]*180/np.pi}]")
    return robot_pose, landmarks


def search_for_fruit(fruit_waypoints):
    for waypoint in fruit_waypoints:
        drive_to_point(waypoint)
    print(f'Fruit reached, robot eepy')
    time.sleep(3)

# rotate robot to scan for landmarks
def rotate_robot(num_turns=4):
    global baseline, scale
    
    wheel_vel = 20
    turn_offset = 0
    # angle per turn
    turn_resolution = 2*np.pi/num_turns
    turn_time = (abs(turn_resolution)*baseline)/(2.0*scale*wheel_vel) + turn_offset

    for _ in range(num_turns):
        lv, rv = pibot_control.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
        turn_drive_meas = Drive(lv, rv, turn_time-turn_offset)
        
        time.sleep(0.5)
        # self.take_pic()
        # self.update_slam(turn_drive_meas)

        # update pygame display
        # self.draw(canvas)
        # pygame.display.update()
        
        # self.waypoint_update()

        print(f"Position rotate: {ekf.robot.state.squeeze().tolist()}")

def capture_and_read_image():
    # take photo of fruit
    # i give up i should just bring in the entirety of operate.py here
    raw_img = pibot_control.get_image()
    global aruco_img
    measurements, aruco_img = aruco_sensor.detect_marker_positions(raw_img)
    cv2.imshow('aruco labelled', aruco_img)
    cv2.waitKey(0)



def display_true_map(aruco_true_pos, fruits_copy):
    # plot result of path planning and show in a figure 
    x_aruco = []
    y_aruco = []
    x_fruits = []
    y_fruits = []

    for i in range(len(aruco_true_pos)):
        x_aruco.append(aruco_true_pos[i][0])
        y_aruco.append(aruco_true_pos[i][1])
    
    plt.plot(x_aruco, y_aruco, 'ok') 
    for i, (x, y) in enumerate(zip(x_aruco, y_aruco), 1):
        plt.annotate(f'aruco_{i}', (x, y), textcoords="offset points", xytext=(0,-10), ha='center')
    
    # fruit_color = [[128, 0, 0], [155, 255, 70], [255, 85, 0], [255, 180, 0], [0, 128, 0]]
    fruit_colour = ["red", "cyan", "orange", "yellow", "green"]

    for i in range(len(fruits_copy)):
        x_fruits.append(fruits_copy[i][0])
        y_fruits.append(fruits_copy[i][1])
    plt.scatter(x_fruits, y_fruits, c=fruit_colour, s=100)
    for i, (x, y) in enumerate(zip(x_fruits, y_fruits), 1):
        plt.annotate(f'{i+10}', (x, y), textcoords="offset points", xytext=(0,-10), ha='center')
    
    plt.grid()
    plt.show()       



########################################################################################################################
# main loop
if __name__ == "__main__":

    # arguments for robot
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map.txt')
    parser.add_argument("--ip", metavar='', type=str, default='192.168.0.104')
    parser.add_argument("--port", metavar='', type=int, default=5000)
    args, _ = parser.parse_known_args()

    # obtain robot settings and parameters (MAKE THEM GLOBAL FOR SANITY)
    global scale, baseline, camera_matrix, dist_coeffs
    fileK = "calibration/param/intrinsic.txt"
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    fileD = "calibration/param/distCoeffs.txt"
    dist_coeffs = np.loadtxt(fileD, delimiter=',')
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')

    # robot start up using arguments (yoinked from operate)
    pibot_control = PibotControl(args.ip,args.port)
    robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
    ekf = EKF(robot)
    aruco_sensor = ArucoSensor(robot, marker_length=0.06) # size of the ARUCO markers (6cm)
    
    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)

    # setup markers to ekf (yoinked from ekf), # TODO mod later
    landmarks = []
    print(aruco_true_pos)
    for i,landmark in enumerate(aruco_true_pos):
        measurement_landmark = Marker(np.array([[landmark[0]],[landmark[1]]]),i+1)
        landmarks.append(measurement_landmark)
    # print(landmarks)
    ekf.add_landmarks(landmarks)

    search_list = read_search_list()
    print("======= GT ==============")
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    print("=========================")

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0, np.pi/2]


######################################################################################################################

    # The following code is only a skeleton code the semi-auto fruit searching task
    # --------------------- LEVEL 2 CODE ---------------------------------------------------

    # obtain a searching index and the corresponding true pos of the fruits for algorithm
    search_index = []
    for i in range(len(search_list)):          ## The shopping list only, so 3
        for j in range(len(fruits_list)):      ## The full list at 5
            if search_list[i] == fruits_list[j]:
                search_index.append(j)

    search_true_pos = []        # list of the true poses of the fruits in the search list
    for i in range(len(search_index)):
        search_true_pos.append(fruits_true_pos[search_index[i]])
    fruits_copy = copy.deepcopy(fruits_true_pos.tolist())

    src = [0,0]  
    waypoints = []  

    for i in range(len(search_list)):
        # define one fruit as the destination, from the start of the search list
        grid_src = astar.convert_coord_to_grid(src)
        dest = search_true_pos[i]
        
        for j in range (len(dest)):
                if dest[j] < 0:
                    dest[j] += 0.1
            
                else:
                    dest[j] -= 0.1
                dest[j] = round(dest[j], 2)
            # print(dest)
        grid_dest = astar.convert_coord_to_grid(dest)
        grid = astar.modify_obstacles(aruco_true_pos.tolist(), search_index[i], fruits_true_pos.tolist())
        waypoints.append(astar.a_star_search(grid, grid_src, grid_dest))
        src = dest
    
    for i in range(len(waypoints)):
        print(f'argh {waypoints[i]}')


    display_true_map(aruco_true_pos, fruits_copy)


