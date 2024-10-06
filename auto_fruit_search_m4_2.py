# this unit can go fuck itself 
# Latest version of the L2 code
# Last update: 05/10/2024
# Runs the motion model by taking small increments of time up until the encoder ticks tell it to stop


# basic python packages
import cv2 
import time
import os, sys
import numpy as np
import pygame # python package for GUI
from pibot import Drive
from pibot import PibotControl # access the robot

# import SLAM components (M2)
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
from slam.aruco_sensor import ArucoSensor

# import CV components (M3)
from YOLOv8.detector import ObjectDetector
# M4 - Autonomous fruit searching
import ast
import json
import argparse
import matplotlib.pyplot as plt
from slam.aruco_sensor import Marker
import copy
import a_star_path_planning as astar
import object_pose_estyolo as obj_est


class Operate:
    def __init__(self, args):
        # Initialise robot controller object
        self.pibot_control = PibotControl(args.ip, args.port)
        self.command = {'wheel_speed':[0, 0], # left wheel speed, right wheel speed
                        'save_slam': False,
                        'run_obj_detector': False,                       
                        'save_obj_detector': False,
                        'save_image': False,
                        'load_true_map': False,
                        'auto_fruit_search': False}
                        
        # TODO: Tune PID parameters here. If you don't want to use PID, set use_pid = 0
        # self.pibot_control.set_pid(use_pid=1, kp=0.1, ki=0, kd=0.0005)  
        # self.pibot_control.set_pid(use_pid=1, kp=0.005, ki=0, kd=0.0005)  
        self.pibot_control.set_pid(use_pid=0, kp=0.0001, ki=0, kd=0.001)  

        # Initialise SLAM parameters (M2)
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_sensor = ArucoSensor(self.ekf.robot, marker_length=0.06) # size of the ARUCO markers (6cm)
        
        if args.yoloV8 == "":
            self.obj_detector = None
            self.prediction_img = cv2.imread('ui/8bit/detector_splash.png')
        else:
            self.obj_detector = ObjectDetector(args.yoloV8)
            self.prediction_img = np.ones((480,640,3))* 100

        # Other auxiliary objects/variables      
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.obj_detector_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        self.count_down = 600 # 10 min timer 
        self.start_time = time.time()
        # NOTE changed out time.time to time.perfcounter for more accurate 
        # self.control_clock = time.time()
        self.control_clock = time.perf_counter()
        self.img = np.zeros([480,640,3], dtype=np.uint8)
        self.aruco_img = np.zeros([480,640,3], dtype=np.uint8)
        self.obj_detector_pred = np.zeros([480,640], dtype=np.uint8)        
        self.bg = pygame.image.load('ui/gui_mask.jpg')

        # Aux objects/vars added for m4
        self.waypoints_list = []
        self.curr_waypoint_count = 0
        self.localising_flag = False
        self.prev_pose = np.zeros((3,1))
        self.guessed_pose = np.zeros((3,1))
        self.pos_read_flag = False
        self.camera_offset = 0.17   # NOTE: measured from front of car to front of ECE4078 label on car
        self.wheel_diameter = 68e-3 # yoinked from cytron 
        self.ticks_per_revolution = 20

    # wheel control -- default for keyboard operation 
    def control(self):       
        left_speed, right_speed = self.pibot_control.set_velocity(self.command['wheel_speed'])
        dt = time.time() - self.control_clock
        drive_meas = Drive(left_speed, right_speed, dt)
        self.control_clock = time.time()
        return drive_meas
    
    # NOTE this one unused, but this drives by a fixed time
    def control_time(self, lv, rv, drive_time):       
        # print("Turning for {:.2f} seconds at {} {}".format(turning_time, lv, rv))
        left_speed, right_speed = self.pibot_control.set_velocity([lv, rv])         # start moving robot
        time.sleep(drive_time)                                                    # wait until time passed
        self.pibot_control.set_velocity([0,0])                                      # stop robot
        drive_meas = Drive(left_speed, right_speed, drive_time)                   # obtain drive_meas to update location
        return drive_meas
    

    # NOTE on the spot, it returns the number of aruco markers seen
    def control_zero_ticks(self):
        self.take_pic()
        drive_meas = Drive(0.0, 0.0, 0,0)
        len_measurements = self.slam_gui_update(drive_meas, canvas)
    
        return len_measurements
    

    # NOTE Drive by number of ticks continuously
    def control_tick(self, lv, rv, num_ticks):
        # This file now contains time.sleeps throughout, so the timer would have to be reset at the start
        # so that the small increments of dt would be more accurate
        # self.control_clock = time.time()
        self.control_clock = time.perf_counter()
        initial_ticks = self.pibot_control.get_counter_values()
        ticks_travelled_left, ticks_travelled_right = 0,0

        # this is basically the while true loop of the operate's main 
        while True: 
            # call to operate take pic
            self.take_pic()

            # call to control
            left_speed, right_speed = self.pibot_control.set_velocity([lv, rv])
            # dt = time.time() - self.control_clock
            dt = time.perf_counter() - self.control_clock
            curr_ticks = self.pibot_control.get_counter_values()
            ticks_travelled_left = curr_ticks[0] - initial_ticks[0]
            ticks_travelled_right = curr_ticks[1] - initial_ticks[1]
            # print(f"Curr ticks: {curr_ticks}")
            drive_meas = Drive(left_speed, right_speed, dt)
            # self.control_clock = time.time()
            self.control_clock = time.perf_counter()

            # call to update_slam
            self.slam_gui_update(drive_meas, canvas)
            # print(f"pose update: {self.get_robot_pose()}")

            if ticks_travelled_left >= num_ticks and ticks_travelled_right >= num_ticks: 
                break

        self.pibot_control.set_velocity([0,0])
        time.sleep(0.25)

        return drive_meas
    

    # rotate by 1 tick -- does not call to update slam
    def rotate_step(self):

        initial_ticks = self.pibot_control.get_counter_values()
        ticks_travelled_left, ticks_travelled_right = 0,0
        wheel_rot_speed = 0.5
        lv, rv = -wheel_rot_speed, wheel_rot_speed

        self.pibot_control.set_velocity([lv, rv])

        while True:
            curr_ticks = self.pibot_control.get_counter_values()
            ticks_travelled_left = curr_ticks[0] - initial_ticks[0]
            ticks_travelled_right = curr_ticks[1] - initial_ticks[1]
            if ticks_travelled_left >= 1 and ticks_travelled_right >= 1:
                break
        self.pibot_control.set_velocity([0,0])

        # sleep between steps for 0.25s 
        time.sleep(0.25)
    
        return None
    
        
        
    # NOTE one stop call to update slam and the pygame window
    def slam_gui_update(self, drive_meas, canvas):
        measurements = self.update_slam(drive_meas)
        self.draw(canvas)
        pygame.display.update()
        return measurements

    # camera control
    def take_pic(self):
        self.img = self.pibot_control.get_image()

    # wheel and camera calibration for SLAM
    def init_ekf(self, calib_dir, ip):
        fileK = os.path.join(calib_dir, 'intrinsic.txt')
        self.camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = os.path.join(calib_dir, 'distCoeffs.txt')
        self.dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = os.path.join(calib_dir, 'scale.txt')
        self.scale = np.loadtxt(fileS, delimiter=',')
        fileB = os.path.join(calib_dir, 'baseline.txt')
        self.baseline = np.loadtxt(fileB, delimiter=',')
        # robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        robot = Robot(self.baseline, self.scale, self.camera_matrix, self.dist_coeffs)
        return EKF(robot)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        measurements, self.aruco_img = self.aruco_sensor.detect_marker_positions(self.img)
        
        if self.ekf_on:
            
            self.prev_pose = self.get_robot_pose()
            self.ekf.predict(drive_meas)                # perform prediction based on motion model
            self.guessed_pose = self.get_robot_pose()   # the guessed pose is blindly fixed on the motion model calculated pose
            # print(f"self.guessed_pose {self.guessed_pose}")
            # print(f"self.prev_pose {self.prev_pose}")
            # M3, disable updates to aruco markers if true map is loaded
            if self.command['load_true_map']:
                self.notification = 'SLAM locates robot pose'
                self.command['load_true_map'] = False

            # NOTE changed logic here
            # else: 
            #     if self.localising_flag  and len(measurements) >= 3:
            #         print('Updating SLAM')
            #         self.ekf.add_landmarks(measurements)
            #         self.ekf.update(measurements)
            #         self.notification = 'Updating SLAM'
            #     else:
            #         self.notification = 'Insufficient markers for updating SLAM'

            # Recovered original update_slam
            else:
                self.ekf.add_landmarks(measurements)
                self.ekf.update(measurements)

        return measurements

            # else: 
            #     # corrects the robot location if it sees any arucos
            #     if (len(measurements) >= 2):
            #         self.ekf.add_landmarks(measurements)
            #         self.ekf.update(measurements)
            #         print("Reading SLAM position)")

                # # if there are more than 3 aruco markers, 
                # if (len(measurements) >= 2):        # if there are more than 3 aruco markers in sight
                #     self.pos_read_flag = True       # the position to be read is the ekf corrected pose
                #     print("Reading SLAM position)")

                # else: 
                #     self.pos_read_flag = False      # otherwise, position takes the motion model blind pose
                #     # if the robot is currently rotating on the spot
                #     if self.localising_flag:
                #         print("---- Localising")
                #         xp,yp,thetap = self.prev_pose   # the previous x and y is accessed and forced into state
                #         self.ekf.robot.state[0,0], self.ekf.robot.state[1,0] = xp, yp 
                #     else:
                #         print("-- Motion model Estimation")
                #         xg,yg,thetag = self.guessed_pose
                #         print(f"new x, y {xg} {yg}")

                #         # force the state to be the driving one 
                #         self.ekf.robot.state[0,0], self.ekf.robot.state[1,0], self.ekf.robot.state[2,0] = xg,yg,thetag 

        # print(f"Final pred pose {self.get_robot_pose()}")
        # return len(measurements)

        
    # save SLAM map // unused for lvl2
    def record_data(self):
        # this saves slam map to slam.txt upon pressing 's' 
        if self.command['save_slam']:
            self.ekf.save_map(fname=os.path.join(self.lab_output_dir, 'slam.txt'))
            self.notification = 'Map is saved'
            self.command['save_slam'] = False
        
        if self.command['save_obj_detector']:
            if self.obj_detector_output is not None:            
                    # obj_detector_output = (bounding_boxes, robot_state)
                print("Robot State: ", self.obj_detector_output[1])

                # write information to respective txt files, and obtain the image name
                self.pred_fname = self.obj_detector.write_image(self.obj_detector_output[0], self.obj_detector_output[1], self.pred_output_dir)
                image = self.pibot_control.get_image()
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # TODO sandra 
                # save images to respective directories (note fin_prediction_image is just the colour converted image for visualising)
                fbbox = os.path.join(self.pred_output_dir, f'pred_{self.image_id}.png')
                f_ = os.path.join(self.raw_img_dir, f'pred_{self.image_id}.png')
                cv2.imwrite(f_, image)
                cv2.imwrite(fbbox, self.fin_prediction_img)
                self.image_id += 1
                self.notification = f'Prediction is saved to {operate.pred_fname}'
            else:
                self.notification = f'No prediction in buffer, save ignored'
            self.command['save_obj_detector'] = False

    # using computer vision to detect objects
    def detect_object(self):
        # upon pressing 'p', this calls run inference on the current image, obtains bounding boxes and 
        # an output image with the bbox drawn, saved into self.obj_detector_pred and self.fin_prediction_img respectively
        if self.command['run_obj_detector'] and self.obj_detector is not None:
            # obj_detector_pred = bounding_boxes, prediction_img = output_img
            self.obj_detector_pred, self.prediction_img = self.obj_detector.detect_single_image(self.img)
            str_preds = [str(pred) for pred in self.obj_detector_pred]
            self.fin_prediction_img = cv2.cvtColor(self.prediction_img, cv2.COLOR_RGB2BGR)
            self.command['run_obj_detector'] = False
            self.obj_detector_output = (self.obj_detector_pred, self.ekf.robot.state.tolist())
            self.notification = f'{len(np.unique(str_preds))} object type(s) detected'

    # save raw images taken by the camera after pressing "i"
    def save_image(self):
        if self.command['save_image']:
            image = self.pibot_control.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # TODO since the model runs inference directly on this set of pics and obtains bboxes, 
            # in the original operate, this raw_img saves the camera picture, then pred_n.png is a greyscale image
            # we have it saved together by pressing 'n'
            f_ = os.path.join(self.save_output_dir, f'pred_{self.image_id}.png')
            cv2.imwrite(f_, image)
            
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # paint the GUI            
    def draw(self, canvas):
        TEXT_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 40)

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
        # TODO - REEN
        TITLE_FONT = pygame.font.Font('ui/8-BitMadness.ttf', 35)
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    # Keyboard teleoperation 
    def update_keyboard(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                # pass # TODO
                self.command['wheel_speed'] = [0.7, 0.7]
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['wheel_speed'] = [-0.7, -0.7]
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['wheel_speed'] = [-0.6, 0.6]
            # turn right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['wheel_speed'] = [0.6, -0.6]
            # stop
            elif event.type == pygame.KEYUP or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
                self.command['wheel_speed'] = [0, 0]
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused' 
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['save_slam'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()          
            # run object/fruit detector
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['run_obj_detector'] = True
            # save object detection outputs
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                self.command['save_obj_detector'] = True
            # capture and save raw image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True

            # load true map into ekf
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                if args.map:
                    self.ekf.load_true_map(args.map)
                    self.notification = 'Loading true map...'
                    self.command['load_true_map'] = True
                else: 
                    self.notification = "No true map found."
            
        if self.quit:
            pygame.quit()
            sys.exit()
            
############################## A* FUNCTIONS FOR OPERATE CLASS ########################
    def display_map(self, aruco_true_pos, fruits_true_pos):
        '''
        Function provides a preview of the map generated by the path planning algorithm
        but i think this only plots the trueposes
        '''
        # plot result of path planning and show in a figure 
        x_aruco, y_aruco, x_fruits, y_fruits = [],[],[],[]

        for i in range(len(aruco_true_pos)):
            x_aruco.append(aruco_true_pos[i][0])
            y_aruco.append(aruco_true_pos[i][1])
        
        plt.plot(x_aruco, y_aruco, 'ok') 
        for i, (x, y) in enumerate(zip(x_aruco, y_aruco), 1):
            plt.annotate(f'aruco_{i}', (x, y), textcoords="offset points", xytext=(0,-10), ha='center')
        
        # fruit_color = [[128, 0, 0], [155, 255, 70], [255, 85, 0], [255, 180, 0], [0, 128, 0]]
        fruit_colour = ["red", "cyan", "orange", "yellow", "green"]

        for i in range(len(fruits_true_pos)):
            x_fruits.append(fruits_true_pos[i][0])
            y_fruits.append(fruits_true_pos[i][1])
        plt.scatter(x_fruits, y_fruits, c=fruit_colour, s=100)
        for i, (x, y) in enumerate(zip(x_fruits, y_fruits), 1):
            plt.annotate(f'{i+10}', (x, y), textcoords="offset points", xytext=(0,-10), ha='center')
        
        plt.grid(True)
        plt.show() 

    def generate_path_astar(self, search_list, fruits_list, fruits_true_pos):
        self.waypoints_list = astar.main()

################################ MAIN ALGORITHM ########################################

    def auto_fruit_search(self):
        '''
        Perform fruit search to all fruits in search list
        '''
        if self.command['auto_fruit_search']:
            print(f"Starting auto_fruit_search..")
            if any(self.waypoints_list):
                # fruit number for printing purposes
                curr_fruit = len(search_list) - len(self.waypoints_list) + 1

                if self.waypoints_list[0]:
                    self.curr_waypoint_count += 1

                    # robot tries to drive to the waypoint
                    
                    print("\n\n----------------------------------------------------------")
                    waypoint_to_go = self.waypoints_list[0][0]
                    print(f"Fruit {curr_fruit}, Waypoint {self.curr_waypoint_count}: {waypoint_to_go}")
                    self.drive_to_point(waypoint_to_go)
                    robot_pose = self.get_robot_pose()

                    print("Finished driving to waypoint: {}; New robot pose: {}".format(self.waypoints_list[0][0],robot_pose))
                    print()
                    
                    # localise self at every sub-waypoint except for first and last at that point
                    # while not self.is_close_to_waypoint(waypoint_to_go, self.get_robot_pose()):
                    #     if self.curr_waypoint_count >= 1:
                    #         print(f"\n\n///// Returning to waypoint {self.curr_waypoint_count}")
                    #         self.localising_flag = True
                    #         self.localise_rotate_robot()
                    #         self.localising_flag = False
                    #         self.drive_to_point(waypoint_to_go)

                    # localise self at every sub-waypoint except for first
                    if self.curr_waypoint_count >= 1:
                        self.est_robot_pose()

                    # remove that waypoint off the waypoint list
                    self.waypoints_list[0].pop(0)
                    print(f"New waypoints list: {self.waypoints_list}")

                    # if the waypoint is the last one in its list, means fruit is found
                    if not self.waypoints_list[0]:
                        self.waypoints_list.pop(0)
                        self.curr_waypoint_count = 0
                        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                        print("Fruit reached, robot sleeps for 3 seconds\n\n")
                        time.sleep(3)

                    self.record_data()

            else:
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print("Waypoint list is empty.\n\n")
                self.curr_waypoint_count = 0
                self.waypoints_list = []
                self.command['auto_fruit_search'] = False
    
########################## ROBOT DRIVE FUNCTIONS FOR OPERATE CLASS #####################
    # drive to a waypoint from current position
    def drive_to_point(self, waypoint):
        '''
        Function for robot to drive to a waypoint through the following procedures:
        1. Turn to waypoint
        2. Head straight to waypoint
        '''
        # compute x and y distance to waypoint
        x,y,theta = self.get_robot_pose()
        print(f"Starting position: {self.get_robot_pose()}")
        print(f"Robot is driving!")

        # 1. Robot rotates, turning towards the waypoint
        # ===================================================

        y_dist_to_waypt = waypoint[1] - (y - self.camera_offset*np.sin(theta))
        x_dist_to_waypt = waypoint[0] - (x - self.camera_offset*np.cos(theta))
        angle_to_waypt = np.arctan2(y_dist_to_waypt, x_dist_to_waypt) # angle measured in rad, from theta = 0
        # this is the angle that the robot needs to turn, in radians. sign determines direction of turning
        turning_angle = angle_to_waypt - theta        
        self.rotate_drive(turning_angle)

        time.sleep(0.3)
        print(f"Post Turn Position: {self.get_robot_pose()}\n")
                
        # 2. Robot drives straight towards waypt
        # ===============================================
        dist_to_waypt = np.hypot(x_dist_to_waypt, y_dist_to_waypt)
        print(f' --- dist to waypoint: {x_dist_to_waypt}, {y_dist_to_waypt}')
        self.straight_drive(dist_to_waypt)
        
        time.sleep(0.3)
        print(f"Post Drive Position: {self.get_robot_pose()}")
        
        print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))

        return None
    

    def rotate_drive(self, turning_angle=0, turn_ticks=0,wheel_lin_speed=0.5, wheel_rot_speed=0.4, rotate_speed_offset=0.05):
        '''
        This function makes the robot turn a fixed angle by counting encoder ticks
        '''

        # clamp angle between -180 to 180 deg 
        turning_angle = clamp_angle(turning_angle)
        turning_angle_deg = np.rad2deg(turning_angle)

        abs_turning_angle_deg = abs(turning_angle_deg)

        # wheel circumference
        wheel_circum = np.pi * self.wheel_diameter
        # If the robot pivoted 360°, the distance traveled by each wheel = circumference of this pivot circle
        pivot_circum = np.pi * self.baseline 
        distance_per_wheel = abs(turning_angle / (2*np.pi)) * pivot_circum

        # distance_per_wheel = (baseline/2) * turning_angle
        turning_revolutions = distance_per_wheel / wheel_circum

        # if turn_ticks has been given by default, none of the previous calculations matter anymore
        if turn_ticks != 0:
            num_ticks = turn_ticks
        else:
            num_ticks = np.round(abs(turning_revolutions * self.ticks_per_revolution))
        
        print(f' /// Turning for {num_ticks:.2f} ticks to {turning_angle_deg:.2f}')

        # assign speed direction according to sign of angle
        turn_speeds = [-wheel_rot_speed, wheel_rot_speed] if turning_angle > 0 else [wheel_rot_speed, -wheel_rot_speed]
        turn_speeds = [0.0, 0.0] if num_ticks == 0 else turn_speeds

        # drive by ticks, updating slam throughout
        if num_ticks != 0:
            self.control_tick(turn_speeds[0], turn_speeds[1], num_ticks)

        return None

    def straight_drive(self, dist_to_waypt=0, wheel_lin_speed=0.5, wheel_rot_speed=0.4):
        '''
        this function makes the robot drive straight a certain time automatically 
        '''

        # wheel circumference
        wheel_circum = np.pi * self.wheel_diameter
        drive_revolutions = dist_to_waypt / wheel_circum
        num_ticks = np.round(drive_revolutions * self.ticks_per_revolution)

        # time to drive straight for 
        drive_time = dist_to_waypt / (self.scale * wheel_lin_speed)
        drive_speeds = [wheel_lin_speed, wheel_lin_speed] 

        # number of ticks to drive striaght for
        print(f'/// Driving for {num_ticks:.2f} ticks to {dist_to_waypt:.2f}')

        # Drive to point, updating slam throughout
        if num_ticks != 0:
            self.control_tick(drive_speeds[0], drive_speeds[1], num_ticks)

        return None
    
    """
    Getter method to call to get robot state from ekf
    """
    def get_robot_pose(self):
        return self.ekf.robot.state.squeeze().tolist()
    

    ''' NEWNEWNEW NIGHTMARE NIGHTMARE 
    Estimate robot pose using SLAM and ArUco markers, trying to rewrite localise robot here
    '''
    def est_robot_pose(self):

        print("Robot panning and estimating own location....")

        robot_pose = self.get_robot_pose()
        turning_angle = np.pi/24            # 1 tick increments   
        total_rotations = 0   
        num_turns = int(2*np.pi / turning_angle)
        markers_found = []
        
        for i in range(num_turns):
        
        # while True:
            # turn for one tick
            self.rotate_step()

            # take pic and return measurements on the spot
            measurements = self.control_zero_ticks()

            # print pose for funsies
            print(f"Turn Pose: {self.get_robot_pose()}")

            # if there are any aruco's read 
            if measurements:
                for m in measurements:
                    if m.tag not in [marker.tag for marker in markers_found]:
                        markers_found.append(m)
                    
                if len(measurements) >= 2:
                    # use this to approximate location
                    print(f"{len(measurements)} markers found. Reorienting for 2 seconds")
                    time.sleep(1.5)

                    # call to confirm robot's own pose after reorientation
                    self.confirm_pose()
                    print(f"Determined pose: {self.get_robot_pose()}")

                    # pose is now determined, exit loop and continue on with next waypoint
                    break
            
        return None
    


    

############################ WAYPOINT UPDATE AND SLAM HELPER FUNCTIONS FOR OPERATE CLASS ####################
  
    """
    Function to confirm the robot's current location on the map
    """
    def confirm_pose(self):
        # get initial robot pose
        robot_pose = self.get_robot_pose()

        # keep updating slam but with zero movement
        pose_confirmation_count = 0
        target_confirmation_count = 5
        while True:
            self.control_zero_ticks()
            # curr_best_pose = self.get_robot_pose()

            # this waits until the pose is stabilised within 0.005
            # ---- scuffed ass np call
            if np.all(np.abs(np.array(self.get_robot_pose()) - np.array(robot_pose)) < 0.005):
                pose_confirmation_count += 1
                if pose_confirmation_count >= target_confirmation_count:
                    print("::::: Pose Confirmed! :::::")
                    break
            robot_pose = self.get_robot_pose()
            time.sleep(0.05)
            # the pose held by self.ekf.robot.state is now the confirmed pose 
        return robot_pose
    

    """
    Function to have robot pan and find markers to localise itself
    """
    def localise_rotate_robot(self, num_turns=0, wheel_rot_speed=0.5):

        print("Robot panning and localising....")

        num_rotations = 0
        turning_angle = np.pi/24            # 1 tick increments      
        num_turns = int(2*np.pi / turning_angle)
        
    # Method 1: some scuffed marker finding bullshit that is not implemented yet
##############################################################

        # while True:
        #     flag = False
        #     target_marker_count = 3

        #     while (self.control_static() < target_marker_count):
        #         self.rotate_step()
        #         num_rotations += 1 
                
        #         if (num_rotations >= 40):
        #             num_rotations = 0
        #             target_marker_count -= 1 
        #             flag = True
        #             print(f"3 Markers not found. Finding 2 Markers")
                
        #     if flag == True:
        #         self.confirm_pose()
        #         break
            
    # Method 2: forced rotation for 2pi 
#################################################################

        for i in range(num_turns):
            print(f'Rotation: {i+1}, Total turned: {turning_angle*i}')
            
            # fixed to turn 1 tick at a time, this is done without updating slam
            self.rotate_step()
            # self.rotate_drive(turn_ticks=1, turning_angle=turning_angle, wheel_rot_speed=wheel_rot_speed)
            
            # confirm its position for a while
            self.confirm_pose()

            # printing pose 
            print(f"Position after rotating: {self.get_robot_pose()}")
            time.sleep(0.25)
        return None
    

    # NOTE: This checks if the robot is sort of at the waypoint. otherwise the robot will keep trying to drive
    def is_close_to_waypoint(self, waypoint, current_pose):
        x, y, _ = current_pose
        x_goal, y_goal = waypoint
        threshold = 0.1

        return abs(x - x_goal) <= threshold and abs(y - y_goal) <= threshold
    
    # NOTE: implement cv for checks, currently unused
    def take_fruit_pic_and_estimate_fruit_pose(self, target_fruit, target_fruit_true_pos=None):

        fruit_aligned = False
        while not fruit_aligned:
            # take a picture of the current line of sight
            self.take_pic()
            # run object detector to identify fruits in sight, if there is one, 
            # there will be something in the assigned to obj detector output
            self.detect_object()

            # if there is any fruit identified
            if self.obj_detector_output is not None: 
                # --- obj_detector_output: [bboxes, robot_pose], see detect_object() function
                # initialise a dictionary for all the bboxes found in the image
                image_outputs = {}
                # for each bounding box captured in the picture
                for bbox in self.obj_detector_output[0]:
                    # obtain label and the corresponding bbox coordinates
                    label = bbox[0]
                    bounding_box = bbox[1].tolist()
                    # if the fruit is unique
                    if label not in image_outputs:
                        image_outputs[label] = []
                    image_outputs[label].append([bounding_box])

            if target_fruit in image_outputs.keys():
                print("Fruit in line of sight!")
                found_target_fruit_bbox = image_outputs[target_fruit]
                # this is x,y pose. imma need access to the theta orientation to help in 
                object_pose_entry, _ = obj_est.estimate_pose(self.camera_matrix, target_fruit, found_target_fruit_bbox)
                print(f"Fruit {target_fruit} found at {object_pose_entry}")

                # perform angle alignment
                print(f'Camera aligning itself to see fruit')
                # --- obtain angle of fruit 

                # --- calculate turning angle 

                # --- update location 

            else:
                print("Fruit not in line of sight, abit far away it seems, should try to account for this?")


        measurements, aruco_img = self.aruco_sensor.detect_marker_positions(self.img)

        # visualise
        operate.draw(canvas)
        pygame.display.update()
        
        return measurements, aruco_img
        # return landmarks, detector_output,aruco_corners
        
########################### KEYBOARD OPERATION DONE HERE ##############################
    
    # Keyboard control for Milestone 4 Level 2
    def update_keyboard_M4(self):
        for event in pygame.event.get():
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['wheel_speed'] = [0.7, 0.7]
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['wheel_speed'] = [-0.7, -0.7]
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['wheel_speed'] = [-0.6, 0.6]
            # turn right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['wheel_speed'] = [0.6, -0.6]
            # stop
            elif event.type == pygame.KEYUP or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
                self.command['wheel_speed'] = [0, 0]
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                # elif n_observed_markers < 3:
                #     self.notification = '> 2 landmarks is required for pausing'
                # else:
                #     if not self.ekf_on:
                #         self.request_recover_robot = True
                #     self.ekf_on = not self.ekf_on
                #     if self.ekf_on:
                #         self.notification = 'SLAM is running'
                #     else:
                #         self.notification = 'SLAM is paused'
                # read in the true map
                lms = []
                for i,lm in enumerate(aruco_true_pos):
                    measure_lm = Marker(np.array([[lm[0]],[lm[1]]]),i+1)
                    lms.append(measure_lm)
                # TODO: check for add landmarks 
                self.ekf.add_landmarks(lms)  

            # run path planning algorithm
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                self.generate_path_astar(search_list, fruit_list, fruit_true_pos)
                
            # run auto fruit searching
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                # if any(self.waypoints_list):
                #      self.rotate_robot(num_turns=8)
                self.command['auto_fruit_search'] = True
                
            # reset path planning algorithm
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self.waypoints_list = []
                self.ekf.reset()
                
                # read in the true map
                # fruit_list, fruit_true_pos, aruco_true_pos = read_true_map('M4_true_map.txt')
                lms = []
                for i,lm in enumerate(aruco_true_pos):
                    measure_lm = Marker(np.array([[lm[0]],[lm[1]]]),i+1)
                    lms.append(measure_lm)
                self.ekf.add_landmarks_true(lms)   
            
            # TODO Load true map into ekf by pressing t
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_t:
                if args.map:
                    self.ekf.load_true_map(args.map)
                    self.notification = 'Loading true map...'
                    self.command['load_true_map'] = True
                else: 
                    self.notification = "No true map found."
                
            # enable/disable object detection 
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                self.command['run_obj_detector'] = True
                
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True

        if self.quit:
            pygame.quit()
            sys.exit()

####################################### TRUE MAP AND MATH HELPER FUNCTIONS  #####################################################
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
        for i in range(len(fruit_list)):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit, fruit, np.round(fruit_true_pos[i][0], 1), np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1

# clamps the angle in radians, to range of -pi to pi
def clamp_angle(turning_angle_raw):
    turning_angle = turning_angle_raw % (2*np.pi) # shortcut to while loop deducting 2pi from the angle
    # if angle more than 180 deg, make a negative angle 
    turning_angle = turning_angle - 2*np.pi if turning_angle > np.pi else turning_angle
    return turning_angle

####################################### MAIN #####################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=5000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--yoloV8", default='YOLOv8/best_10k.pt')
    parser.add_argument("--map", type=str, default="m3set1.txt")
    args, _ = parser.parse_known_args()
    
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


######################################## LOOKIE HERE ########################################

    operate = Operate(args)
    fruit_list, fruit_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print("========================= GT ============================")
    fruit_goals = print_target_fruits_pos(search_list, fruit_list, fruit_true_pos)
    print("=========================================================")

    # TODO: define sequence of items here 
    while start:

        operate.update_keyboard_M4()
        operate.take_pic()
        drive_meas = operate.control()
        operate.detect_object()
        operate.slam_gui_update(drive_meas,canvas)
        # upon pressing 'w', this function completely takes over
        operate.auto_fruit_search()
