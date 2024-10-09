# sandra hates the world
# Latest version of the L3 code, backup for M4 
# Last update: 07/10/2024

# basic python packages
import cv2 
import time
import os, sys
import numpy as np
import shutil
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
import d_star_lite_path_planning as dstar
from d_star_lite_path_planning import Node, DStarLite
import object_pose_estyolo_M4 as obj_est
import object_pose_estyolo_M4 as live_fruit_pose_update


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
                        'auto_fruit_search_astar': False,
                        'auto_fruit_search_dstar': False,
                        'detect_and_est_fruit': False}
                        
        # TODO: Tune PID parameters here. If you don't want to use PID, set use_pid = 0
        # self.pibot_control.set_pid(use_pid=1, kp=0.1, ki=0, kd=0.0005)  
        # self.pibot_control.set_pid(use_pid=1, kp=0.005, ki=0, kd=0.0005)  
        self.pibot_control.set_pid(use_pid=1, kp=0.0001, ki=0, kd=0.001)  

        self.lab_output_dir = 'lab_output/'
        if not os.path.exists(self.lab_output_dir):
            os.makedirs(self.lab_output_dir)

        self.pred_output_dir = 'pred_output/'
        if not os.path.exists(self.pred_output_dir):
            os.makedirs(self.pred_output_dir)
        else:
            # Delete the folder and create an empty one, i.e. every operate.py is run, this folder will be empty.
            shutil.rmtree(self.pred_output_dir)
            os.makedirs(self.pred_output_dir)
        
        self.save_output_dir = 'save_output/'
        if not os.path.exists(self.save_output_dir):
            os.makedirs(self.save_output_dir)
        else:
            # Delete the folder and create an empty one, i.e. every operate.py is run, this folder will be empty.
            shutil.rmtree(self.save_output_dir)
            os.makedirs(self.save_output_dir)      

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
        self.control_clock = time.time()
        # self.control_clock = time.perf_counter()
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

        # NOTE dstar aux vars, doesnt work that well but uh
        self.spoofed_obs = []
        self.fruit_goals_remaining = []

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
        self.control_clock = time.time()
        initial_ticks = self.pibot_control.get_counter_values()
        ticks_travelled_left, ticks_travelled_right = 0,0

        # this is basically the while true loop of the operate's main 
        while True: 
            # call to operate take pic
            self.take_pic()

            # call to control
            left_speed, right_speed = self.pibot_control.set_velocity([lv, rv])
            dt = time.time() - self.control_clock
            # dt = time.perf_counter() - self.control_clock
            curr_ticks = self.pibot_control.get_counter_values()
            ticks_travelled_left = curr_ticks[0] - initial_ticks[0]
            ticks_travelled_right = curr_ticks[1] - initial_ticks[1]
            # print(f"Curr ticks: {curr_ticks}")
            # NOTE: this needs to be tuned! 
            drive_meas = Drive(0.3*left_speed, 0.3*right_speed, dt)
            self.control_clock = time.time()
            # self.control_clock = time.perf_counter()

            # call to update_slam
            self.slam_gui_update(drive_meas, canvas)
            # print(f"pose update: {self.get_robot_pose()}")

            if ticks_travelled_left >= num_ticks and ticks_travelled_right >= num_ticks: 
                break

        self.pibot_control.set_velocity([0,0])
        time.sleep(0.25)

        return drive_meas
    

    # rotate by 1 tick and call to update slam
    def rotate_step(self):

        initial_ticks = self.pibot_control.get_counter_values()
        ticks_travelled_left, ticks_travelled_right = 0,0
        wheel_rot_speed = 0.5
        lv, rv = -wheel_rot_speed, wheel_rot_speed

        self.control_clock = time.time()

        self.pibot_control.set_velocity([lv, rv])

        while True:
            curr_ticks = self.pibot_control.get_counter_values()
            ticks_travelled_left = curr_ticks[0] - initial_ticks[0]
            ticks_travelled_right = curr_ticks[1] - initial_ticks[1]
            if ticks_travelled_left >= 1 and ticks_travelled_right >= 1:
                break
        self.pibot_control.set_velocity([0,0])
        dt = time.time() - self.control_clock
        drive_meas = Drive(0.3*lv, 0.3*rv, dt)
        self.slam_gui_update(drive_meas, canvas)
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

    
   # SLAM with ARUCO markers -- Original      
    def update_slam_ori(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)
    
    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        measurements, self.aruco_img = self.aruco_sensor.detect_marker_positions(self.img)
        
        if self.ekf_on:
            
            # stores the last pose into 'prev_pose'
            self.prev_pose = self.get_robot_pose()
            self.ekf.predict(drive_meas)                # perform prediction based on motion model
            self.guessed_pose = self.get_robot_pose()   # the guessed pose is blindly fixed on the motion model calculated pose
            print(f"self.guessed_pose {self.guessed_pose}")
            print(f"self.prev_pose {self.prev_pose}")
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

                # TODO change to 2
                # if there are more than 3 aruco markers, 
                if (len(measurements) >= 2):        # if there are more than 3 aruco markers in sight
                    self.pos_read_flag = True       # the position to be read is the ekf corrected pose
                    print("Reading SLAM position)")


        print(f"Final pred pose {self.get_robot_pose()}")
        angle = self.ekf.robot.state[2,0] * 180/np.pi
        print(f"Angle: {angle}" )
        return len(measurements)

        
    # save SLAM map 
    def record_data(self):
        # this saves slam map to slam.txt upon pressing 's' 
        if self.command['save_slam']:
            self.ekf.save_map(fname=os.path.join(self.lab_output_dir, 'slam.txt'))
            self.notification = 'Map is saved'
            self.command['save_slam'] = False
        
        if self.command['save_obj_detector']:
            # this is what happens when you press n
            if self.obj_detector_output is not None:            
                    # obj_detector_output = (bounding_boxes, robot_state)
                print("Robot State: ", self.obj_detector_output[1])

                # write information to respective txt files, and obtain the image name
                self.pred_fname = self.obj_detector.write_image(self.obj_detector_output[0], self.obj_detector_output[1], self.pred_output_dir)
                image = self.pibot_control.get_image()
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # NOTE sandra 
                # save images to respective directories (note fin_prediction_image is just the colour converted image for visualising)
                fbbox = os.path.join(self.pred_output_dir, f'pred_{self.image_id}.png')
                f_ = os.path.join(self.save_output_dir, f'pred_{self.image_id}.png')
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
                self.command['wheel_speed'] = [-0.5, 0.4]
            # turn right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['wheel_speed'] = [0.4, -0.5]
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


############################## D* LITE FUNCTIONS FOR OPERATE CLASS ########################
    
    def generate_path_dstar(self):
        self.ox, self.oy = dstar.generate_obstacles(fruit_true_pos, aruco_true_pos)
        self.path_algo = DStarLite(self.ox, self.oy)

        sx, sy, gx, gy, fx, fy, face_angle = dstar.generate_points_L2(fruit_goals, aruco_true_pos)
        
        # Reset waypoints list 
        self.waypoints_list = []
        for i in range(len(sx)):
            # ive got no fucking clue what is happening here 
            _, pathx, pathy = self.path_algo.main(Node(x=sx[i], y=sy[i]), Node(x=gx[i], y=gy[i]), spoofed_ox=[[]], spoofed_oy=[[]])
            pathx.pop(0)
            pathy.pop(0)
            temp = [[x/10.0,y/10.0] for x, y in zip(pathx, pathy)]
            self.waypoints_list.append(temp)
            
        # Feedback
        print(f"Path generated: {self.waypoints_list}")




################################ MAIN ALGORITHMS ########################################

    def auto_fruit_search_AStar(self, canvas):
        '''
        Perform fruit search to all fruits in search list
        '''
        if self.command['auto_fruit_search_astar']:
            print(f"Starting L2 auto_fruit_search..")
            self.curr_waypoint_count = 0
            if any(self.waypoints_list):
                # fruit number for printing purposes
                curr_fruit = len(search_list) - len(self.waypoints_list) + 1

                if self.waypoints_list[0]:
                    self.curr_waypoint_count += 1

                    # robot tries to drive to the waypoint
                    print("\n----------------------------------------------------------")
                    waypoint_to_go = self.waypoints_list[0][0]
                    print(f"Fruit {curr_fruit}, Waypoint {self.curr_waypoint_count}: {waypoint_to_go}")
                    self.drive_to_point(waypoint_to_go, canvas)
                    robot_pose = self.get_robot_pose()

                    print("Finished driving to waypoint: {}; New robot pose: {}".format(self.waypoints_list[0][0],robot_pose))
                    print()
                    
                    # localise self at every sub-waypoint except for first and last at that point
                    if self.curr_waypoint_count > 1 and len(self.waypoints_list[0]) > 1:
                        self.localising_flag = True
                        self.localise_rotate_robot()
                        self.localising_flag = False

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
                self.command['auto_fruit_search_astar'] = False
    

    def auto_fruit_search_DStar(self, canvas):
        '''
        Perform fruit search to all fruits in search list
        '''
        if self.command['auto_fruit_search_dstar']:
            print(f"Starting L3 auto_fruit_search..")
            self.curr_waypoint_count = 0
            if any(self.waypoints_list):
                # fruit number for printing purposes
                curr_fruit = len(search_list) - len(self.waypoints_list) + 1

                if self.waypoints_list[0]:
                    self.curr_waypoint_count += 1

                    # robot tries to drive to the waypoint
                    print("\n----------------------------------------------------------")
                    waypoint_to_go = self.waypoints_list[0][0]
                    print(f"Fruit {curr_fruit}, Waypoint {self.curr_waypoint_count}: {waypoint_to_go}")
                    self.drive_to_point(waypoint_to_go, canvas)
                    robot_pose = self.get_robot_pose()

                    print("Finished driving to waypoint: {}; New robot pose: {}".format(self.waypoints_list[0][0],robot_pose))
                    print()
                    
                    # localise self at every sub-waypoint except for first and last at that point
                    #while (not self.is_close_to_waypoint(waypoint_to_go, self.get_robot_pose())):
                    if self.curr_waypoint_count > 1 and len(self.waypoints_list[0]) > 1:
                        self.localising_flag = True
                        self.localise_rotate_robot()
                        self.localising_flag = False
                        # self.drive_to_point(waypoint_to_go)

                    # remove that waypoint off the waypoint list
                    self.waypoints_list[0].pop(0)
                    print(f"New waypoints list: {self.waypoints_list}")

                    # if the waypoint is the last one in its list, means fruit is found
                    if not self.waypoints_list[0]:
                        self.waypoints_list.pop(0)
                        print(f"Before - Remaining fruit goals: {self.fruit_goals_remaining}")
                        self.fruit_goals_remaining = np.delete(self.fruit_goals_remaining, 0, axis=0)
                        print(f"After - Remaining fruit goals: {self.fruit_goals_remaining}")
                        self.curr_waypoint_count = 0
                        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                        print("Fruit reached! Robot sleeps for 3 seconds\n\n")
                        time.sleep(3)

                    # somehow there is localising here in not_our code 
                    # self.count_rot=self.count_rot+1
                    # if self.count_rot==4:
                    #     self.rotate_robot(num_turns=12)
                    #     self.count_rot=0

                    # not too sure what to do with this here rn
                    self.record_data()

            else:
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                print("Waypoint list is empty.\n\n")
                self.curr_waypoint_count = 0
                self.waypoints_list = []
                self.command['auto_fruit_search_dstar'] = False

########################## ROBOT DRIVE FUNCTIONS FOR OPERATE CLASS #####################
    # drive to a waypoint from current position
    def drive_to_point(self, waypoint, canvas):
        '''
        Function for robot to drive to a waypoint through the following procedures:
        1. Turn to waypoint
        2. Head straight to waypoint
        '''
        # compute x and y distance to waypoint
        robot_pose = self.get_robot_pose()
        print(f"Starting position: {robot_pose}")
        print(f"Robot is driving!")

        # 1. Robot rotates, turning towards the waypoint
        # ===================================================

        y_dist_to_waypt = waypoint[1] - (robot_pose[1] - self.camera_offset*np.sin(robot_pose[2]))
        x_dist_to_waypt = waypoint[0] - (robot_pose[0] - self.camera_offset*np.cos(robot_pose[2]))
        angle_to_waypt = np.arctan2(y_dist_to_waypt, x_dist_to_waypt) # angle measured in rad, from theta = 0
        # this is the angle that the robot needs to turn, in radians. sign determines direction of turning
        turning_angle = angle_to_waypt - robot_pose[2]         
        self.rotate_drive(turning_angle)

        time.sleep(0.5)
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
        if abs_turning_angle_deg > 45:
            tick_offset = 8
        elif abs_turning_angle_deg > 30:
            tick_offset = 5
        else: 
            tick_offset = 0

        # wheel circumference
        wheel_circum = np.pi * self.wheel_diameter
        # If the robot pivoted 360°, the distance traveled by each wheel = circumference of this pivot circle
        pivot_circum = np.pi * self.baseline 
        distance_per_wheel = abs(turning_angle / (2*np.pi)) * pivot_circum

        # distance each wheel must travel
        # distance_per_wheel = (baseline/2) * turning_angle
        turning_revolutions = distance_per_wheel / wheel_circum

        # if turn_ticks has been given be default, none of the previous calculations matter anymore
        if turn_ticks != 0:
            num_ticks = turn_ticks
        else:
            num_ticks = np.round(abs((turning_revolutions * self.ticks_per_revolution) + tick_offset))
        
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

        if dist_to_waypt > 0:
            self.tick_offset = 0
        else: 
            self.tick_offset = 0

        # wheel circumference
        wheel_circum = np.pi * self.wheel_diameter
        drive_revolutions = dist_to_waypt / wheel_circum
        num_ticks = np.round(drive_revolutions * self.ticks_per_revolution + self.tick_offset)

        # time to drive straight for 
        drive_time = dist_to_waypt / (self.scale * wheel_lin_speed)
        drive_speeds = [wheel_lin_speed, wheel_lin_speed] 

        # number of ticks to drive striaght for
        print(f'/// Driving for {num_ticks:.2f} ticks to {dist_to_waypt:.2f}')

        # Drive to point, updating slam throughout
        if num_ticks != 0:

            # NOTE detect if fruit in path before driving
            update_flag = self.live_estimate_and_path_replanning()
            # do not drive if fruit in the way
            if update_flag:
                return

            self.control_tick(drive_speeds[0], drive_speeds[1], num_ticks)
        

        return None
    
    """
    Getter method to call to get robot state from ekf
    """
    def get_robot_pose(self):
        '''
        Returns the current robot pose from ekf
        '''    
        return self.ekf.robot.state.squeeze().tolist()


    
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
        counter = 0
        while True:
            if counter >= 20:                   # this is taking too long
                break
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
            counter+=1
        return robot_pose
    

    """
    Function to have robot pan and find markers to localise itself
    """
    def localise_rotate_robot(self, num_turns=0, wheel_rot_speed=0.5):

        print("Robot panning and localising....")

        num_rotations = 0
        turning_angle = np.pi/24            # 1 tick increments      
        num_turns = int(2*np.pi / turning_angle)

        for i in range(num_turns):
            print(f'Rotation: {i+1}, Total turned: {turning_angle*i}')
            
            # fixed to turn 1 tick at a time, updating slam along the way
            self.rotate_step()            
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
        threshold = 0.05

        return abs(x - x_goal) <= threshold and abs(y - y_goal) <= threshold
    
    # NOTE: newnew nightmare for vision! 
    def live_estimate_and_path_replanning(self, target_fruit=None, target_fruit_true_pos=None):

        # if the keyboard calls to detect and estimate fruit pose
        if self.command['detect_and_est_fruit']: 
            
            self.obj_est = {}

            # doing this twice..?
            for _ in range(2): 
                # take a picture
                self.take_pic()

                # run one detection
                self.command['run_obj_detector'] = True
                self.detect_object()
                self.command['run_obj_detector'] = False

                self.draw(canvas)
                pygame.display.update()

                # save that estimation
                self.command['save_obj_detector'] = True
                self.record_data()
                self.command['save_obj_detector'] = False

            # run estimations on the image saved 
            self.obj_est = obj_est.main()

        self.command['detect_and_est_fruit'] = False

        # generate new waypoints, not sure what to do with this ret value 
        update_flag = self.path_update()
    
        return update_flag


    def path_update(self):

        update_flag= False

        # check if the fruit is in the list
        for label, pose in self.obj_est.items():
            if label not in fruit_list: 
                fruit_x= pose['x']
                fruit_y= pose['y']

                # what is going on here 
                # snap to grid
                fruit_x = dstar.round_nearest(fruit_x, 0.4) # can change to round number 1
                fruit_y = dstar.round_nearest(fruit_y, 0.4)
                
                fruit_coord= np.array([fruit_x,fruit_y])
                print(f"Fruit detected: {fruit_coord}")

                if self.spoofed_obs:
                    if not(fruit_coord==self.spoofed_obs).all(1).any():
                        self.spoofed_obs.append(fruit_coord)
                        print(f"New obstacle detected at position:{fruit_coord}")
                        update_flag = True

                    else: 
                        self.spoofed_obs.append(fruit_coord)
                        print(f"New obstacle detected at position:{fruit_coord}")
                        update_flag = True

                # if the code calls to update
                if update_flag:
                    obs_x,obs_y = dstar.generate_spoofed_obs(self.spoofed_obs)
                    self.ox.extend(obs_x)
                    self.oy.extend(obs_y)
                    self.path_algo= DStarLite(self.ox,self.oy)

                    # im not even gonna mess with htis number anymore... 
                    current_pose = self.get_robot_pose
                    current_x = current_pose[0]
                    current_y = current_pose[1]
                    rounded_x = dstar.round_nearest(current_x,0.2)
                    rounded_y = dstar.round_nearest(current_y,0.2)
                    current_pose = [rounded_x,rounded_y]

                    # what even 
                    sx, sy, gx, gy, fx, fy, face_angle = dstar.generate_points_L3(current_pose, self.fruit_goals_remaining, aruco_true_pos, self.spoofed_obs)

                    # generate new path
                    new_waypoints_list= []
                    for i in range (len(sx)):
                        _, path_x, path_y = self.path_algo.main(Node(x=sx[i], y=sy[i]), Node(x=gx[i], y=gy[i]), spoofed_ox=[[]], spoofed_oy=[[]])
                        path_x.pop(0)
                        path_y.pop(0)
                        temp = [[x/10.0,y/10.0] for x, y in zip(path_x, path_y)]
                        self.new_waypoints_list.append(temp)

                    self.waypoints_list = new_waypoints_list
                    self.waypoints_list[0].insert(0,current_pose)

                    print(f"\n\nNew path generated: {self.waypoints_list}")

                return update_flag
        


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
                # NOTE by default, read in the true map
                lms = []
                for i,lm in enumerate(aruco_true_pos):
                    measure_lm = Marker(np.array([[lm[0]],[lm[1]]]),i+1)
                    lms.append(measure_lm)
                # TODO: check for add landmarks 
                self.ekf.add_landmarks(lms)  

            # run path planning algorithm
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                self.generate_path_astar(search_list, fruit_list, fruit_true_pos)
                
            # run auto fruit searching astar
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_w:
                self.command['auto_fruit_search_astar'] = True
                
            # run auto fruit searching dstar
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                self.command['auto_fruit_search_dstar'] = True
            

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
            
            # take fruit pic, save it, and estimate where it is 
            # at this point i dont give a fuck what the keybindings are
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                self.command['detect_and_est_fruit'] = True


                
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
    parser.add_argument("--map", type=str, default="m4demo.txt")
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
        # QUESTION: should fruit est be done before or after update slam?
        # would it be the latter 
        operate.live_estimate_and_path_replanning()
        operate.slam_gui_update(drive_meas,canvas)
        # upon pressing 'w', this function completely takes over
        operate.auto_fruit_search_AStar(canvas) 
        operate.auto_fruit_search_DStar(canvas)
