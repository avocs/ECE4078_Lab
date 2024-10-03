# teleoperate the robot, perform SLAM and object detection
# Adapted stuffs for M4 
# first fucked around by: sandra
# date time fuck around: 25/09/2024 1:16am
# last date time fuck around: 29/09/2024 1am

# NOTE:
'''
yoinked operate and adapted code in a futile attempt to run astar. 
How to run this file: 
python afruitsearch2.py --ip 192.168.137.45 --map m4test1.txt 
1. enter pygame
2. press ENTER -- starts slam
3. press T -- loads true map
4. (optional for now) press P -- enables object detection
5. press A -- generate search path, close matplot window when done looking
6. press W - start auto fruit search

'''

# basic python packages
import cv2 
import time
import shutil
import os, sys
import numpy as np

# import utility functions
import pygame # python package for GUI
from pibot import Drive
from pibot import PibotControl # access the robot

# import SLAM components (M2)
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
from slam.aruco_sensor import ArucoSensor

# import CV components (M3)
#sys.path.insert(0,"{}/cv/".format(os.getcwd()))
from YOLOv8.detector import ObjectDetector

# M4 - Autonomous fruit searching
import ast
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import math
from slam.aruco_sensor import Marker
from object_pose_estyolo_M4 import live_fruit_pose_update
import copy
import a_star_path_planning as astar
import object_pose_estyolo as obj_est
import d_star_lite_m4 as dstar



class Operate:
    def _init_(self, args):
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
        self.pibot_control.set_pid(use_pid=1, kp=0.0001, ki=0, kd=0.001)  

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
        self.control_clock = time.time()
        self.img = np.zeros([480,640,3], dtype=np.uint8)
        self.aruco_img = np.zeros([480,640,3], dtype=np.uint8)
        self.obj_detector_pred = np.zeros([480,640], dtype=np.uint8)        
        self.bg = pygame.image.load('ui/gui_mask.jpg')

        # TODO sandra: aux objects/vars added for m4
        self.waypoints_list = []
        # self.default_rot_speeds = [0.4, -0.4]   # pivot during rotate-n-search
        self.curr_waypoint_count = 0
        self.update_slam_flag = False

    # wheel control
    def control(self):       
        left_speed, right_speed = self.pibot_control.set_velocity(self.command['wheel_speed'])
        dt = time.time() - self.control_clock
        drive_meas = Drive(left_speed, right_speed, dt)
        self.control_clock = time.time()
        return drive_meas
        
    # camera control
    def take_pic(self):
        self.img = self.pibot_control.get_image()

    # wheel and camera calibration for SLAM
    def init_ekf(self, calib_dir, ip):
        fileK = os.path.join(calib_dir, 'intrinsic.txt')
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = os.path.join(calib_dir, 'distCoeffs.txt')
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = os.path.join(calib_dir, 'scale.txt')
        scale = np.loadtxt(fileS, delimiter=',')
        fileB = os.path.join(calib_dir, 'baseline.txt')
        baseline = np.loadtxt(fileB, delimiter=',')
        # sandra: added objects 
        self.scale = scale
        self.baseline = baseline
        self.camera_matrix = camera_matrix
        robot = Robot(baseline, scale, camera_matrix, dist_coeffs)
        return EKF(robot)
    

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        measurements, self.aruco_img = self.aruco_sensor.detect_marker_positions(self.img)
        # if self.request_recover_robot:
        #     is_success = self.ekf.recover_from_pause(measurements)
        #     if is_success:
        #         self.notification = 'Robot pose is successfuly recovered'
        #         self.ekf_on = True
        #     else:
        #         self.notification = 'Recover failed, need >2 landmarks!'
        #         self.ekf_on = False
        #     self.request_recover_robot = False
        if self.ekf_on:
            self.ekf.predict(drive_meas)
             
            # M3, disable updates to aruco markers if true map is loaded
            if self.command['load_true_map']:
                self.notification = 'SLAM locates robot pose'
                self.command['load_true_map'] = False
            else: 
                if self.update_slam_flag  and len(measurements) >= 2:
                    print('Updating SLAM')
                    self.ekf.add_landmarks(measurements)
                    self.ekf.update(measurements)
                    self.notification = 'Updating SLAM'
                else:
                    # self.ekf.add_landmarks(measurements)
                    # self.ekf.update(measurements)
                    self.notification = 'Insufficient markers for updating SLAM'


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

    # Keyboard teleoperation    UNUSUED FOR M4
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

        for i in range(len(fruits_true_pos)):
            x_fruits.append(fruits_true_pos[i][0])
            y_fruits.append(fruits_true_pos[i][1])
        plt.scatter(x_fruits, y_fruits, c=fruit_colour, s=100)
        for i, (x, y) in enumerate(zip(x_fruits, y_fruits), 1):
            plt.annotate(f'{i+10}', (x, y), textcoords="offset points", xytext=(0,-10), ha='center')
        
        plt.grid(True)
        plt.show()    

    # TODO will need to mod here
    def generate_path_astar(self, search_list, fruits_list, fruits_true_pos):
        '''
        This is the a* algorithm implemented to generate waypoints towards all the fruits to search for
        '''
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

        src_coord = [0,0]  
        waypoints = []  
        self.waypoints_list = []


        for i in range(len(search_list)):
            # define one fruit as the destination, from the start of the search list
            grid_src = astar.convert_coord_to_grid(src_coord)
            dest_coord = search_true_pos[i]
                 # Call the Convert Coordinates to Grid Function to convert dest_coord to dest_grid
        dest_grid = astar.convert_coord_to_grid(dest_coord)
        # print("Dest Grid: ", dest_grid)
        
        # Call the Modify Obstacles Function to modify the grid every new run. Setting the current destination as an obstacle and the next destination as not an obstacle
        grid = astar.modify_obstacles(aruco_true_pos.tolist(), search_index[i], fruits_true_pos.tolist())
        
        distances = []
        # print(search_index)
        col = fruits_true_pos[search_index[i]][0]
        row = fruits_true_pos[search_index[i]][1]
        for j in range(len(aruco_true_pos)):
            value = astar.calculate_h_value(row, col, aruco_true_pos[j])
            distances.append(value)
            # print("Distance ", i, " ", value)
        
        count = 0
        for k in range(len(aruco_true_pos)):
            if distances[k] <= 0.3:
                count += 1
        
        if count >= 1:
            dest_grid = astar.modify_destinations(dest_grid, grid)
            # print("Came here!")
        else:
            # print("Else!")
            for l in range (len(dest_coord)):
            # print(dest_coord)
            # print(dest_coord[j])
                if dest_coord[l] < 0:
                    dest_coord[l] += astar.how_far_from_fruits
                    
                else:
                    dest_coord[l] -= astar.how_far_from_fruits
                dest_coord[l] = round(dest_coord[l], 2)
            dest_grid = astar.convert_coord_to_grid(dest_coord)
           
            waypoints = astar.a_star_search(grid, grid_src, dest_grid)
            simplified_waypoints = astar.simplify_path(waypoints)
            self.waypoints_list.append(simplified_waypoints)
            astar.plot_waypoints(simplified_waypoints)
            src_coord = dest_coord
            # Feedback
        print(f"Path generated: {self.waypoints_list}") 
        astar.plot_full_map(aruco_true_pos, fruits_copy)
        # self.display_map(aruco_true_pos, fruits_copy)



############################### D* function for operate class ##########################

# TODO Cheryl
    def generate_path_dstar(self):
        self.ox, self.oy = generate_obstacles(fruit_true_pos, aruco_true_pos)
        self.path_planning = dstar(self.ox, self.oy)
        
        sx, sy, gx, gy, fx, fy, face_angle = generate_points_L2(fruit_goals, aruco_true_pos)
        
        self.waypoints_list = []
        for i in range(len(sx)):
            _, pathx, pathy = self.path_planning.main(Node(x=sx[i], y=sy[i]), Node(x=gx[i], y=gy[i]), spoofed_ox=[[]], spoofed_oy=[[]])
            pathx.pop(0)
            pathy.pop(0)
            temp = [[x/10.0,y/10.0] for x, y in zip(pathx, pathy)]
            self.waypoints_list.append(temp)
            
        # Feedback
        print(f"Path generated: {self.waypoints_list}") 


    # TODO stuckkkkk
    def fruit_detect_obstacle(self):
        self.take_pic()

        # to detect the obj in detector
        self.obj_detector_pred, self.prediction_img = self.obj_detector.detect_single_image(self.img)
        self.fin_prediction_img = cv2.cvtColor(self.prediction_img, cv2.COLOR_RGB2BGR)
        self.obj_detector_output = (self.obj_detector_pred, self.ekf.robot.state.tolist())


        # the raw image, predicted image, predicted pose are saved
        self.pred_fname = self.obj_detector.write_image(self.obj_detector_output[0], self.obj_detector_output[1], self.pred_output_dir)

        # estimate the position of the fruit
        fruit_est= live_fruit_pose_update()
        print("The detected fruit position: {fruit_est}")

        # check whetehr there are obstacles found
        obstacles_found_flag=0

        # if the fruit detected does not match the one in the fruit list, get the coordinates of the obstacles
        for key in fruit_est:
            if (key.split('_')[0]) not in fruit_list:
                obstacle_fruit_x= fruit_est[key]['x']
                obstacle_fruit_y= fruit_est[key]['y']

                obstacle_fruit_x= round_nearest(obstacle_fruit_x, 0.4)
                obstacle_fruit_y= round_nearest(obstacle_fruit_y, 0.4)

                obstacle_fruit_coord= np.array([obstacle_fruit_x,obstacle_fruit_y])
                
                if self.spoofed_obs:
                    if not (obstacle_fruit_coord == self.spoofed_obs).all(1).any():
                        self.spoofed_obs.append(obstacle_fruit_coord) 
                        print(f"New obstacles detected at position: {obstacle_fruit_coord}")  
                        obstacles_found_flag = 1
                else:
                    self.spoofed_obs.append(obstacle_fruit_coord) # list of array
                    print(f"New obstacles detected at position: {obstacle_fruit_coord}")  
                    obstacles_found_flag = 1
                print(self.spoofed_obs)
        
        print(f"Update flag: {obstacles_found_flag}")

        if obstacles_found_flag: 
            spoofed_ox, spoofed_oy = generate_spoofed_obs(self.spoofed_obs)
            self.ox.extend(spoofed_ox)
            self.oy.extend(spoofed_oy)
            self.path_planning = dstar(self.ox, self.oy)
            
            curr_pose = self.ekf.robot.state.squeeze().tolist()
            x = round_nearest(curr_pose[0], 0.2)
            y = round_nearest(curr_pose[1], 0.2)
            curr_pose = [x, y]
            
            sx, sy, gx, gy, fx, fy, face_angle = generate_points_L3(curr_pose, self.fruit_goals_remain, aruco_true_pos, self.spoofed_obs)
                
            # generate new path, continued from before meeting obstacles
            waypoints_list_new = []
            for i in range(len(sx)):
                _, pathx, pathy = self.path_planning.main(Node(x=sx[i], y=sy[i]), Node(x=gx[i], y=gy[i]), spoofed_ox=[[]], spoofed_oy=[[]])

                pathx.pop(0)
                pathy.pop(0)
                    
                temp = [[x/10.0,y/10.0] for x, y in zip(pathx, pathy)]
                waypoints_list_new.append(temp)
                
            self.waypoints_list = waypoints_list_new
            self.waypoints_list[0].insert(0, curr_pose)
            # Feedback
            print(f"New path generated due to fruit: {self.waypoints_list}")
            
        return obstacles_found_flag




################################ MAIN ALGORITHM ########################################

    def auto_fruit_search(self, canvas):
        '''
        Perform fruit search to all fruits in search list
        '''

        if self.command['auto_fruit_search']:
            print(f"Starting auto_fruit_search..")
            if any(self.waypoints_list):
                # fruit number for printing purposes
                curr_fruit = len(search_list) - len(self.waypoints_list) + 1
                if self.waypoints_list[0]:
                    # waypoint_to_go = self.waypoints_list[0][0]
                    self.curr_waypoint_count += 1

                    # robot tries to drive to the waypoint
                    print("\n---------------------------------------")
                    print(f"Fruit {curr_fruit}, Waypoint {self.curr_waypoint_count}: {self.waypoints_list[0][0]}")
                    _ , _ = self.drive_to_point(self.waypoints_list[0][0], canvas)
                    robot_pose = self.get_robot_pose()
                    print("Finished driving to waypoint: {}; New robot pose: {}".format(self.waypoints_list[0][0],robot_pose))
                    print()
                    
                    # remove that waypoint off the waypoint list
                    self.waypoints_list[0].pop(0)
                    print(f"New waypoints list: {self.waypoints_list}")

                    # localise self at every sub-waypoint except for first and last at that point
                    if self.curr_waypoint_count > 1 and self.waypoints_list[0]:
                        self.update_slam_flag = True
                        self.localise_rotate_robot()
                        self.update_slam_flag = False
                    # if the waypoint is the last one in its list, means fruit is found
                    if not self.waypoints_list[0]:
                        self.waypoints_list.pop(0)
                        self.curr_waypoint_count = 0
                        print("Fruit reached, robot sleeps for 3 seconds")
                        time.sleep(3)

                    self.record_data()
            else:
                print("Waypoints list is empty")
                self.curr_waypoint_count = 0
                self.waypoints_list = []
                self.command['auto_fruit_search'] = False
    
########################## ROBOT DRIVE FUNCTIONS FOR OPERATE CLASS #####################

# TODO add fruit_detect_obstacle


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
        y_dist_to_waypt = waypoint[1] - robot_pose[1]
        x_dist_to_waypt = waypoint[0] - robot_pose[0]
        angle_to_waypt = np.arctan2(y_dist_to_waypt, x_dist_to_waypt) # angle measured in rad, from theta = 0
        # this is the angle that the robot needs to turn, in radians. sign determines direction of turning
        turning_angle = angle_to_waypt - robot_pose[2]         # compute minimum turning angle to waypoint
        

        # print(f' curr orientation {robot_pose[2]}, angle_towaypt {angle_to_waypt}, turning_angle {turning_angle}')
        turn_drive_meas = self.robot_move_rotate(turning_angle)

        time.sleep(0.5)
        # self.take_pic()

        robot_pose = self.get_robot_pose()
        # print("Turning for {:.2f} seconds".format(turning_time))
        print(f"Post Turn Position: {robot_pose}\n")
        # print(f"Post Turn Position: {operate.ekf.robot.state.squeeze().tolist()}")
        # print(f"Position: {robot_pose[0]:.2f}, {robot_pose[1]:.2f}, {robot_pose[2] + turning_angle:.2f}")        

        # 2. Before driving stright towards waypt, scan to check whether there is 
        # new obstacles detected
        self.fruit_detect_obstacle()


        # 3. Robot drives straight towards waypt
        # ===============================================
        dist_to_waypt = math.hypot(x_dist_to_waypt, y_dist_to_waypt)
        print(f' --- dist to waypoint: {x_dist_to_waypt}, {y_dist_to_waypt}')
        # self.take_pic()
        straight_drive_meas = self.robot_move_straight(dist_to_waypt)
        
        time.sleep(0.5)
            
        # print("Driving for {:.2f} seconds".format(drive_time))
        # print(f"Robot Pose: {robot_pose}\n")
        print(f"Post Drive Position: {operate.ekf.robot.state.squeeze().tolist()}")
        # print(f"Position: {waypoint[0], waypoint[1], robot_pose[2] + turning_angle}")
        

        print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))

        return turn_drive_meas, straight_drive_meas
    
   ######################################################### 
    # TODO nyoom nyoom
    def robot_move_rotate(self, turning_angle=0, turn_ticks=0,wheel_lin_speed=0.5, wheel_rot_speed=0, rotate_speed_offset=0.05):
        '''
        This function makes the robot turn a fixed angle by counting encoder ticks

        '''
        wheel_rot_speed_big = 0.5
        wheel_rot_speed_small = 0.6
        # global robot_pose

        ticks_per_revolution = 20
        wheel_diameter = 68e-3            # yoinked from cytron
        baseline = self.baseline

        # clamp angle between -180 to 180 deg 
        turning_angle = turning_angle % (2*np.pi) # shortcut to while loop deducting 2pi 
        # if the angle is more than 180 deg, make this a negative angle instead 
        turning_angle = turning_angle - 2*np.pi if turning_angle > np.pi else turning_angle
        turning_angle_deg = turning_angle * 180 / np.pi

        abs_turning_angle_deg = abs(turning_angle_deg)
        if abs_turning_angle_deg > 45:
            tick_offset = 8
        elif abs_turning_angle_deg > 30:
            tick_offset = 5
        else: 
            tick_offset = 0

        # if a wheel rotation speed is not specified as an argument, this determines the speed it should use
        if wheel_rot_speed == 0:
            if abs(turning_angle) >= (np.pi/4):
                wheel_rot_speed = wheel_rot_speed_big
            else:
                wheel_rot_speed = wheel_rot_speed_small
        else: 
            wheel_rot_speed = wheel_rot_speed

        # move the robot to perform rotation 

        # wheel circumference
        wheel_circum = np.pi * wheel_diameter
        #  If the robot pivoted 360°, the distance traveled by each wheel 
        # would be equal to the circumference of this pivot circle
        pivot_circum = np.pi * baseline 
        distance_per_wheel = abs(turning_angle / (2*np.pi)) * pivot_circum

        turning_time = distance_per_wheel/(self.scale * wheel_rot_speed)

        # distance each wheel must travel
        # distance_per_wheel = (baseline/2) * turning_angle
        turning_revolutions = distance_per_wheel / wheel_circum
        if turn_ticks != 0:
            num_ticks = turn_ticks
        else:
            num_ticks = np.round(abs((turning_revolutions * ticks_per_revolution) + tick_offset))
        
        # manual override
        print(f' /// Turning for {num_ticks:.2f} ticks to {turning_angle_deg:.2f}')
        # print(f"turning for {turning_time}s to {turning_angle_deg}")

        # -- direction of wheels, depending on sign
        if num_ticks != 0: # if the car is not going straight/has to turn
            if (turning_angle) > 0: # turn left 
                lv, rv = [-wheel_rot_speed, wheel_rot_speed+rotate_speed_offset]
            elif turning_angle < 0: # turn right
                lv, rv = [wheel_rot_speed+rotate_speed_offset, -wheel_rot_speed] 
        else: 
            lv, rv = [0.0, 0.0]
        
        turn_speeds = [lv, rv]

        # alt nyoom 
        initial_ticks = self.pibot_control.get_counter_values()
        ticks_travelled_left, ticks_travelled_right = 0,0

        if turning_angle != 0:
            # NOTE: sandra tried moving this line into turn speeds
            self.pibot_control.set_velocity(turn_speeds)

            while True: 
                self.take_pic()
                # self.pibot_control.set_velocity(turn_speeds)
                curr_ticks = self.pibot_control.get_counter_values()
                ticks_travelled_left = curr_ticks[0] - initial_ticks[0]
                ticks_travelled_right = curr_ticks[1] - initial_ticks[1]
                # print(f"curr_ticks {curr_ticks}")

                if ticks_travelled_left >= num_ticks and ticks_travelled_right >= num_ticks:
                    break
                # self.draw(canvas)
                # pygame.display.update()
            self.pibot_control.set_velocity([0,0])

        # update own location only after finished driving
        time.sleep(0.5)
        self.take_pic()
        # turn_drive_meas = Drive(1*lv, 1*rv, turning_time)  
        turn_drive_meas = Drive(0.2*lv, 0.2*rv, turning_time)   
 
        # print(f'turndrv {turn_drive_meas.left_speed} {turn_drive_meas.right_speed} {turn_drive_meas.dt}')
        self.update_slam(turn_drive_meas)
        #turn_drive_meas = Drive(0.9*lv, 0.9*rv, turning_time)
        turn_drive_meas = Drive(1*lv, 1*rv, turning_time)  
        # turn_drive_meas = Drive(lv/0.2, rv/0.2, turning_time)

        # self.update_slam(turn_drive_meas)

        # update display 
        self.draw(canvas)
        pygame.display.update()

        return turn_drive_meas 
#####################################################

    # TODO nyoom nyoom 
    def robot_move_straight(self, dist_to_waypt=0, wheel_lin_speed=0.6, wheel_rot_speed=0.4):
        '''
        this function makes the robot drive straight a certain time automatically 
        '''
        ticks_per_revolution = 20
        wheel_diameter = 68e-3            # yoinked from cytron

        if dist_to_waypt > 0:
            tick_offset = 17
        else: 
            tick_offset = 0

        # wheel circumference
        wheel_circum = np.pi * wheel_diameter
        drive_revolutions = dist_to_waypt / wheel_circum
        num_ticks = np.round(drive_revolutions * ticks_per_revolution + tick_offset)

        # time to drive straight for 
        drive_time = dist_to_waypt / (self.scale * wheel_lin_speed)
        lv, rv = wheel_lin_speed, wheel_lin_speed
        drive_speeds = [lv, rv] 

        # number of ticks to drive striaght for
        print(f'/// Driving for {num_ticks:.2f} ticks to {dist_to_waypt:.2f}')
        # print(f"driving for {drive_time}s")

        # alt nyoom 
        initial_ticks = self.pibot_control.get_counter_values()
        ticks_travelled_left, ticks_travelled_right = 0,0
        # NOTE : sandra tried moving the next line into the while loop
        self.pibot_control.set_velocity(drive_speeds)

        while True: 
            # self.take_pic()
            self.pibot_control.set_velocity(drive_speeds)
            curr_ticks = self.pibot_control.get_counter_values()
            # print(f"curr_ticks {curr_ticks}")

            ticks_travelled_left = curr_ticks[0] - initial_ticks[0]
            ticks_travelled_right = curr_ticks[1] - initial_ticks[1]
            # self.draw(canvas)
            # pygame.display.update()
            if ticks_travelled_left >= num_ticks and ticks_travelled_right >= num_ticks:
                break
        
        self.pibot_control.set_velocity([0,0])

        # update own location only after finished driving
        time.sleep(0.5)
        self.take_pic()
        straight_drive_meas = Drive(1.0*lv, 1.0*rv, drive_time)
        # print(f'strdrv {straight_drive_meas.left_speed} {straight_drive_meas.right_speed} {straight_drive_meas.dt}')
        self.update_slam(straight_drive_meas)

        # update display 
        self.draw(canvas)
        pygame.display.update()

        return straight_drive_meas

    def get_robot_pose(self):
        '''
        Returns the current robot pose
        
        '''    
        return self.ekf.robot.state.squeeze().tolist()
    
############################ WAYPOINT UPDATE AND SLAM HELPER FUNCTIONS FOR OPERATE CLASS ####################

    # TODO added, not sure if i still need it, probably yes
    def location_update(self, steps=2):
        for _ in range(steps):
            self.take_pic()
            lv, rv = self.pibot_control.set_velocity([0, 0])
            drive_meas = Drive(lv, rv, 0.0)
            # self.update_slam(drive_meas)
        
            # update pygame display
            self.draw(canvas)
            pygame.display.update()
        
    # TODO dra dra modified
    def localise_rotate_robot(self, num_turns=0, wheel_rot_speed=0.5):

        print("Robot trying to localise itself..")

        turning_angle = np.pi/12            # 15 deg increments
        num_turns = int(2*np.pi / turning_angle)

        # perform rotations and update location with each turn
        for i in range(num_turns):
            print(f'Rotation: {i}, Total turned: {turning_angle*i}')
            self.robot_move_rotate(turning_angle=turning_angle, wheel_rot_speed=wheel_rot_speed)
            print(f"Position after rotating: {self.get_robot_pose()}")
            time.sleep(1)

        # recover initial orientation prior to turning
        # self.robot_move_rotate(-turning_angle*num_turns, wheel_rot_speed=wheel_rot_speed)

        # print(f"Position after rotating: {self.get_robot_pose()}")

        return None

    
    # NOTE: implement cv for checks
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



########################### NOTE: KEYBOARD OPERATION DONE HERE ##############################
    
    # Keyboard control for Milestone 4 Level 2
    def update_keyboard_M4(self):
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
                # fruit_list, fruit_true_pos, aruco_true_pos = read_true_map('M4_true_map.txt')
                lms = []
                for i,lm in enumerate(aruco_true_pos):
                    measure_lm = Marker(np.array([[lm[0]],[lm[1]]]),i+1)
                    lms.append(measure_lm)
                # TODO: check for add landmarks 
                self.ekf.add_landmarks(lms)  

            # run path planning algorithm
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                # self.generate_path_astar(search_list, fruit_list, fruit_true_pos)
                self.generate_path_dstar(search_list,fruit_list,fruit_true_pos)
                
            # drive to waypoints
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

####################################### END OPERATE CLASS DEFINITION  ########################################################################
####################################### TRUE MAP AND SEARCH LIST HELPER FUNCTIONS  #####################################################
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
        for i in range(5):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit, fruit, np.round(fruit_true_pos[i][0], 1), np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1

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

        # 
        operate.update_keyboard_M4()
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.detect_object()
        # update pygame display
        operate.draw(canvas)
        pygame.display.update()
        
        # upon pressing 'w', this function completely takes over
        operate.auto_fruit_search(canvas)
