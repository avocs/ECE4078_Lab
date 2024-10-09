import cv2
import math
import json
import pygame
import numpy as np


class EKF:   
    # Implementation of an EKF for SLAM
    # The EKF state is a column vector, defined as [x; y; theta; x_lm1; y_lm1; x_lm2; y_lm2; ....]
    # lm stands for landmark, ie the aruco markers.
    # Therefore, the state is the concatenation of the robot state and the position of the landmarks.

    def __init__(self, robot):
        # State components
        self.robot = robot
        self.markers = np.zeros((2,0))
        self.taglist = []
        
        # NOTE sandra added
        self.update_landmarks_flag = True

        # Covariance matrix
        self.P = np.zeros((3,3))
        self.init_lm_cov = 1e3
        self.robot_init_state = None
        self.lm_pics = []
        for i in range(1, 11):
            f_ = f'./ui/8bit/lm_{i}.png'
            self.lm_pics.append(pygame.image.load(f_))
        f_ = f'./ui/8bit/lm_unknown.png'
        self.lm_pics.append(pygame.image.load(f_))
        self.pibot_pic = pygame.image.load(f'./ui/8bit/pibot_top.png')
        
        # NODE
        self.val_offset = 0.23
        self.val_offset_enabled = False
        
    def reset(self):
        self.robot.state = np.zeros((3, 1))
        
        # NOTE ADDED 
        self.robot.prev_state = np.zeros((3,1))

        # #TODO
        # self.robot.state[1,0] += 0.2/100.0
        if self.val_offset_enabled:
            y_robot_value = self.robot.state[1,0]
            if y_robot_value < 0:
                self.robot.state[1,0] += self.val_offset
            else:
                self.robot.state[1,0] -= self.val_offset

        self.markers = np.zeros((2,0))
        self.taglist = []
        # Covariance matrix
        self.P = np.zeros((3,3))
        self.init_lm_cov = 1e3
        self.robot_init_state = None
    

    # TODO M3, loads the true map coordinates into the workspace
    def load_true_map(self, map_file):
        with open(map_file,'r') as f:
            true_map = json.load(f)
            print(f'truemap: {true_map}')
        
        # reset
        # new_markers = np.zeros((2, 0))
        # print(new_markers)
        self.markers = np.zeros((2, 10))
        self.taglist = []

        for i, (tag, position) in enumerate(true_map.items()):
            # Extract x and y positions from the map
            if tag.startswith('aruco'):
                num_tag = i + 1
                print(f'i: {i}, tag: {tag}, position: {position}')
                self.markers[:, i] = [position['x'], position['y']]
                # Add the tag to the taglist
                self.taglist.append(int(num_tag))
                # self.taglist.append(int(tag.replace("aruco", "").replace("_0", "")))

        # Initialize the covariance matrix for the robot's state
        self.P = np.zeros((3 + 2 * len(self.taglist), 3 + 2 * len(self.taglist)))
        for j in range(len(self.taglist)):
            self.P[3 + 2*j, 3 + 2*j] = 0
            self.P[3 + 2*j + 1, 3 + 2*j + 1] = 0

        # Landmarks are now fixed, so we can disable updates
        self.update_landmarks_flag = False

    def number_landmarks(self):
        return int(self.markers.shape[1])

    def get_state_vector(self):
        state = np.concatenate(
            (self.robot.state, np.reshape(self.markers, (-1,1), order='F')), axis=0)
        return state
    
    # time to mod this guy 
    def set_state_vector(self, state):
        self.robot.state = state[0:3,:]
        # print(self.robot.state)
        self.markers = np.reshape(state[3:,:], (2,-1), order='F')
        # TODO 
        # print(f'markers:{self.markers}')
    
    def save_map(self, fname="slam.txt"):
        if self.number_landmarks() > 0:
            d = {}

            for i, tag in enumerate(self.taglist):
                
                d["aruco" + str(tag) + "_0"] = {"x": self.markers[0,i], "y":self.markers[1,i]}
            
            # d = { "taglist": self.taglist, "map": self.markers.tolist(), "covariance": self.P[3:,3:].tolist() }
        with open(fname, 'w') as map_f:
            json.dump(d, map_f, indent=4)

    def recover_from_pause(self, measurements):
        if not measurements:
            return False
        else:
            lm_new = np.zeros((2,0))
            lm_prev = np.zeros((2,0))
            tag = []
            for lm in measurements:
                if lm.tag in self.taglist:
                    lm_new = np.concatenate((lm_new, lm.position), axis=1)
                    tag.append(int(lm.tag))
                    lm_idx = self.taglist.index(lm.tag)
                    lm_prev = np.concatenate((lm_prev,self.markers[:,lm_idx].reshape(2, 1)), axis=1)
            if int(lm_new.shape[1]) > 2:
                R,t = self.umeyama(lm_new, lm_prev)
                theta = math.atan2(R[1][0], R[0][0])
                self.robot.state[:2]=t[:2]
                self.robot.state[2]=theta
                return True
            else:
                return False
            
    ##########################################
    # EKF functions
    # Tune your SLAM algorithm here
    # ########################################

    # the prediction step of EKF
    def predict(self, drive_meas):

        F = self.state_transition(drive_meas)       # This is A in lecture notes which is derivative matrix
        x = self.get_state_vector()                 # Obtaining the current state vector
        Q = self.predict_covariance(drive_meas)     # This is Sigma Q in lecture notes
        #TODO CHANGEF FRM 0.01 TO 0.1
        Q[0:3,0:3] += 0.01*np.eye(3)                # Motion model's noise covariance matrix      

        # TODO: add your codes here to compute the predicted x

        self.P = F @ self.P @ F.T + Q
        # 1. Predict state by calling the drive function
        # self.set_state_vector(F @ x)
        self.robot.drive(drive_meas)


        # Get the current state
        # x_pred = self.get_state_vector()
        # x_pred = F @ x


        # # 2. Derivate Drive which is A
        # # 3. Get covariance which is Q
        # # 4. Update the covariance matrix
        # P = self.P
        # P_pred = F @ P @ F.T + Q                         # (bar sigma k) = A * (sigma k-1) * (A transpose) + (sigma Q)
               
        # # Update the P (Covariance Matrix)
        # self.P = P_pred                                  # Bar Sigma k in lecture notes is P

        # Update the predicted state
        # self.set_state_vector(x_pred)    


    # the update/correct step of EKF
    def update(self, measurements):
        if not measurements:
            return


        # NOTE sandra is trying to see if update landmarks can be disabled here
        # Construct measurement index list
        tags = [lm.tag for lm in measurements]
        idx_list = [self.taglist.index(tag) for tag in tags]

        # Stack measurements and set covariance
        z = np.concatenate([lm.position.reshape(-1,1) for lm in measurements], axis=0)
        R = np.zeros((2*len(measurements),2*len(measurements)))
        for i in range(len(measurements)):
            #TODO CHANGED Y=T=FRM 0.1 TO 0.01
            R[2*i:2*i+2,2*i:2*i+2] = 0.1*np.eye(2)

        # Compute own measurements
        z_hat = self.robot.measure(self.markers, idx_list)              # This is h function
        z_hat = z_hat.reshape((-1,1), order="F")
        H = self.robot.derivative_measure(self.markers, idx_list)       # This is C in lecture notes. Jacobian Matrix

        x = self.get_state_vector()

        # TODO: add your codes here to compute the updated x
        # For Correction:
        # K = (bar sigma k) * (C transpose) * [(C) * (bar sigma k) * (C transpose) + (sigma R)]^-1

        S = H @ self.P @ H.T + R    # This is in the [square brackets]
        K = self.P @ H.T @ np.linalg.inv(S) # This is the K in lecture notes

        # z_hat = h function (bar mew k)
        # mew k = (bar mew k) + (K) * [z_k - h function (bar mew k)]        Which is x
        y = z - z_hat   # This is the calculate [square bracket]
        x = x + K @ y   # This is to calulcate mew k

        # Update the new state
        self.set_state_vector(x)
        # print(f"EKF.py correction pose {self.get_state_vector()[:3].flatten()}")

        # sigma k = (I - K * C) * (bar sigma k)
        # bar sigma k = P               lecture notes = code
        # C = H
        self.P = (np.eye(len(x)) - K @ H) @ self.P
        

    def state_transition(self, drive_meas):
        n = self.number_landmarks()*2 + 3
        F = np.eye(n)
        F[0:3,0:3] = self.robot.derivative_drive(drive_meas)
        return F
    
    def predict_covariance(self, drive_meas):
        n = self.number_landmarks()*2 + 3
        Q = np.zeros((n,n))
        Q[0:3,0:3] = self.robot.covariance_drive(drive_meas)
        return Q

    def add_landmarks(self, measurements):
        # NOTE if there aren't any measurements or if the true map is already provided, return 
        if not measurements or not self.update_landmarks_flag:
            return

        th = self.robot.state[2]

        # TODO added after the +
        # robot_xy = self.robot.state[0:2,:] + np.array([[0], [0.2/100]])
        robot_xy = self.robot.state[0:2,:]

        R_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])

        # Add new landmarks to the state
        for lm in measurements:
            if lm.tag in self.taglist:
                # ignore known tags
                continue
            
            lm_position = lm.position
            lm_state = robot_xy + R_theta @ lm_position

            self.taglist.append(int(lm.tag))
            self.markers = np.concatenate((self.markers, lm_state), axis=1)

            # Create a simple, large covariance to be fixed by the update step
            self.P = np.concatenate((self.P, np.zeros((2, self.P.shape[1]))), axis=0)
            self.P = np.concatenate((self.P, np.zeros((self.P.shape[0], 2))), axis=1)
            self.P[-2,-2] = self.init_lm_cov**2
            self.P[-1,-1] = self.init_lm_cov**2

    ##########################################
    ##########################################
    ##########################################

    @staticmethod
    def umeyama(from_points, to_points):

    
        assert len(from_points.shape) == 2, \
            "from_points must be a m x n array"
        assert from_points.shape == to_points.shape, \
            "from_points and to_points must have the same shape"
        
        N = from_points.shape[1]
        m = 2
        
        mean_from = from_points.mean(axis = 1).reshape((2,1))
        mean_to = to_points.mean(axis = 1).reshape((2,1))
        
        delta_from = from_points - mean_from # N x m
        delta_to = to_points - mean_to       # N x m
        
        cov_matrix = delta_to @ delta_from.T / N
        
        U, d, V_t = np.linalg.svd(cov_matrix, full_matrices = True)
        cov_rank = np.linalg.matrix_rank(cov_matrix)
        S = np.eye(m)
        
        if cov_rank >= m - 1 and np.linalg.det(cov_matrix) < 0:
            S[m-1, m-1] = -1
        elif cov_rank < m-1:
            raise ValueError("colinearility detected in covariance matrix:\n{}".format(cov_matrix))
        
        R = U.dot(S).dot(V_t)
        t = mean_to - R.dot(mean_from)
    
        return R, t

    # Plotting functions
    # ------------------
    @ staticmethod
    def to_im_coor(xy, res, m2pixel):
        w, h = res
        x, y = xy
        x_im = int(-x*m2pixel+w/2.0)
        y_im = int(y*m2pixel+h/2.0)
        return (x_im, y_im)
    
    def draw_slam_state(self, res = (320, 500), not_pause=True):
        # Draw landmarks
        m2pixel = 100
        if not_pause:
            bg_rgb = np.array([213, 213, 213]).reshape(1, 1, 3)
        else:
            bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
        canvas = np.ones((res[1], res[0], 3))*bg_rgb.astype(np.uint8)
        # in meters, 
        lms_xy = self.markers[:2, :]
        robot_xy = self.robot.state[:2, 0].reshape((2, 1))
        lms_xy = lms_xy - robot_xy
        robot_xy = robot_xy*0
        robot_theta = self.robot.state[2,0]
        # plot robot
        start_point_uv = self.to_im_coor((0, 0), res, m2pixel)
        
        p_robot = self.P[0:2,0:2]
        axes_len,angle = self.make_ellipse(p_robot)
        canvas = cv2.ellipse(canvas, start_point_uv, 
                    (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
                    angle, 0, 360, (0, 30, 56), 1)
        # draw landmards
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                # plot covariance
                Plmi = self.P[3+2*i:3+2*(i+1),3+2*i:3+2*(i+1)]
                axes_len, angle = self.make_ellipse(Plmi)
                canvas = cv2.ellipse(canvas, coor_, 
                    (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
                    angle, 0, 360, (244, 69, 96), 1)

        surface = pygame.surfarray.make_surface(np.rot90(canvas))
        surface = pygame.transform.flip(surface, True, False)
        surface.blit(self.rot_center(self.pibot_pic, robot_theta*57.3),
                    (start_point_uv[0]-15, start_point_uv[1]-15))
        if self.number_landmarks() > 0:
            for i in range(len(self.markers[0,:])):
                xy = (lms_xy[0, i], lms_xy[1, i])
                coor_ = self.to_im_coor(xy, res, m2pixel)
                try:
                    surface.blit(self.lm_pics[self.taglist[i]-1],
                    (coor_[0]-5, coor_[1]-5))
                except IndexError:
                    surface.blit(self.lm_pics[-1],
                    (coor_[0]-5, coor_[1]-5))
        return surface

    # def draw_slam_state(self, res = (320, 500), not_pause=True):
    #     # Draw landmarks
    #     m2pixel = 100
    #     if not_pause:
    #         bg_rgb = np.array([213, 213, 213]).reshape(1, 1, 3)
    #     else:
    #         bg_rgb = np.array([120, 120, 120]).reshape(1, 1, 3)
    #     canvas = np.ones((res[1], res[0], 3))*bg_rgb.astype(np.uint8)
    #     # in meters, 
    #     lms_xy = self.markers[:2, :]
    #     robot_xy = self.robot.state[:2, 0].reshape((2, 1))
    #     lms_xy = lms_xy - robot_xy
    #     robot_xy = robot_xy*0
    #     robot_theta = self.robot.state[2,0]
    #     # plot robot
    #     start_point_uv = self.to_im_coor((0, 0), res, m2pixel)
        
    #     p_robot = self.P[0:2,0:2]
    #     axes_len,angle = self.make_ellipse(p_robot)
    #     canvas = cv2.ellipse(canvas, start_point_uv, 
    #                 (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
    #                 angle, 0, 360, (0, 30, 56), 1)
    #     # draw landmards
    #     if self.number_landmarks() > 0:
    #         for i in range(len(self.markers[0,:])):
    #             xy = (lms_xy[0, i], lms_xy[1, i])
    #             coor_ = self.to_im_coor(xy, res, m2pixel)
    #             # plot covariance
    #             Plmi = self.P[3+2*i:3+2*(i+1),3+2*i:3+2*(i+1)]
    #             axes_len, angle = self.make_ellipse(Plmi)
    #             canvas = cv2.ellipse(canvas, coor_, 
    #                 (int(axes_len[0]*m2pixel), int(axes_len[1]*m2pixel)),
    #                 angle, 0, 360, (244, 69, 96), 1)

    #     surface = pygame.surfarray.make_surface(np.rot90(canvas))
    #     surface = pygame.transform.flip(surface, True, False)
    #     surface.blit(self.rot_center(self.pibot_pic, robot_theta*57.3),
    #                 (start_point_uv[0]-15, start_point_uv[1]-15))
    #     if self.number_landmarks() > 0:
    #         # TODO delete later
    #         # print(f'selftaglist: {self.taglist}')
    #         for i in range(len(self.markers[0,:])):
    #             xy = (lms_xy[0, i], lms_xy[1, i])
    #             coor_ = self.to_im_coor(xy, res, m2pixel)
    #             try:
    #                 surface.blit(self.lm_pics[self.taglist[i]-1],
    #                 (coor_[0]-5, coor_[1]-5))
    #             except IndexError:
    #                 surface.blit(self.lm_pics[-1],
    #                 (coor_[0]-5, coor_[1]-5))
    #     return surface

    @staticmethod
    def rot_center(image, angle):
        """rotate an image while keeping its center and size"""
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image       
 
    @staticmethod
    def make_ellipse(P):
        e_vals, e_vecs = np.linalg.eig(P)
        idx = e_vals.argsort()[::-1]   
        e_vals = e_vals[idx]
        e_vecs = e_vecs[:, idx]
        alpha = np.sqrt(4.605)
        axes_len = e_vals*2*alpha
        if abs(e_vecs[1, 0]) > 1e-3:
            angle = np.arctan(e_vecs[0, 0]/e_vecs[1, 0])
        else:
            angle = 0
        return (axes_len[0], axes_len[1]), angle