import numpy as np


class Robot:
    def __init__(self, baseline, scale, camera_matrix, dist_coeffs):
        # State is a 3 x 1 vector containing information on x-pos, y-pos, and orientation, ie [x; y; theta]
        # Positive x-axis is the direction the robot is facing, positive y-axis 90 degree anticlockwise of positive x-axis
        # For orientation, it is in radian. Positive when turning anticlockwise (left)
        self.state = np.zeros((3,1))
        self.prev_state = np.zeros((3,1))
        
        # Wheel parameters
        self.baseline = baseline  # The distance between the left and right wheels
        self.scale = scale  # The scaling factor converting M/s to m/s

        # Camera parameters
        self.camera_matrix = camera_matrix  # Matrix of the focal lengths and camera centre
        self.dist_coeffs = dist_coeffs  # Distortion coefficients
    
    def drive(self, drive_meas):
        # This is the "f" function in EKF
        # left_speed and right_speed are the speeds in M/s of the left and right wheels.
        # dt is the length of time to drive for

        # Compute the linear and angular velocity
        linear_velocity, angular_velocity = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)

        # Store prev state before assigning new state
        self.prev_state = self.state

        # Apply the velocities
        dt = drive_meas.dt
        if angular_velocity == 0:
            
            self.state[0] += np.cos(self.state[2]) * linear_velocity * dt
            self.state[1] += np.sin(self.state[2]) * linear_velocity * dt
            # print(f"state {self.state}")
        else:
            th = self.state[2]
            self.state[0] += linear_velocity / angular_velocity * (np.sin(th+dt*angular_velocity) - np.sin(th))
            self.state[1] += -linear_velocity / angular_velocity * (np.cos(th+dt*angular_velocity) - np.cos(th))
            self.state[2] += dt*angular_velocity
        
        # clamp angle from -pi to pi
        self.state[2] = self.state[2] % (2*np.pi)
        self.state[2] = self.state[2] - 2*np.pi if self.state[2] > np.pi else self.state[2]

        print(f"robot.py Motion Model State {self.state}")


    def convert_wheel_speeds(self, left_speed, right_speed):
        # Convert to m/s
        left_speed_m = left_speed * self.scale
        right_speed_m = right_speed * self.scale

        # Compute the linear and angular velocity
        linear_velocity = (left_speed_m + right_speed_m) / 2.0
        angular_velocity = (right_speed_m - left_speed_m) / self.baseline
        
        return linear_velocity, angular_velocity
    
    def measure(self, markers, idx_list):
        # This is the "h" function in EKF
        # Markers are 2d landmarks in a 2xn structure where there are n landmarks.
        # The index list tells the function which landmarks to measure in order.
        
        # Construct a 2x2 rotation matrix from the robot angle
        th = self.state[2]
        Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        robot_xy = self.state[0:2,:]

        measurements_hat = []
        for idx in idx_list:
            lm_state = markers[:,idx:idx+1]
            lm_position = Rot_theta.T @ (lm_state - robot_xy) # transpose of rotation matrix is the inverse
            
            # #TODO: changes for offset here
            # lm_position[1,:] -= 0.2

            measurements_hat.append(lm_position)
            # print("\tlmpos: ", " ".join(map(str, [item[0] for item in lm_position])))
            # print("\tstate: ", " ".join(map(str, [item[0] for item in lm_state])))
            print("\trbtstate: ", " ".join(map(str, [item[0] for item in self.state])))
            # print(f'lmpos: {lm_position}, state: {lm_state}, rbtstate: {self.state}\n')
            # print('state', lm_state)

        # Stack the measurements in a 2xn structure.
        measurements_hat = np.concatenate(measurements_hat, axis=1)
        return measurements_hat

    # Derivatives and Covariance
    # --------------------------

    def derivative_drive(self, drive_meas):
        # Compute the differential of drive w.r.t. the robot state
        DFx = np.zeros((3,3))
        DFx[0,0] = 1
        DFx[1,1] = 1
        DFx[2,2] = 1

        lin_vel, ang_vel = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)

        dt = drive_meas.dt
        th = self.state[2]
        
        # TODO: add your codes here to compute DFx using lin_vel, ang_vel, dt, and th
        # Derivative Matrix:
        # del f1 / del x     # del f1 / del y       # del f1 / del theta
        # del f2 / del x     # del f2 / del y       # del f2 / del theta
        # del f3 / del x     # del f3 / del y       # del f3 / del theta

        # There are 2 kinds of motion the robot can undergo
        # First motion is going straight, where angular velocity is 0
        if ang_vel == 0:

            # The equations of motions are:
            # x_k1 = x_k + v_k * cos(theta) * dt            f1
            # y_k1 = y_k + v_k * sin(theta) * dt            f2
            # theta_k1 = theta_k                            f3

            # Derivative w.r.t. x and y for straight line motion
            # Identity Matrix Already Given in 3 by 3
            DFx[0, 2] = lin_vel * -np.sin(th) * dt 
            DFx[1, 2] = lin_vel * np.cos(th) * dt

        # Second motion is if the robot is turning   
        else:
            
            # The equations of motions are:
            # R = lin_vel / ang_vel
            # theta_k1 = theta_k + ang_vel * dt                 f3
            # x_k1 = x_k + R * [-sin(theta_k) + sin(theta_k1)]  f1
            # y_k1 = y_k + R * [cos(theta_k) - cos(theta_k1)]   f2
            # Derivative w.r.t. x and y for circular arc motion

            # First define what theta_k1 is
            th_k1 = th + ang_vel * dt

            # Define R
            R = lin_vel / ang_vel

            # Derivative w.r.t. x and y for turning motion
            DFx[0, 2] = R * (- np.cos(th) + np.cos(th_k1))
            DFx[1, 2] = R * (- np.sin(th) + np.sin(th_k1))

        return DFx

    def derivative_measure(self, markers, idx_list):
        # Compute the derivative of the markers in the order given by idx_list w.r.t. robot and markers
        n = 2*len(idx_list)
        m = 3 + 2*markers.shape[1]

        DH = np.zeros((n,m))

        robot_xy = self.state[0:2,:]
        th = self.state[2]        
        Rot_theta = np.block([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        DRot_theta = np.block([[-np.sin(th), -np.cos(th)],[np.cos(th), -np.sin(th)]])

        for i in range(n//2):
            j = idx_list[i]
            # i identifies which measurement to differentiate.
            # j identifies the marker that i corresponds to.

            lmj_state = markers[:,j:j+1]
            # lmj_bff = Rot_theta.T @ (lmj_state - robot_xy)

            # robot xy DH
            DH[2*i:2*i+2,0:2] = - Rot_theta.T
            # robot theta DH
            DH[2*i:2*i+2, 2:3] = DRot_theta.T @ (lmj_state - robot_xy)
            # lm xy DH
            DH[2*i:2*i+2, 3+2*j:3+2*j+2] = Rot_theta.T

            # print(DH[i:i+2,:])

        return DH
    
    def covariance_drive(self, drive_meas):
        # Derivative of lin_vel, ang_vel w.r.t. left_speed, right_speed
        Jac1 = np.array([[self.scale/2, self.scale/2],
                [-self.scale/self.baseline, self.scale/self.baseline]])
        
        lin_vel, ang_vel = self.convert_wheel_speeds(drive_meas.left_speed, drive_meas.right_speed)
        th = self.state[2]
        dt = drive_meas.dt
        th2 = th + dt*ang_vel

        # Derivative of x,y,theta w.r.t. lin_vel, ang_vel
        Jac2 = np.zeros((3,2))
        
        # TODO: add your codes here to compute Jac2 using lin_vel, ang_vel, dt, th, and th2

        # Jacobian 2:
        # del x/ del v          # del x / del w
        # del y / del v         # del y / del w
        # del theta / del v     # del theta / del v

        # Note: k1 is k+1 | k is k

        # There are 2 kinds of motion the robot can undergo
        # First motion is going straight, where angular velocity is 0
        if ang_vel == 0:

            # The equations of motions are:
            # x_k1 = x_k + v_k * cos(theta) * dt
            # y_k1 = y_k + v_k * sin(theta) * dt
            # theta_k1 = theta_k

            # Calculating the Jacobian
            Jac2[0, 0] = np.cos(th) * dt    # del x/ del v 
            Jac2[1, 0] = np.sin(th) * dt    # del y / del v
        
        # Second motion is if the robot is turning
        else:

            # The equations of motions are:
            # R = lin_vel / ang_vel
            # theta_k1 = theta_k + ang_vel * dt
            # x_k1 = x_k + R * [-sin(theta_k) + sin(theta_k1)]
            # y_k1 = y_k + R * [cos(theta_k) - cos(theta_k1)]

            # Jac2[0, 0] = (1 / ang_vel) * (- np.sin(th) + np.sin(th2))                   # del x/ del v
            # Jac2[0, 1] = (-lin_vel / (ang_vel*ang_vel)) * (-np.sin(th) + np.sin(th2))   # del y/ del w

            # Jac2[1, 0] = (1 / ang_vel) * (np.cos(th) - np.cos(th2))                     # del y / del v
            # Jac2[1, 1] = (-lin_vel / (ang_vel*ang_vel)) * (np.cos(th) - np.cos(th2))   # del y/ del w

            # Jac2[2, 1] = dt                                                             # del theta/ del w


            Jac2[0, 0] = (1 / ang_vel) * (- np.sin(th) + np.sin(th2))                   # del x/ del v
            Jac2[0, 1] = (-lin_vel / (ang_vel*ang_vel)) * (-np.sin(th) + np.sin(th2)) + ((lin_vel / ang_vel) * (np.cos(th2) * dt))  # del y/ del w

            Jac2[1, 0] = (1 / ang_vel) * (np.cos(th) - np.cos(th2))                     # del y / del v
            Jac2[1, 1] = (-lin_vel / (ang_vel*ang_vel)) * (np.cos(th) - np.cos(th2)) + ((lin_vel / ang_vel) * np.sin(th2) * dt)  # del y/ del w

            Jac2[2, 1] = dt                                                             # del theta/ del w

        # Derivative of x,y,theta w.r.t. left_speed, right_speed
        Jac = Jac2 @ Jac1

        # Compute covariance
        cov = np.diag((drive_meas.left_cov, drive_meas.right_cov))
        cov = Jac @ cov @ Jac.T
        
        return cov