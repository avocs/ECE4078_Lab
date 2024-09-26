# access each wheel and the camera onboard of pibot
import numpy as np
import requests
import cv2 

class PibotControl:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.wheel_vel = [0, 0]
        
    def set_pid(self, use_pid, kp, ki, kd):
        requests.get(f"http://{self.ip}:{self.port}/pid?use_pid="+str(use_pid)+"&kp="+str(kp)+"&ki="+str(ki)+"&kd="+str(kd))

    # Change the robot speed here
    # The value should be between -1 and 1.
    # Note that this is just a number specifying how fast the robot should go, not the actual speed in m/s
    def set_velocity(self, wheel_speed): 
        left_speed = max(min(wheel_speed[0], 1), -1) 
        right_speed = max(min(wheel_speed[1], 1), -1)
        self.wheel_vel = [left_speed, right_speed]
        requests.get(f"http://{self.ip}:{self.port}/move?left_speed="+str(left_speed)+"&right_speed="+str(right_speed))
        return left_speed, right_speed
    
    # TODO dra mod 
    def get_counter_values(self):
            response = requests.get(f"http://{self.ip}:{self.port}/encoders")  # Replace with your Pi's IP
            if response.status_code == 200:
                data = response.json()
                # print(f"Received data: {data}")  # Debugging print statement
                self.left_encoder = data["left_encoder"]
                self.right_encoder = data["right_encoder"]
                return self.left_encoder, self.right_encoder
            else:
                print("Failed to get encoder values.")
                return None, None
        
    def get_image(self):
        try:
            r = requests.get(f"http://{self.ip}:{self.port}/image")
            img = cv2.imdecode(np.frombuffer(r.content,np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print("Image retrieval timed out.")
            img = np.zeros((480,640,3), dtype=np.uint8)
        return img
        
        
# This class stores the wheel velocities of the robot, to be used in the EKF.
class Drive:
    def __init__(self, left_speed, right_speed, dt, left_cov = 1, right_cov = 1):
        self.left_speed = left_speed
        self.right_speed = right_speed
        self.dt = dt
        self.left_cov = left_cov
        self.right_cov = right_cov