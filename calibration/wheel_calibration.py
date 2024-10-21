# for computing the wheel calibration parameters
import os
import sys
import time
import numpy as np
sys.path.insert(0, "..")
from pibot import PibotControl
import math


def calibrateScale():
    # The scale parameter is used to convert the raw speed (assume unit is M/s) specified by you (0 to 1) to actual speed in m/s.
    # That is, actual speed = raw speed * scale
    # We can get actual speed measurements by driving the robot at a fixed raw speed for a known distance (eg 1 meter) and record the time taken for it.
    # We repeat the procedures multiple times (can use different raw speed), so that we can obtain the average value for a more robust measurement.

    # Feel free to change the range
    # wheel_speed_range = [[0.5, 0.5], [0.6, 0.6], [0.7, 0.7]] # Raw Speed
    wheel_speed_range = [[0.5, 0.5]] # Raw Speed
    # wheel_speed_range = [[0.6, 0.6], [0.7, 0.7], [0.8, 0.8]] # Raw Speed
    delta_times = []

    for wheel_speed in wheel_speed_range:
        print("Driving at {} M/s.".format(wheel_speed))
        
        # Repeat the test until the correct time is found.      
        while True:
            delta_time = float(input("Input the time to drive in seconds: "))
            start = time.time()
            elapsed = 0
            while elapsed < delta_time:
                pibot.set_velocity(wheel_speed)
                elapsed = time.time() - start
            pibot.set_velocity([0,0])
            uInput = input("Did the robot travel 1m? [y/N]")
            if uInput == 'y':
                delta_times.append(delta_time)
                print("Recording that the robot drove 1m in {:.2f} seconds at wheel speed {}.\n".format(delta_time, wheel_speed))
                break

    # Once finished driving, compute the scale parameter using wheel_speed and delta_time. Remember to take the average.
    # Helpful tips: the unit of the scale parameter is m/M.
    num = len(wheel_speed_range)
    scale = 0
    for delta_time, wheel_speed in zip(delta_times, wheel_speed_range):
        # pass # TODO: compute the scale parameter

        # Get the average raw speed values for both wheels, by taking sum of wheel speed divide by length of it
        avg_wheel_speed = sum(wheel_speed)/len(wheel_speed) # Raw Speed

        # Calculating the scale parameter using the formula: actual speed = raw speed * scale
        # Where actual speed = distance / time
        # So, to rearrange, distance / time = raw speed * scale
        # Scale = distance / (time * raw speed)
        scale_param = 1/(avg_wheel_speed*delta_time)

        # Add up all the scale parameters
        scale += scale_param
    
    # Get the average scale
    scale = scale/num
    
    print("The scale parameter is estimated as {:.6f} m/M.".format(scale))

    return scale

# NOTE: sandra just straight hardcoded the baseline to 12.45e-2 m. im about to lose it
def calibrateBaseline(scale):
    # The baseline parameter is the distance between the wheels.
    # This part is similar to the calibrateScale function, difference is that the robot is spinning 360 degree.
    # From the wheel_speed and delta_time, find out mathematically how to calculate the baseline.

    # Feel free to change the range / step
    wheel_speed_range = [[-0.4, 0.4], [-0.45, 0.45], [-0.5, 0.5]]
    delta_times = []

    for wheel_speed in wheel_speed_range:
        print("Driving at {} M/s.".format(wheel_speed))
        
        # Repeat the test until the correct time is found.      
        while True:
            delta_time = float(input("Input the time to drive in seconds: "))
            start = time.time()
            elapsed = 0
            while elapsed < delta_time:
                pibot.set_velocity(wheel_speed)
                elapsed = time.time() - start
            pibot.set_velocity([0,0])
            uInput = input("Did the robot spin 360 degree? [y/N]")
            if uInput == 'y':
                delta_times.append(delta_time)
                print("Recording that the robot spun 360 degree in {:.2f} seconds at wheel speed {}.\n".format(delta_time, wheel_speed))
                break

    # Once finished driving, compute the baseline parameter using wheel_speed and delta_time. Remember to take the average.
    # Helpful tips: the unit of the baseline parameter is m. Think about the circumference of a circle. You may also need the scale parameter here.
    num = len(wheel_speed_range)
    baseline = 0
    for delta_time, wheel_speed in zip(delta_times, wheel_speed_range):
        # pass # TODO: replace with your code to compute the baseline parameter using scale, wheel_speed, and delta_time

        # The formula for baseline is speed in (m/s) / angular velocity (rad/s)
        # First, convert the raw speed into actual speed, by multiplying it by the scale
        wheel_speed_ms = wheel_speed[1] * scale  # Convert raw speed to m/s

        # Then, theta = angular velocity * time
        # So, rearrange to get angular velocity, where theta = 2Pi since 360 degrees
        angular_velocity = 2 * math.pi / delta_time  # Full rotation in radians/time

        # Substitute the values into the baseline formula
        baseline_param = wheel_speed_ms / angular_velocity

        # Sum up all the individual baseline parameters
        baseline += baseline_param

    # Get the average baseline
    baseline = baseline / num

    print("The baseline parameter is estimated as {:.6f} m.".format(baseline))

    return baseline


def find_num_ticks_straight(): 

# Feel free to change the range / step
    # wheel_speed_range = [[-0.4, 0.4], [-0.45, 0.45], [-0.5, 0.5]]
    # wheel_speed_range = [0.4,0.45,0.5]
    # wheel_speed_range = [0.4,0.45,0.5]
    wheel_speed_range = [0.6]
    distance_range= [0.5,1,1.5,2]
    angle_range = [np.pi/4 ,np.pi/2, np.pi, -np.pi]
    delta_ticks = []
    wheel_rot_speed = 0.5

    for distance in distance_range:
        print("Driving to {} meter".format(distance))

        for wheel_speed in wheel_speed_range:
            print("Driving with wheel speed {}.".format(wheel_speed))
            lv,rv =[wheel_speed,wheel_speed]

            straight_speed= [lv,rv]
            print("Going stright speed {}. ".format(straight_speed))

            while True:
                delta_ticks = int(input("Input the ticks to drive to: "))
                
                initial_ticks = pibot.get_counter_values()
                ticks_travelled_left, ticks_travelled_right = 0,0
                pibot.set_velocity(straight_speed)

                while True: 
                    curr_ticks = pibot.get_counter_values()
                    ticks_travelled_left = curr_ticks[0] - initial_ticks[0]
                    ticks_travelled_right = curr_ticks[1] - initial_ticks[1]
                    print(f"tickcheck {ticks_travelled_left} {ticks_travelled_right} curr_ticks {curr_ticks}")

                    if ticks_travelled_left >= delta_ticks and ticks_travelled_right >= delta_ticks:
                        break
            
                pibot.set_velocity([0,0])

                uInput = input(f"Did the robot travel {distance}? [y/N]")
                if uInput == 'y':
                    print("Recording that the robot travel {} meters in {} ticks at wheel speed {}.\n".format(distance, delta_ticks, straight_speed))
                    break


def find_num_ticks_rotate():

    # Feel free to change the range / step
    wheel_speed_range = [[-0.4, 0.4], [-0.45, 0.45], [-0.5, 0.5]]
    angle_range = [-np.pi/4 ,-np.pi/2, np.pi, -np.pi]
    delta_ticks = []
    wheel_rot_speed= 0.45
    wheel_rot_speed_offset = 0.0


    for angle in angle_range:
        print("Driving to {} M/s.".format(angle))
        
        # -- direction of wheels, depending on sign
        if angle > 0: # turn left 
            lv, rv = [-wheel_rot_speed, wheel_rot_speed+wheel_rot_speed_offset]
        elif angle < 0: # 
            lv, rv = [wheel_rot_speed+wheel_rot_speed_offset, -wheel_rot_speed] 
        
        turn_speeds = [lv, rv]
        print("Turn speeds {}. ".format(turn_speeds))

        # Repeat the test until the correct ticks is found.      
        while True:
            delta_ticks = int(input("Input the ticks to drive to: "))
            
            initial_ticks = pibot.get_counter_values()
            ticks_travelled_left, ticks_travelled_right = 0,0
            pibot.set_velocity(turn_speeds)

            while True: 
                curr_ticks = pibot.get_counter_values()
                ticks_travelled_left = curr_ticks[0] - initial_ticks[0]
                ticks_travelled_right = curr_ticks[1] - initial_ticks[1]
                print(f"tickcheck {ticks_travelled_left} {ticks_travelled_right} curr_ticks {curr_ticks}")

                if ticks_travelled_left >= delta_ticks and ticks_travelled_right >= delta_ticks:
                    break
        
            pibot.set_velocity([0,0])

            uInput = input(f"Did the robot spin {angle}? [y/N]")
            if uInput == 'y':
                print("Recording that the robot spun {} rad in {} ticks at wheel speed {}.\n".format(angle, delta_ticks, turn_speeds))
                break




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=5000)
    args, _ = parser.parse_known_args()

    pibot = PibotControl(args.ip,args.port)
    # pibot.set_pid(use_pid=1, kp=0.005, ki=0, kd=0.0005)
    # pibot.set_pid(use_pid=1, kp=0.0001, ki=0.0, kd=0.001)
    pibot.set_pid(use_pid=1, kp=0.028, ki=0.000, kd=0.0000)

    # calibrate pibot scale and baseline
    dataDir = "{}/param/".format(os.getcwd()) 

    # print('Calibrating PiBot scale...\n')
    # scale = calibrateScale()
    # fileNameS = "{}scale.txt".format(dataDir)
    # np.savetxt(fileNameS, np.array([scale]), delimiter=',')

    # print('Calibrating PiBot baseline...\n')
    # baseline = calibrateBaseline(scale)
    # fileNameB = "{}baseline.txt".format(dataDir)
    # np.savetxt(fileNameB, np.array([baseline]), delimiter=',')

    print('Calibrating number of ticks...\n')
    find_num_ticks_rotate() 

    # find_num_ticks_straight() 

    print('Finished wheel calibration')