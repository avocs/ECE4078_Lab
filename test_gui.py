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

import matplotlib.pyplot as plt
import math



# Helper function for waypoint selection 
def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))

# Function for handling user interaction and waypoint selection on the GUI
def enter_waypoint_on_click(event, fig, px, py, idx, waypoint_callback):
    global waypoint
    space = np.array([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])

    x = round_nearest(event.xdata, 0.2)
    y = round_nearest(event.ydata, 0.2)

    if event.button == 1:
        # Left click: add point
        px.append(x)
        py.append(y)
        idx += 1
        waypoint = [x,y]

        # update plot with clicked point 
        plt.clf()
        plt.plot(0, 0, 'rx')

        for i in range(len(px)):
            plt.scatter(px[i], py[i], color='C0')
            plt.text(px[i] + 0.05, py[i] + 0.05, i + 1, color='C0', size=12)

        plt.xlabel("X");
        plt.ylabel("Y")
        plt.xticks(space); plt.yticks(space)
        plt.grid()
        fig.canvas.draw()

        # callback function? 
        waypoint_callback(waypoint)
    elif event.button == 3:
        # Right click: delete last point
        del px[-1]
        del py[-1]
        idx -= 1
        waypoint = None


    # Clear and redraw the plot with updated waypoints
    plt.clf()
    plt.plot(0, 0, 'rx')

    for i in range(len(px)):
        plt.scatter(px[i], py[i], color='C0')
        plt.text(px[i] + 0.05, py[i] + 0.05, i + 1, color='C0', size=12)

    plt.xlabel("X");
    plt.ylabel("Y")
    plt.xticks(space); plt.yticks(space)
    plt.grid()
    fig.canvas.draw()

    
# Function to set up the initial GUI plot
def gui_setup():
    space = np.array([-1.6, -1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2, 1.6])

    fig = plt.figure()
    plt.plot(0, 0, 'rx')
    plt.xlabel("X");
    plt.ylabel("Y")
    plt.xticks(space); plt.yticks(space)
    plt.grid()
    return fig




# Function to launch the GUI and handle interaction
def generate_gui(fig, px, py, idx, waypoint_callback):
    print("Specify waypoint on grid for robot to drive to")
    fig.canvas.mpl_connect('button_press_event', lambda event: enter_waypoint_on_click(event, fig, px, py, idx, waypoint_callback))
    plt.show()


def waypoint_callback(waypoint):
    print("Waypoint selected:", waypoint)
    
    

if __name__ == '__main__':
    # import argparse

    # parser = argparse.ArgumentParser()
    # parser.add_argument("outputfilename", type=str, help='output filename')
    # args = parser.parse_args()

    
    # Variables, p will contains clicked points, idx contains current point that is being selected
    px, py = [], []
    idx = 0

    
    fig = gui_setup()
    generate_gui(fig, px, py, idx, waypoint_callback)


    

    
