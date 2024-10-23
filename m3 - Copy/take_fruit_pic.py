import os
import sys
import cv2
import pygame
import numpy as np
sys.path.insert(0, "..")
from pibot import PibotControl


# im not sure what this class does
class calibration:
    def __init__(self,args):
        self.pibot_control = PibotControl(args.ip, args.port)
        self.img = np.zeros([480,640,3], dtype=np.uint8)
        self.command = {'motion':[0, 0], 'image': False}
        self.image_collected = 0
        self.finish = False

    # Collect images
    def image_collection(self, dataDir):
        if self.command['image']:
            image = self.pibot_control.get_image()

            # edit filename
            filename = os.path.join(dataDir, 'arenag' + str(self.image_collected) + '.png')
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)                                      # CONVERTS PIXEL FORMAT TO BGR (RASPI CAMERA COLOUR FORMAT)
            cv2.imwrite(filename, image)
            self.image_collected +=1
            print('Collected {} images for camera calibration.'.format(self.image_collected))               
        if self.image_collected == images_to_collect:
                self.finish = True
        self.command['image']= False

    def update_keyboard(self):
        for event in pygame.event.get():
            # Replace with your M1 codes if you want to drive around and take picture
            # Holding the Pibot and take pictures are also fine
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                # pass 
                self.command['wheel_speed'] = [0.8,0.8]
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                # pass 
                self.command['wheel_speed'] = [-0.8,-0.8]
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                # pass 
                self.command['wheel_speed'] = [0,1]
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                # pass 
                self.command['wheel_speed'] = [1,0]
            # stop
            elif event.type == pygame.KEYUP or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
                self.command['wheel_speed'] = [0, 0]
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                self.command['image'] = True

    def control(self):
        left_speed, right_speed = self.pibot_control.set_velocity(self.command['motion'])

    def take_pic(self):
        self.img = self.pibot_control.get_image()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=5000)
    args, _ = parser.parse_known_args()

    # SETUP DIRECTORY TO SAVE PICS IN
    currentDir = os.getcwd()        # GET CURRENT WORKING DIRECTORY
    dataDir = os.path.join(currentDir,'arena')   # ADD A SUBDIRECTORY
    if not os.path.exists(dataDir):                 # CREATES THE FOLDER IF NOT ALREADY CREATED
        os.makedirs(dataDir)
    
    images_to_collect = 10 # feel free to change this

    calib = calibration(args)

    # THIS DEFINES THE IMAGE DIMENSIONS
    width, height = 640, 480
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Calibration')
    canvas.fill((0, 0, 0))
    pygame.display.update()
    
    # collect data
    print('Collecting {} images for dataset.'.format(images_to_collect))
    print('Press ENTER to capture image.')
    while not calib.finish:
        calib.update_keyboard()
        calib.control()
        calib.take_pic()
        calib.image_collection(dataDir)
        img_surface = pygame.surfarray.make_surface(calib.img)
        img_surface = pygame.transform.flip(img_surface, True, False)
        img_surface = pygame.transform.rotozoom(img_surface, 90, 1)
        canvas.blit(img_surface, (0, 0))
        pygame.display.update()
    print('Finished image collection.\n')
    print('Images Saved at: \n',dataDir)

