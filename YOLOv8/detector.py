import os
import cv2
import torch
import numpy as np
import json
from copy import deepcopy
from ultralytics import YOLO 
from ultralytics.utils import ops

class ObjectDetector:
    def __init__(self, model_path):
        self.model= YOLO(model_path)
        self.model.conf= 0.8
        #bounding box colours, RGB format
        self.class_colour = {
            '0': (220,220,220),
            '1': (128,0,0),
            '2': (155,255,70),
            '3': (255,85,0),
            '4': (255,180,0),
            '5': (0,128,0)
        }
        # counter for labels
        self.pred_count=0
        # opens file to store pred labels and corresponding robot pose
        self.pred_pose_fname = open(os.path.join('lab_output', 'pred.txt'),'w')
        # opens file to store bbox info
        self.bbox_fname = open(os.path.join('lab_output', 'bbox.txt'), 'w')

    # in this version of single image detection, instead of generating a colour map, 
    # directly draws the bounding box corresponding to the labels onto the image for display on the detector
    def detect_single_image(self, img):
        # calls model to inference and return the bounding boxes coordinates
        boundingboxes= self.get_bounding_box(img)

        # index the bounding box coordinates for drawing
        for boundingbox in boundingboxes:
            xyxy = ops.xywh2xyxy(boundingbox[1])
            x1 = int(xyxy[0])
            y1 = int(xyxy[1])
            x2 = int(xyxy[2])
            y2 = int(xyxy[3])

            # draw bounding box
            output_img = cv2.rectangle(img, (x1, y1), (x2, y2), self.class_colour[boundingbox[0]], thickness=2)

            # labelling the bounding boxes
            output_img = cv2.putText(output_img, boundingbox[0], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  self.class_colour[boundingbox[0]], 2)
            
        # return boundingboxes, output_img
        return boundingboxes, img
    
    def get_bounding_box(self,cv_img):

        # TODO this converts the image obtained from camera to rgb channels before inference..?
        newimg = cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)

        # ultralytics yolo models return a list of Results objects
        #    -- incuding bounding boxes, masks, keypoints, probs objects, oriented boxes objects
        # TODO changed imgsz argument from 320 to 640 to match training data img size
        predictions = self.model.predict(newimg, imgsz=640, verbose=False)

        # append the information of the bounding boxes and append to a list
        bounding_boxes = []
        # for each object it detects
        for prediction in predictions:
            # get the possible bounding boxes
                # - if the model is very confident, then it may just return one box, otherwise it could return multiple for the same obj
            boxes = prediction.boxes
            for box in boxes:
                # obtain the box coordinates as a numpy array, and the class name of the object
                box_cord = box.xywh[0]
                box_label = box.cls
                bounding_boxes.append([prediction.names[int(box_label)], np.asarray(box_cord)])

        # returns all the bounding boxes coordinates for each fruit, labelled
        return bounding_boxes
    
    def write_image(self, bounding_box_info, state, pred_output_dir, bbox_fname=None, pred_pose_fname=None):      

        # save the prediction label images to png called pred_num.png
        # Generate a image name for saving the raw images to
        pred_fname = os.path.join(pred_output_dir, 'pred_' + str(self.pred_count) + '.png')
        self.pred_count += 1

        if bbox_fname is None:
            bbox_fname = self.bbox_fname
        else: 
            bbox_fname = bbox_fname

        if pred_pose_fname is None:
            pred_pose_fname = self.pred_pose_fname
        else:
            pred_pose_fname = pred_fname
        # -- for every prediction label image (predfname), save the state of the robot (pose) to pred.txt (its position when pressing "p")
        # at the same time, saves the bbox obtained from each image to bbox.txt
        # this information is needed to estimate the pose of the object
        # write each bbox info as a dictionary into bbox.txt
        for bounding_box in bounding_box_info: 
            label = bounding_box[0]
            bbox = bounding_box[1].tolist()
            bbox_dict = {"label": label, 'bbox': bbox, "predfname": pred_fname}
            bbox_fname.write(json.dumps(bbox_dict) + '\n')
            bbox_fname.flush()
            
        # write each robot pose info as a dictionary into pred.txt
        img_dict = {"pose": state, "predfname": pred_fname}
        pred_pose_fname.write(json.dumps(img_dict) + '\n')
        pred_pose_fname.flush()
        
        # return the png name for operate to assist in saving the captured image
        return f'pred_{self.pred_count-1}.png'
    

