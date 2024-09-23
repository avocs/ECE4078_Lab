import os
import cv2
import time
import json
import torch
import numpy as np
from args import args
from torchvision import transforms
from res18_skip import Resnet18Skip

class ObjectDetector:

    # Initialising an instance of ObjectDetector
    def __init__(self, ckpt, use_gpu=False):
        self.args = args
        self.model = Resnet18Skip(args)

        # boolean flag to indicate whether to run inference on gpu
        if torch.cuda.torch.cuda.device_count() > 0 and use_gpu:
            self.use_gpu = True
            self.model = self.model.cuda()
        else:
            self.use_gpu = False

        # loads model weights, sets model to evaluation mode
        self.load_weights(ckpt)
        self.model = self.model.eval()

        # defines colour code for different obj classes
        self.colour_code = np.array([(220, 220, 220), (128, 0, 0), (155, 255, 70), (255, 85, 0), (255, 180, 0), (0, 128, 0)]) # color of background, redapple, greenapple, orange, mango, capsicum
        # opens file to store pred labels and corresponding robot pose
        self.pred_pose_fname = open(os.path.join('lab_output', 'pred.txt'), 'w')
        # counter for labels
        self.pred_count = 0

    # detects objects in a single image by running model, taking in a numpy array representing an image
    def detect_single_image(self, np_img):

        # converts the input image to a pytorch tensor
        torch_img = self.np_img2torch(np_img)
        tick = time.time()

        # performs forward pass through the model to obtain predictions
        with torch.no_grad():
            pred = self.model.forward(torch_img)
            if self.use_gpu:
                pred = torch.argmax(pred.squeeze(), dim=0).detach().cpu().numpy()
            else:
                pred = torch.argmax(pred.squeeze(), dim=0).detach().numpy()
        dt = time.time() - tick
        print(f'Inference Time {dt:.2f}s, approx {1/dt:.2f}fps', end="\r")

        # calls to visualise and creates a colour coded map
        # pred is a numpy array containing predicted object class labels for each pixel in the array
        colour_map = self.visualise_output(pred)
        return pred, colour_map
    
    # saves the predicted class labels and robot state into files
    def write_image(self, pred, state, lab_output_dir):      
        # Save the prediction label images
        # saves the predicted class labels into a png, this should show up as a grayscale image
        pred_fname = os.path.join(lab_output_dir, 'pred_' + str(self.pred_count) + '.png')
        self.pred_count += 1
        cv2.imwrite(pred_fname, pred)
        
        # for every prediction label image, save the state of the robot (its position when pressing "p")
        # this information is needed to estimate the pose of the object
        # creates a dictionary containing the robot state and its corresponding prediction png, returns the file name of the prediction png
        img_dict = {"pose": state, "predfname": pred_fname}
        self.pred_pose_fname.write(json.dumps(img_dict) + '\n')
        self.pred_pose_fname.flush()
        
        return f'pred_{self.pred_count-1}.png'

    # creates a color coded image based on the class labels
    # nn output, numpy array containing predicted class labels
    def visualise_output(self, nn_output):
        # creates empty arrays for red green blue channels
        r = np.zeros_like(nn_output).astype(np.uint8)
        g = np.zeros_like(nn_output).astype(np.uint8)
        b = np.zeros_like(nn_output).astype(np.uint8)

        # stacks the channels into a 3d array
        for class_idx in range(0, self.args.n_classes + 1):
            idx = nn_output == class_idx
            r[idx] = self.colour_code[class_idx, 0]
            g[idx] = self.colour_code[class_idx, 1]
            b[idx] = self.colour_code[class_idx, 2]
        colour_map = np.stack([r, g, b], axis=2)
        # resizes it to fit into the Detector subwindow in pygame
        colour_map = cv2.resize(colour_map, (320, 240), cv2.INTER_NEAREST)
        w, h = 10, 10
        pt = (10, 160)
        pad = 5
        labels = ['redapple', 'greenapple', 'orange', 'mango', 'capsicum']
        font = cv2.FONT_HERSHEY_SIMPLEX 
        # draws a legend with class name and corresponding colours on the colour map
        for i in range(1, self.args.n_classes + 1):
            c = self.colour_code[i]
            colour_map = cv2.rectangle(colour_map, pt, (pt[0]+w, pt[1]+h), (int(c[0]), int(c[1]), int(c[2])), thickness=-1)
            colour_map  = cv2.putText(colour_map, labels[i-1],
            (pt[0]+w+pad, pt[1]+h-1), font, 0.4, (0, 0, 0))
            pt = (pt[0], pt[1]+h+pad)
        return colour_map

    # load the pt file into the program
    def load_weights(self, ckpt_path):
        ckpt_exists = os.path.exists(ckpt_path)
        if ckpt_exists:
            ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(ckpt['weights'])
        else:
            print(f'checkpoint not found, weights are randomly initialised')
            
    @staticmethod
    def np_img2torch(np_img, use_gpu=False, _size=(192, 256)):
        preprocess = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(size=_size),
                                         # transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.05),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        img = preprocess(np_img)
        img = img.unsqueeze(0)
        if use_gpu: img = img.cuda()
        return img