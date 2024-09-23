import cv2
import numpy as np
import os
import random
from PIL import Image

# Directory paths
# FRUITS_DIR = 'fruits/'
# BACKGROUND_DIR = 'backgrounds/'
# OUTPUT_DIR_IMAGES = 'cv/dataset/images/'
# OUTPUT_DIR_LABELS = 'cv/dataset/labels/'

# Fruit labels
FRUIT_LABELS = {
    'redapple': 1,
    'greenapple': 2,
    'orange': 3,
    'mango': 4,
    'capsicum': 5
}

# Create directories if they don't exist
# os.makedirs(OUTPUT_DIR_IMAGES, exist_ok=True)
# os.makedirs(OUTPUT_DIR_LABELS, exist_ok=True)

def remove_background(image_path):
    """Remove the background from an image with a uniform background color."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #     # Adaptive thresholding
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # mask_thresh = 255 - thresh

    # get image in hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)         

    # Define range for background color (e.g., white or a specific color)
    # h = [0, 179], s = [0, 255], v = [0, 255]
    lower_bound = np.array([0, 0, 60])
    upper_bound = np.array([179, 85, 255])

    # Create mask to detect the background
    mask_hsvbg = cv2.inRange(hsv, lower_bound, upper_bound)
    # inverse is the fruit
    mask_hsv = cv2.bitwise_not(mask_hsvbg)
    # mask_hsv = 255 - mask_hsvbg

    # mask = mask_thresh
    # mask = cv2.bitwise_and(mask_thresh, mask_hsv)
    init_mask = mask_hsv
    kernel = np.ones((3,3), np.uint8)
    kernel2 = np.ones((70,70), np.uint8)
    kernel3 = np.ones((2,2), np.uint8)
    kernel4 = np.ones((10,10), np.uint8)
    
    mask = cv2.morphologyEx(init_mask, cv2.MORPH_OPEN, kernel)  # Open to remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)  # Close to fill gaps
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close to fill gaps
    mask = cv2.dilate(init_mask, kernel4, iterations=2)

    # Extract the fruit from the image
    fruit = cv2.bitwise_and(image, image, mask=mask)


    cv2.imshow('init', init_mask)
    cv2.imshow('mask', mask)
    cv2.imshow('result', fruit)
 
    # close image
    cv2.waitKey(0)
    cv2.destroyAllWindows();

    return fruit, mask

# def superimpose_fruit(background, fruit, mask, label_value):
#     """Randomly place a fruit on the background image."""
#     bg_h, bg_w, _ = background.shape
#     fruit_h, fruit_w, _ = fruit.shape

#     # Randomize scale
#     scale_factor = random.uniform(0.5, 1.5)
#     fruit = cv2.resize(fruit, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
#     mask = cv2.resize(mask, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

#     fruit_h, fruit_w, _ = fruit.shape

#     # Randomize position
#     max_x = bg_w - fruit_w
#     max_y = bg_h - fruit_h
#     x_offset = random.randint(0, max_x)
#     y_offset = random.randint(0, max_y)

#     # Region of Interest (ROI) where the fruit will be placed
#     roi = background[y_offset:y_offset+fruit_h, x_offset:x_offset+fruit_w]

#     # Create a mask for the fruit and its inverse
#     mask_inv = cv2.bitwise_not(mask)

#     # Black-out the area of the fruit in the ROI
#     bg_with_fruit = cv2.bitwise_and(roi, roi, mask=mask_inv)

#     # Add the fruit to the background
#     combined = cv2.add(bg_with_fruit, fruit)
#     background[y_offset:y_offset+fruit_h, x_offset:x_offset+fruit_w] = combined

#     # Create a label mask with the fruit area marked
#     label_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
#     label_mask[y_offset:y_offset+fruit_h, x_offset:x_offset+fruit_w] = label_value * (mask > 0).astype(np.uint8)

#     return background, label_mask

# def generate_dataset(num_images=1000):
#     """Generate a dataset of images with random fruit placements."""
#     fruit_files = [f for f in os.listdir(FRUITS_DIR) if f.endswith('.png') or f.endswith('.jpg')]
#     background_files = [f for f in os.listdir(BACKGROUND_DIR) if f.endswith('.png') or f.endswith('.jpg')]

#     for i in range(num_images):
#         # Choose a random background image
#         bg_file = random.choice(background_files)
#         background = cv2.imread(os.path.join(BACKGROUND_DIR, bg_file))

#         # Initialize the label image
#         label_image = np.zeros((background.shape[0], background.shape[1]), dtype=np.uint8)

#         # Number of fruits to place on the background
#         num_fruits = random.randint(1, 5)

#         for _ in range(num_fruits):
#             # Choose a random fruit image
#             fruit_file = random.choice(fruit_files)
#             fruit_name = fruit_file.split('.')[0]  # Extract the fruit name
#             fruit_label = FRUIT_LABELS.get(fruit_name, 0)

#             # Remove the background from the fruit image
#             fruit_path = os.path.join(FRUITS_DIR, fruit_file)
#             fruit, mask = remove_background(fruit_path)

#             # Superimpose the fruit on the background
#             background, fruit_label_mask = superimpose_fruit(background, fruit, mask, fruit_label)

#             # Update the label image
#             label_image = cv2.bitwise_or(label_image, fruit_label_mask)

#         # Save the generated image and label
#         image_filename = os.path.join(OUTPUT_DIR_IMAGES, f'image_{i}.png')
#         label_filename = os.path.join(OUTPUT_DIR_LABELS, f'image_{i}_label.png')
#         cv2.imwrite(image_filename, background)
#         cv2.imwrite(label_filename, label_image)

#         print(f"Generated image and label {i + 1}/{num_images}")

# # Generate the dataset
# generate_dataset(1000)

if __name__ == '__main__':

    
    curr_dir = os.getcwd()
    img_path = os.path.join(curr_dir, 'm3/datatest')
    dir_path = os.path.join(curr_dir, 'm3/testouts')

    os.makedirs(dir_path, exist_ok=True)

    num = 0
    for image in os.listdir(img_path):
        if image.startswith('fruit') and image.endswith('png'):
            curr_filepath = os.path.join(img_path, image)
            fruit_extracted, mask_inv = remove_background(curr_filepath)
            new_file_name = f"extfruit_{num}.png"
            new_file_path = os.path.join(dir_path, new_file_name)
            # fruit_extracted_pic = fruit_extracted.astype(np.uint8)
            cv2.imwrite(new_file_path, fruit_extracted)
            num+=1 
    print('Images saved')


