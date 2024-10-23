import os
import cv2
import numpy as np

def generate_training_data(base_dir, arena_photo_start, arena_photo_end, fruit_photo_start, fruit_photo_end, no_generated):
    """
    Generates synthetic training data for object detection by randomly placing fruit images onto background images.

    Args:
        base_dir (str): The base directory for the dataset.
        arena_photo_start (int): The starting index for arena photos.
        arena_photo_end (int): The ending index for arena photos.
        fruit_photo_start (int): The starting index for fruit photos.
        fruit_photo_end (int): The ending index for fruit photos.
        no_generated (int): The number of synthetic images to generate per fruit-arena pair.
    """


    # set up target directories
    images_dir = os.path.join(base_dir, 'images')
    labels_dir = os.path.join(base_dir, 'labels')
    bg_arena = os.path.join(base_dir, 'arena')
    fg_fruits = os.path.join(base_dir, 'fruits_ext')

    # create the directories if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # time-seeding for Actually Random rng
    np.random.seed(None)  

    # photo numbering
    output_photo_number = 0
    for i in range(arena_photo_start, arena_photo_end + 1):
        for j in range(fruit_photo_start, fruit_photo_end + 1):

            # Load foreground and background images
            foreground_name = os.path.join(fg_fruits, f"fruit{j}.png")
            background_name = os.path.join(bg_arena, f"arena_{i}.png")

            background_arena = cv2.imread(background_name)
            background_arena = cv2.resize(background_arena, (background_arena.shape[1] * 2, background_arena.shape[0] * 2))

            label = (j // 20) + 1  # Determine label(based on fruit index)

            for k in range(no_generated):
                foreground_fruit = cv2.imread(foreground_name)
                # Randomly scale the foreground image

                scale_factor = np.random.rand() * 0.75 + 0.25
                foreground_fruit = cv2.resize(foreground_fruit, (int(foreground_fruit.shape[1] * scale_factor), int(foreground_fruit.shape[0] * scale_factor)))

                # Create a mask for the foreground image
                mask = cv2.cvtColor(foreground_fruit, cv2.COLOR_BGR2GRAY) > 5
                mask = np.stack([mask] * 3, axis=-1)

                # Randomly place the foreground image on the background
                offset_x = np.random.randint(0, background_arena.shape[1] - foreground_fruit.shape[1] + 1)
                offset_y = np.random.randint(0, background_arena.shape[0] - foreground_fruit.shape[0] + 1)

                # Create a region of interest (ROI) in the background image
                roi = background_arena[offset_y:offset_y+foreground_fruit.shape[0], offset_x:offset_x+foreground_fruit.shape[1], :]
                # Overlay the foreground image onto the ROI using the mask
                roi[mask] = foreground_fruit[mask]
                # Replace the ROI in the background image with the modified ROI
                background_arena[offset_y:offset_y+foreground_fruit.shape[0], offset_x:offset_x+foreground_fruit.shape[1], :] = roi

                # Calculate bounding box coordinates and dimensions
                mask_channel = mask[:, :, 0]
                indices = np.where(mask_channel)

                if len(indices[0]) >= 2:
                    col1, col2 = indices[1][[0, -1]]
                    mask_channel = mask_channel.T
                    row1, row2 = indices[0][[0, -1]]
                    width = (col2 - col1) / background_arena.shape[1]
                    height = (row2 - row1) / background_arena.shape[0]
                    x_val = (offset_x + col1 + (col2 - col1) / 2) / background_arena.shape[1]
                    y_val = (offset_y + row1 + (row2 - row1) / 2) / background_arena.shape[0]

                else:
                    # Handle cases where there are not enough points
                    width = 0
                    height = 0
                    x_val = 0
                    y_val = 0

                image_filename = f"image_{output_photo_number}.png"
                label_filename = f"image_{output_photo_number}.txt"
                cv2.imwrite(os.path.join(images_dir, image_filename), background_arena)

                with open(os.path.join(labels_dir, label_filename), 'w') as f:
                    f.write(f"{label} {x_val} {y_val} {width} {height}")

                output_photo_number += 1

    print(f"Ended at {output_photo_number}")

# Example usage:
base_dir = os.path.join(os.getcwd(), 'm3')
#base_dir = 'dataset'
generate_training_data(base_dir, 0, 29, 0, 99, 1)