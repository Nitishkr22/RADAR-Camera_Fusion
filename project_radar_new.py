import numpy as np
import cv2
import csv
import os
from PIL import Image


def compute_world2img_projection(world_points, M, is_homogeneous=False):
  

    if not is_homogeneous:
        # convert to homogeneous coordinates
        points_h = np.vstack((world_points[:3,:], np.ones(world_points.shape[1])))

    h_points_i = M @ points_h  # matrix multiplication


    h_points_i[0, :] = h_points_i[0, :] / h_points_i[2, :]
    h_points_i[1, :] = h_points_i[1, :] / h_points_i[2, :]

    points_i = h_points_i[:2, :]

    return points_i


# Function to convert bounding box labels to pixel coordinates
def convert_labels_to_pixels(image_width, image_height, boxes):
    converted_boxes = []

    for box in boxes:
        class_name, x_center, y_center, width, height = box
        x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)
        
        # Convert center coordinates to pixel coordinates
        x1 = int((x_center - (width / 2)) * image_width)
        y1 = int((y_center - (height / 2)) * image_height)
        x2 = int((x_center + (width / 2)) * image_width)
        y2 = int((y_center + (height / 2)) * image_height)
        
        converted_boxes.append([class_name, x1, y1, x2, y2])

    return converted_boxes

# Function to get coordinates inside the bounding box
def get_coordinates_in_bounding_box(x_min, y_min, x_max, y_max):
    coordinates = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            coordinates.append((x, y))
    return coordinates

# Function to check if coordinates are within a bounding box
def check_coordinates(bounding_box_coordinates, x, y):
    for coord in bounding_box_coordinates:
        if coord[0] == x and coord[1] == y:
            return True
    return False

# Inputs go here
filename = '/home/radar/Documents/fusion_dataset/scene3_radar_synch/scene3/20/13-19-16.624532.csv'
img_path = '/home/radar/Documents/fusion_dataset/scene3_front/center_camera/13-19-16.728046.jpg'
text_file_path = "/home/radar/yolov5/runs/detect/exp3/labels/13-19-16.728046.txt"

data_array = np.genfromtxt(filename, delimiter=',')

height = 0.78
data_array[2, :] =  height

projection_matrix = np.load('NDLT_matrix.npy')

predictions = compute_world2img_projection(data_array, projection_matrix, is_homogeneous=False)

filtered_array = predictions.copy()

formatted_array = np.round(filtered_array.T)

image = cv2.imread(img_path)
desired_width = 640
desired_height = 400
resized_image = cv2.resize(image, (desired_width, desired_height))

coordinates = formatted_array

circle_radius = 3
colors = (0, 0, 255)

# Create a list to store points within bounding boxes
points_list = []

# Read image dimensions
image = Image.open(img_path)
image_width, image_height = image.size

# Read the bounding box coordinates from the text file
boxes = []
with open(text_file_path, 'r') as file:
    for line in file:
        box = line.strip().split()
        boxes.append(box)

# Convert bounding box labels to pixel coordinates
converted_boxes = convert_labels_to_pixels(image_width, image_height, boxes)

# Get coordinates inside bounding boxes
coordn = []
for box in converted_boxes:
    x_min, y_min, x_max, y_max = box[1], box[2], box[3], box[4]
    bounding_box_coordinates = get_coordinates_in_bounding_box(x_min, y_min, x_max, y_max)
    coordn.append(bounding_box_coordinates)

# Iterate through points and check if they are inside any bounding box
# for (x, y) in coordinates:
#     for i in range(len(coordn)):
#         is_present = check_coordinates(coordn[i], int(x), int(y))
#         if is_present:
#             points_list.append((int(x), int(y)))
#             cv2.circle(resized_image, (int(x), int(y)), circle_radius, colors, -1)
csv_file = "radar_data.csv"
for (x, y) in coordinates:
    for i in range(len(coordn)):
        is_present = check_coordinates(coordn[i], int(x), int(y))
        if is_present:
            class_name, x_min, y_min, x_max, y_max = converted_boxes[i]
            data_to_append = [int(x), int(y), RCS, SNR, class_name, x_min, y_min, x_max, y_max]
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(data_to_append)

# Display the resulting image
if not os.path.exists(csv_file):
    header = ['x', 'y', 'RCS', 'SNR', 'class_name', 'X_min', 'Y_min', 'X_max', 'Y_max']
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

cv2.imshow("Image with Radar Projection", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
