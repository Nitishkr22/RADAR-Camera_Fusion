import numpy as np
import cv2



def compute_world2img_projection(world_points, M, is_homogeneous=False):
  

    if not is_homogeneous:
        # convert to homogeneous coordinates
        points_h = np.vstack((world_points[:3,:], np.ones(world_points.shape[1])))

    h_points_i = M @ points_h  # matrix multiplication


    h_points_i[0, :] = h_points_i[0, :] / h_points_i[2, :]
    h_points_i[1, :] = h_points_i[1, :] / h_points_i[2, :]

    points_i = h_points_i[:2, :]

    return points_i


# Read CSV file into a NumPy array
filename = '/home/radar/Desktop/ayush/icra_possible_plots/13-12-30.474777_obj.csv'
data_array = np.genfromtxt(filename, delimiter=',')

# Display the NumPy array
# print(type(data_array))

height = 0.78
# num_points = data_array.shape[1]
data_array[2, :] =  height
# print(data_array[:3,:])
# data_array = np.load('naam.npy')
projection_matrix = np.load('Radar_camera_Projection_matrix.npy')

predictions = compute_world2img_projection(data_array, projection_matrix, is_homogeneous=False)
# predictions = np.load('naam.npy')
# print(predictions)

filtered_array = predictions.copy()


# filtered_array[0, filtered_array[0] > 1080] = 0.0
# filtered_array[1, filtered_array[1] > 720] = 0.0

# filtered_array[0, filtered_array[0] < 0] = 0.0
# filtered_array[1, filtered_array[1] < 0] = 0.0


formatted_array = np.round(filtered_array.T)

# print((formatted_array))
# print(formatted_array)

# rounded_array = np.round(filtered_array)

img_path = '/home/radar/Desktop/ayush/icra_possible_plots/image_2023-09-02-13-12-30.445924.jpg'

# coordinates = np.array([[577,300],[578,302]])
coordinates = formatted_array

circle_radius = 3  # Radius of the circles to be drawn
colors = (0, 0, 255)  # (B, G, R) format for colors

# Create an empty image with the desired dimensions
# image_width = 640
# image_height = 480
# image = np.zeros((image_height, image_width), dtype=np.uint8)  # Creating an empty grayscale image


# Read the image from file
# image_path = "/home/radar/tunnel/tunnel/center_20/image_2023-07-22-12-10-27.719312.jpg"  # Replace this with the actual path to your image file
image = cv2.imread(img_path)

desired_width = 640
desired_height = 440

resized_image = cv2.resize(image, (desired_width, desired_height))
# Draw circles at the specified coordinates
points_list = []
for (x, y) in (coordinates):
    # print(int(x),int(y))
    p = int(x)
    q = int(y)
    if (p<640 and p>0):
        if(q<480 and q>0):
            points_list.append((p, q))
            cv2.circle(resized_image, (p,q), circle_radius, colors, -1)

with open("points.txt", "w") as file:
    for point in points_list:
        file.write(f"({point[0]}, {point[1]})\n")

# Display the resulting image
cv2.imshow("Image with Circles", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()