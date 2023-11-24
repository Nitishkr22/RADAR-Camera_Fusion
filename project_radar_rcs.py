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
filename = '/home/radar/Desktop/ayush/icra_projection/hm/output.csv'
data_array = np.genfromtxt(filename, delimiter=',')

height = 0.78
# num_points = data_array.shape[1]
data_array[2, :] =  height
projection_matrix = np.load('NDLT_matrix.npy')

predictions = compute_world2img_projection(data_array, projection_matrix, is_homogeneous=False)
filtered_array = predictions.copy()

formatted_array = np.round(filtered_array.T)

img_path = '/home/radar/Desktop/ayush/icra_projection/image_2023-08-29-17-19-37.294030.jpg'

coordinates = formatted_array

circle_radius = 3  # Radius of the circles to be drawn
colors = (0, 0, 255)  # (B, G, R) format for colors

image = cv2.imread(img_path)

desired_width = 640
desired_height = 400
resized_image = cv2.resize(image, (desired_width, desired_height))

# Create an empty list to store RCS values
rcs_values = data_array[4, :]

# Draw circles at the specified coordinates and display RCS values
for i, (x, y) in enumerate(coordinates):
    p = int(x)
    q = int(y)
    if (p < 640 and p > 0) and (q < 480 and q > 0):
        # Draw the circle
        cv2.circle(resized_image, (p, q), circle_radius, colors, -1)
        
        # Display RCS value next to the radar point
        rcs_text = f"{rcs_values[i]:.2f}"
        cv2.putText(resized_image, rcs_text, (p + 10, q), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors, 1)

# Display the resulting image
cv2.imshow("Image with Circles", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()