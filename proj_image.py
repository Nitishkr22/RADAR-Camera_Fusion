import numpy as np
import cv2

# Example pixel coordinates (x, y) and corresponding pixel values
coordinates = np.array([[577,300],[578,302]])
circle_radius = 4  # Radius of the circles to be drawn
colors = [(255, 0, 0), (0, 0, 255)]  # (B, G, R) format for colors

# Create an empty image with the desired dimensions
# image_width = 640
# image_height = 480
# image = np.zeros((image_height, image_width), dtype=np.uint8)  # Creating an empty grayscale image


# Read the image from file
image_path = "/home/radar/tunnel/tunnel/center_20/image_2023-07-22-12-10-27.719312.jpg"  # Replace this with the actual path to your image file
image = cv2.imread(image_path)

desired_width = 1080
desired_height = 720
resized_image = cv2.resize(image, (desired_width, desired_height))
# Draw circles at the specified coordinates
for i, (x, y) in enumerate(coordinates):
    color = colors[i]
    cv2.circle(resized_image, (x, y), circle_radius, color, -1)

# Display the resulting image
cv2.imshow("Image with Circles", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
