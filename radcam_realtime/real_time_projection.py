import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Float32MultiArray, String
from conti_radar.msg import radar_img
from datetime import datetime
import numpy as np
import csv
import os
from sklearn.cluster import DBSCAN

# Initialize ROS node
rospy.init_node('listener', anonymous=True)

# Global variables to store data
a = []  # List to store radar data timestamps
b = []  # List to store camera data timestamps
i = 0   # Flag to indicate the presence of camera data
current_x = None
current_y = None  # Variable to store the current radar data
range = None
RCS = None
SNR = None
current_bounding_boxes = None  # Variable to store the current bounding boxes

# Load camera projection matrix
ndlt = np.load('ndlt_webcam.npy')

# Radius of the circles to be drawn
circle_radius = 4

# Color in (B, G, R) format for drawing circles
colors = (0, 0, 255)

###############3
# Function to create or append data to the CSV file
def append_to_csv(file_name, data):
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

# # Define the file name
timestamp1 = datetime.now().strftime('%H:%M:%S.%f')
csv_file = "rec_data/"+timestamp1+".csv"

# Write the header if the file doesn't exist
if not os.path.exists(csv_file):
    header = ['timestamps','range','X', 'Y','RCS','SNR','class_name','X_min','Y_min','X_max','Y_max']
    append_to_csv(csv_file, header)

# Example data to append
# data_to_append = [
#     [100, 'A'],
#     [200, 'B'],
#     [300, 'C'],
# ]

# # Append data to the CSV file
# for data in data_to_append:
#     append_to_csv(csv_file, data)

# print("Data appended successfully.")

#########33


def get_coordinates_in_bounding_box(x_min, y_min, x_max, y_max):
    coordinates = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            coordinates.append((x, y))
    return coordinates

def compute_world2img_projection(world_points, M, is_homogeneous=False):
    if not is_homogeneous:
        # convert to homogeneous coordinates
        points_h = np.vstack((world_points[:3,:], np.ones(world_points.shape[1])))

    h_points_i = M @ points_h  # matrix multiplication


    h_points_i[0, :] = h_points_i[0, :] / h_points_i[2, :]
    h_points_i[1, :] = h_points_i[1, :] / h_points_i[2, :]

    points_i = h_points_i[:2, :]

    return points_i

def closest_timestamp(cv_image=None):
    global a, b, i, current_x, current_y, range, RCS, SNR, current_bounding_boxes 
    if i == 1 and len(a) >= 2 and len(b) > 0:
        radar_timestamps = [datetime.strptime(ts, '%H:%M:%S.%f') for ts in a[-2:]]
        camera_timestamp = datetime.strptime(b[-1], '%H:%M:%S.%f')

        closest_ts = min(radar_timestamps, key=lambda ts: abs(ts - camera_timestamp))
        closest_ts_str = closest_ts.strftime('%H:%M:%S.%f')
        print("Closest timestamp of {} is {}".format(b[-1], closest_ts_str))

        # Use the current_radar_data and current_bounding_boxes directly
        x = np.asarray(current_x)
        y = np.asarray(current_y)
        rangep = np.asarray(range)
        RCSp = np.asarray(RCS)
        SNRp = np.asarray(SNR)
        # print("sssssssssssssssssssss ",rangep[10])

        z = np.ones(x.shape[0])*0.78
        xandy = np.vstack((x, y, z))
        # print("bbbbbb ",xandy.shape)

        ndlt = np.load('ndlt_webcam.npy')
        circle_radius = 4  # Radius of the circles to be draw
        colors = (0, 0, 255)  # (B, G, R) format for colors
        predictions = compute_world2img_projection(xandy, ndlt)
        # print("sssssssss ",predictions.shape)
        predictions = np.round(predictions.T)
        
        with open(csv_file, mode='a', newline='') as file:

            if cv_image is not None:
                for class_name, x_min, y_min, x_max, y_max in current_bounding_boxes:
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    # ranclus = []
                    for (x, y) in predictions:
                        
                        p = int(x)
                        q = int(y)
                        if x_min <= p <= x_max and y_min <= q <= y_max:
                            if 0 <= q < cv_image.shape[1] and 0 <= p < cv_image.shape[0]:
                                indices = np.where(predictions==x)[0]
                                # print("qqqqqqqqqqqqqq ",indices[0])
                                # print("xxxx: ",predictions[indices[0],0])
                                # print("xxxxxxxxx: ",xandy[0,indices[0]])
                                # print("yyyyyyyyyyy: ",xandy[1,indices[0]])
                                # print("yyyyyyyyyyy: ",rangep[indices[0]])
                                # print("aaaaaaaaaaaAA ",closest_ts_str)
                                # ranclus.append(rangep[indices[0]])
                                

                        # dbscan = DBSCAN(eps=0.5,min_samples=2)
                        # cluster_labels = dbscan.fit_predict([[r] for r in ranclus])
                        # for labels in set(cluster_labels):
                        #     if (labels != -1): 
                                data_to_append =[closest_ts_str,xandy[0,indices[0]],rangep[indices[0]], xandy[1,indices[0]],RCSp[indices[0]],SNRp[indices[0]],class_name,x_min,y_min,x_max,y_max]
                                append_to_csv(csv_file, data_to_append)
                                cv2.circle(cv_image, (p, q), circle_radius, colors, -1)



                    cv2.imshow("Received image", cv_image)
                    cv2.waitKey(1)

    i = 1   # Indicate that camera data has been found


def callback(data):
    timestamp = datetime.now().strftime('%H:%M:%S.%f')
    print(timestamp)
    a.append(timestamp)
    global current_x, current_y, range, RCS, SNR
    current_x , current_y, range, RCS, SNR = data.x_dist, data.y_dist, data.range, data.RCS, data.SNR
    closest_timestamp()

def time_callback(msg):
    global b, i
    rospy.loginfo("Received Timestamp for Image %s", msg.data)
    b.append(msg.data)
    closest_timestamp()

def bbox_callback(msg):
    global current_bounding_boxes
    list_of_lists = unflatten_data(msg.data)
    rospy.loginfo("Received data: %s", list_of_lists)
    current_bounding_boxes = list_of_lists

def image_callback(msg):
    global cv_image
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    closest_timestamp(cv_image)

def unflatten_data(flattened_data):
    list_of_lists = []
    sublist = []
    for item in flattened_data:
        if item == -1:
            list_of_lists.append(sublist)
            sublist = []
        else:
            sublist.append(item)
    return list_of_lists



rospy.Subscriber('/radar_img', radar_img, callback)
rospy.Subscriber("/object_topic", Float32MultiArray, bbox_callback)
rospy.Subscriber("/time_topic", String, time_callback)
rospy.Subscriber("/image_topic", Image, image_callback)

rospy.spin()
