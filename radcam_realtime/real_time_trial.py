import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Float32MultiArray, String
from conti_radar.msg import radar_img
from datetime import datetime
import numpy as np
import os
import csv

rospy.init_node('listener', anonymous=True)

a = []  # List to store radar data timestamps
b = []  # List to store camera data timestamps
i = 0   # Flag to indicate the presence of camera data
current_x = None
current_y = None
range = None
current_bounding_boxes = None
radar_data = {}
rcs = None
snr = None

# Load camera projection matrix
ndlt = np.load('ndlt_webcam.npy')

# Radius of the circles to be drawn
circle_radius = 4

# Color in (B, G, R) format for drawing circles
colors = (0, 0, 255)

def save_data_to_folder(image, timestamp):
    folder_path = os.path.join('data', timestamp)
    os.makedirs(folder_path, exist_ok=True)

    # Save image
    image_path = os.path.join(folder_path, f'{timestamp}.jpg')
    cv2.imwrite(image_path, image)

    # Find the closest radar data to the current timestamp
    closest_ts = min(a, key=lambda ts: abs(datetime.strptime(ts, '%H:%M:%S.%f') - datetime.strptime(timestamp, '%H:%M:%S.%f')))
    range, closest_x, closest_y, rcs_value, snr_value = radar_data.get(closest_ts, (None, None, None, None))

    if closest_x is not None and closest_y is not None:
        csv_path = os.path.join(folder_path, f'{timestamp}.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for class_name, x_min, y_min, x_max, y_max in current_bounding_boxes:
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                for radar_timestamp, (range, radar_x, radar_y, rcs, snr) in radar_data.items():
                    # print(radar_x)
                    # radar_x = float(radar_x)
                    # radar_y = float(radar_y)
                    if x_min <= radar_x <= x_max and y_min <= radar_y <= y_max:
                        print(radar_x)
                        csv_writer.writerow([radar_x, radar_y, class_name, x_max, y_max, x_min, y_min, range, rcs, snr])


def get_coordinates_in_bounding_box(x_min, y_min, x_max, y_max):
    coordinates = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            coordinates.append((x, y))
    return coordinates

def compute_world2img_projection(world_points, M, is_homogeneous=False):
    if not is_homogeneous:
        points_h = np.vstack((world_points[:3,:], np.ones(world_points.shape[1])))

    h_points_i = M @ points_h  # matrix multiplication


    h_points_i[0, :] = h_points_i[0, :] / h_points_i[2, :]
    h_points_i[1, :] = h_points_i[1, :] / h_points_i[2, :]

    points_i = h_points_i[:2, :]

    return points_i

def closest_timestamp(cv_image=None):
    global a, b, i, current_bounding_boxes, radar_data, rcs, snr

    if i == 1 and len(a) >= 2 and len(b) > 0:
        radar_timestamps = [datetime.strptime(ts, '%H:%M:%S.%f') for ts in a[-2:]]
        camera_timestamp = datetime.strptime(b[-1], '%H:%M:%S.%f')

        closest_ts = min(radar_timestamps, key=lambda ts: abs(ts - camera_timestamp))
        closest_ts_str = closest_ts.strftime('%H:%M:%S.%f')
        print("Closest timestamp of {} is {}".format(b[-1], closest_ts_str))

        range, closest_x, closest_y, rcs, snr = radar_data.get(closest_ts_str, (None, None))

        if closest_x is not None and closest_y is not None:
            x = np.asarray(closest_x)
            y = np.asarray(closest_y)
            z = np.ones(x.shape[0]) * 0.78
            xandy = np.vstack((x, y, z))
            ndlt = np.load('ndlt_webcam.npy')
            circle_radius = 4 
            colors = (0, 0, 255)
            predictions = compute_world2img_projection(xandy, ndlt)
            predictions = np.round(predictions.T)

            if cv_image is not None:
                for class_name, x_min, y_min, x_max, y_max in current_bounding_boxes:
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                    for (x, y) in predictions:
                        p = int(x)
                        q = int(y)
                        if x_min <= p <= x_max and y_min <= q <= y_max:
                            if 0 <= p < cv_image.shape[0] and 0 <= q < cv_image.shape[1]:
                                cv2.circle(cv_image, (p, q), circle_radius, colors, -1)

                cv2.imshow("Received image", cv_image)
                cv2.waitKey(1)

    i = 1  # Indicate that camera data has been found


def callback(data):
    timestamp = datetime.now().strftime('%H:%M:%S.%f')
    print(timestamp)
    a.append(timestamp)
    global current_x, current_y, rcs, snr, radar_data, range
    range, current_x, current_y, rcs, snr = data.range, data.x_dist, data.y_dist, data.RCS, data.SNR
    radar_data[timestamp] = (range, current_x, current_y, rcs, snr)
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
    if current_bounding_boxes and a:
        timestamp = a[-1]
        save_data_to_folder(cv_image, timestamp)

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
