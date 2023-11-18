import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
from conti_radar.msg import radar_img
from datetime import datetime, timedelta

global is_window_created
is_wi = False

# Store the latest radar timestamps
radar_timestamps = []


def unflatten_data(flattened_data):
    list_of_lists = []
    sublist = []
    for item in flattened_data:
        if item == -1:
            list_of_lists.append(sublist)

# Add global variables to store camera timestamp and radar timestamps
camera_timestamp = None
radar_timestamps = []

def find_closest_radar_timestamp(camera_timestamp):
    if not radar_timestamps:
        return None

    closest_timestamp = min(radar_timestamps, key=lambda x: abs(x - camera_timestamp))
    return closest_timestamp

def callback(data):
    global camera_timestamp

    if camera_timestamp is None:
        print("No camera timestamp available.")
        return

    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%H:%M:%S.%f")
    print(timestamp_str)
    print("x_range: {}, y_dist: {}".format(len(data.x_dist), len(data.y_dist)))

    # Find the closest radar timestamp to the camera timestamp
    closest_radar_timestamp = find_closest_radar_timestamp(camera_timestamp)
    if closest_radar_timestamp:
        # Calculate the time difference between the timestamps
        time_difference = closest_radar_timestamp - camera_timestamp
        print("Closest radar timestamp: {}".format(closest_radar_timestamp.strftime("%H:%M:%S.%f")))
        print("Time difference: {}".format(time_difference))
    else:
        print("No radar timestamp available.")

def bbox_callback(msg):
    global radar_timestamps
    list_of_lists = unflatten_data(msg.data)
    rospy.loginfo("Received data: %s", list_of_lists)

    if list_of_lists and len(list_of_lists) > 0:
        # Update the radar timestamps list
        radar_timestamps = [datetime.strptime(time_str, "%H:%M:%S.%f") for time_str in list_of_lists[0]]
    else:
        print("No radar timestamps received.")

def time_callback(msg):
    global camera_timestamp
    rospy.loginfo("Received Timestamp for Image %s", msg.data)
    camera_timestamp = datetime.strptime(msg.data, "%H:%M:%S.%f")
    print(camera_timestamp)
    
def image_callback(msg):
    try:
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imshow("Subscribed Image", cv_image)
        cv2.waitKey(1)
    except Exception as e:
        rospy.logerr("Error processing the image: %s", str(e))

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('/radar_img', radar_img, callback)
    rospy.Subscriber("/object_topic", Float32MultiArray, bbox_callback)
    rospy.Subscriber("/time_topic", String, time_callback)
    rospy.Subscriber("/image_topic", Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
