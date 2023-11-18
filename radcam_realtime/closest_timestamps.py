import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
from conti_radar.msg import radar_img
from datetime import datetime

global a
a = []
global b
b = []

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


def callback(data):
    timestamp = datetime.now()
    timestamp = timestamp.strftime("%H:%M:%S.%f")
    print(timestamp)
    print("x_range: {} , y_dist:  {} " .format(len(data.x_dist),len(data.y_dist)))

def bbox_callback(msg):
    list_of_lists = unflatten_data(msg.data)
    rospy.loginfo("Received data: %s", list_of_lists)


def time_callback(msg):
    rospy.loginfo("Received Timestamp for Image %s", msg.data)
    print(type(a))
    global time
    time = str(msg.data)

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
