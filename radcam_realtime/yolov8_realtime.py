import hydra
import torch
import cv2
from random import randint
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from datetime import datetime
import time

#ros libraries
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String

class_id = 0
rand_color_list = []
timestamps = []  # List to store the timestamps
bounding_boxes = []  # List to store the bounding box details
detected_frames = []  # List to store frames with detected objects
b =[]
class_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
    9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
    15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
    31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }


def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0), class_names=None):
    labels = []  # List to store the labels
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        box_center = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        
        class_id = int(categories[i])
        class_name = class_names.get(class_id, 'Unknown')  # Get class name from dictionary, default to 'Unknown'
        
        label = class_name  # Use class name as the label
        labels.append(label)  # Store the label
        
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 253), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
    return img, labels

def random_color_list():
    global rand_color_list
    rand_color_list = []
    for i in range(0, 5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)

class DetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))
    
    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        global class_names
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p

        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        bbox_xyxy = det[:, :4]
        identities = det[:, 5]
        categories = det[:, 4]  # Adjusted index to retrieve class ID
        draw_boxes(im0, bbox_xyxy, identities, categories, self.model.names, class_names)

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        # Retrieve the timestamp and append to the timestamps list
        timestamp = datetime.now()
        timestamp = timestamp.strftime("%H:%M:%S.%f")  # Retrieve the timestamp of the current frame
        timestamps.append(timestamp)
        print(timestamp)

        # Retrieve the bounding box details and labels from draw_boxes function
        det_img, det_labels = draw_boxes(im0, bbox_xyxy, identities, categories, self.model.names, class_names)

        ###Sending received image via ROS###
        # Initialize the node with a unique name (anonymous=True)
        rospy.init_node('image_publisher', anonymous=True)

        # Create a publisher that publishes on the "/image_topic" topic
        image_pub = rospy.Publisher('/image_topic', Image, queue_size=10)

        # Create a CvBridge object to convert between OpenCV images and ROS Image messages
        bridge = CvBridge()

        # Create a ROS Image message from the OpenCV image
        ros_image_msg = bridge.cv2_to_imgmsg(det_img, "bgr8")
        image_pub.publish(ros_image_msg)
        ###Sending received image via ROS###


        ##send timestamps via ROS##
        time_pub = rospy.Publisher("/time_topic", String, queue_size=10)
        msg = String()
        msg.data = str(timestamp)
        time_pub.publish(msg)
        b.append(timestamp)
        ##send timestamps via ROS##


        # Store the image with bounding boxes and labels in the detected_frames list
        detected_frames.append(det_img)
        # cv2.imshow(f"At time {timestamp}", det_img)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        
        # Concatenate the labels with the bounding box details
        bounding_box_details = []
        for box, identity, category, label in zip(bbox_xyxy, identities, categories, det_labels):
            class_id = int(category)
            class_name = class_names.get(class_id, 'Unknown')
            bounding_box = [class_name] + [int(coord) for coord in box]
            bounding_box_details.append(bounding_box)
        bounding_boxes.append(bounding_box_details)

        ##ROS List##
        pub = rospy.Publisher("/object_topic", Float32MultiArray, queue_size=10)
        msg = Float32MultiArray()
        list_of_lists = bounding_box_details
        for sublist in list_of_lists:
            sublist.append(-1)
        flattened_data = [item for sublist in list_of_lists for item in sublist]
        msg.data = flattened_data
        pub.publish(msg)
        ##ROS List#

        print(bounding_box_details)
        return log_string

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    random_color_list()

    # Define the class names dictionary
    class_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
    9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird',
    15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
    31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }

    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()

if __name__ == "__main__":
    predict()
    time.sleep(0)
