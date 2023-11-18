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

# ROS libraries
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String
from conti_radar.msg import radar_img

class_id = 0
rand_color_list = []
timestamps = []  # List to store the timestamps
bounding_boxes = []  # List to store the bounding box details
detected_frames = []  # List to store frames with detected objects

# ROS callback function for the radar_img topic
def radar_callback(data):
    print("x_range: {} , y_dist:  {} " .format(len(data.x_dist),len(data.y_dist)))

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
        draw_boxes(im0, bbox_xyxy, identities, categories, self.model.names)

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

    def __init__(self, cfg):
        # Initialize the ROS node for the image_publisher and time_publisher
        rospy.init_node('image_time_publisher', anonymous=True)
        self.image_pub = rospy.Publisher('/image_topic', Image, queue_size=10)
        self.time_pub = rospy.Publisher("/time_topic", String, queue_size=10)

        # Subscribe to the radar_img topic
        rospy.Subscriber('/radar_img', radar_img, radar_callback)

        super().__init__(cfg)

    def write_results(self, idx, preds, batch):
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
        draw_boxes(im0, bbox_xyxy, identities, categories, self.model.names)

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        # Retrieve the timestamp and append to the timestamps list
        timestamp = datetime.now()
        timestamp = timestamp.strftime("%H:%M:%S.%f")  # Retrieve the timestamp of the current frame
        timestamps.append(timestamp)
        # print(timestamp)

        # Retrieve the bounding box details and labels from draw_boxes function
        det_img, det_labels = draw_boxes(im0, bbox_xyxy, identities, categories, self.model.names)

        # ROS Image Publisher
        bridge = CvBridge()
        ros_image_msg = bridge.cv2_to_imgmsg(det_img, "bgr8")
        self.image_pub.publish(ros_image_msg)

        # ROS Timestamp Publisher
        msg = String()
        msg.data = str(timestamp)
        self.time_pub.publish(msg)

        # Store the image with bounding boxes and labels in the detected_frames list
        detected_frames.append(det_img)

        # Concatenate the labels with the bounding box details
        bounding_box_details = []
        for box, identity, category, label in zip(bbox_xyxy, identities, categories, det_labels):
            bounding_box = [int(label)] + [int(coord) for coord in box]
            bounding_box_details.append(bounding_box)
        bounding_boxes.append(bounding_box_details)

        # ROS List Publisher
        msg = Float32MultiArray()
        pub = rospy.Publisher("/multi_array_topic", Float32MultiArray, queue_size=10)
        list_of_lists = bounding_box_details
        for sublist in list_of_lists:
            sublist.append(-1)
        flattened_data = [item for sublist in list_of_lists for item in sublist]
        msg.data = flattened_data
        pub.publish(msg)

        return log_string

    
def random_color_list():
    global rand_color_list
    rand_color_list = []
    for i in range(0, 5005):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)

def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    labels = []  # List to store the labels
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        id = int(identities[i]) if identities is not None else 0
        box_center = (int((box[0]+box[2])/2),(int((box[1]+box[3])/2)))
        label = str(id)
        labels.append(label)  # Store the label
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 253), 2)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, label, (x1, y1 - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
    return img, labels
# Existing code...

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    random_color_list()
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()

# Existing code...

if __name__ == "__main__":
    predict()
    time.sleep(3)

    # Visualize frames with detected objects sequentially
    # for idx, frame in enumerate(detected_frames):
    #     cv2.imshow(f"Frame {idx}", frame)
    #     cv2.waitKey(1000)  # Wait for 1000 milliseconds (1 second)
    #     cv2.destroyAllWindows()
