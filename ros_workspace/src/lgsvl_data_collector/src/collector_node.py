#! /usr/bin/env python
import rospy
import os
import numpy as np
import csv
import cv2
from sensor_msgs.msg import CompressedImage
from lgsvl_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
import message_filters
from datetime import datetime

# Save dataset here
folder = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Fall2019/CMPE256/Project/Datasets/lgsvl_2/lgsvl_2_dataset_3/"
main_camera_subfolder = "main_camera"

gt_2d_subfolder = "gt_2d/"

# Annotation Filenames
yolo2_annotations = "gt_2d_yolo2_annotations.csv"

vehicle_labels = {
    "car": "0",
    "SUV": "0",
    "Sedan": "0",
    "Jeep": "0",
    "BoxTruck": "0",
    "SchoolBus": "0",
    "Hatchback": "0",
    "Pedestrian": "1"
}

# Number of datapoints saved
datapoint_count = 0

class DataCollector:

    def __init__(self):
        rospy.init_node("data_collector_1")
        print("YO! The node is running. Now actually collect data")
        self.listener()
        rospy.spin()

    def callback(self, main_camera, gt_2d):
        print("Callback")
        global datapoint_count

        msg_id = str(datetime.now().isoformat())

        if len(gt_2d.detections) > 0:
            self.save_images(main_camera, msg_id)
            self.save_gt_2d(gt_2d, msg_id)
            datapoint_count += 1
        else:
            print("Skipped at: " + msg_id)
        print("Number of datapoints: " + str(datapoint_count))

    def listener(self):

        # Change the topic to apollo camera
        sub_main_camera = message_filters.Subscriber('/apollo/sensor/camera/traffic/image_short/compressed', CompressedImage)
        sub_gt_2d = message_filters.Subscriber('/simulator/ground_truth/2d_detections', Detection2DArray)
        ts = message_filters.ApproximateTimeSynchronizer([sub_main_camera, sub_gt_2d], 1, 0.1)
        ts.registerCallback(self.callback)
    
    def save_images(self, main_camera, msg_id):
        main_camera_np_arr = np.fromstring(bytes(main_camera.data), np.uint8)
        main_camera_cv = cv2.imdecode(main_camera_np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(folder + main_camera_subfolder, 'main-{}.jpg'.format(msg_id)), main_camera_cv)
    
    def save_gt_2d(self, gt_2d, msg_id):

        main_image_filename = 'main-{}.jpg'.format(msg_id)

        yolo2_string = ""

        # 2D ground truths for main camera and depth sensor
        for det in gt_2d.detections:
            # YOLOv2 style (x_min=0, x_max=1920, y_min=0, y_max=1080)
            label = vehicle_labels.get(det.label)
            if label != "Pedestrian":
                x_min = str(int(max(det.bbox.x - det.bbox.width/2, 0)))
                x_max = str(int(min(det.bbox.x + det.bbox.width/2, 1000))) 
                y_min = str(int(max(det.bbox.y - det.bbox.height/2, 0)))
                y_max = str(int(min(det.bbox.y + det.bbox.height/2, 600)))
                print(x_min, y_min, x_max, y_max, label)
                yolo2_string += x_min + "," + y_min + "," + x_max + "," + y_max + "," + label + " "
            else:
                print("Pedestrian ignored")

        yolo2_filename = folder + gt_2d_subfolder + yolo2_annotations
        
        with open(yolo2_filename, 'a') as f_yolo2:
            writer = csv.writer(f_yolo2, delimiter=';')
            writer.writerow([main_image_filename, yolo2_string])

data_collector_node = DataCollector()
