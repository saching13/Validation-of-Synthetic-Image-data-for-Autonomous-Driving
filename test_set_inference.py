import numpy as np
import os
import cv2
from PIL import Image
from yolo import YOLO
import csv
import glob

class TestSetInference:

    labels_mapping = {
        "vehicle" : "0",
        "pedestrian" : "1"
    }

    def __init__(self, 
                test_set_folder, 
                detections_output_file, 
                test_set_images=None,
                weights=None, 
                anchors=None, 
                labels=None,
                score=0.1,
                iou=0.1):
        self.test_set_folder = test_set_folder
        self.detections_output_file = detections_output_file
        self.test_set_images = test_set_images
        self.yolo = YOLO(model_path=weights, anchors_path=anchors, classes_path=labels, score=score, iou=iou)
        self.seq = 0

    def detect_single_image(self, image, display=False, image_name=None, save_image_folder=None):
        self.seq += 1
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image_pil = Image.fromarray(image)
        objects = self.yolo.get_detections(image_pil)
        if display:
            r_image = self.yolo.detect_image(image_pil)
            if save_image_folder and image_name:
                r_image.save(os.path.join(save_image_folder, image_name))
            # cv_image = cv2.cvtColor(np.array(r_image), cv2.COLOR_RGB2BGR)
            # cv2.imshow("Detected Objects", cv_image)
            # cv2.waitKey(1)
        return_list = ""
        for obj in objects:
            xmin = obj['x']
            ymin = obj['y']
            xmax = obj['x'] + obj['width']
            ymax = obj['y'] + obj['height']
            score = obj['score']
            label = TestSetInference.labels_mapping.get(obj['label'])
            obj_str = str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(score) + ',' + label + ' '
            return_list += obj_str
        return return_list
    
    def run_inference(self, save_folder=None, ext='.jpg'):
        # Get a list of images
        files = []
        if self.test_set_images is not None:
            with open(self.test_set_images, 'r') as f_in:
                for count, line in enumerate(f_in):
                    image_name = line.split(' ')[0]
                    files.append(os.path.join(self.test_set_folder, image_name + ext))
        else:
            types = ['*.jpg', '*.png']
            for t in types:
                files.extend(glob.glob(self.test_set_folder + t))
        files.sort()
        print("Files found: " + str(len(files)))
        print(files[0])

        # Get detections for each image
        for file in files:
            fn = os.path.basename(file)
            image = cv2.imread(file)
            # Output detections to the save_folder
            dets = self.detect_single_image(image, display=True, image_name=fn, save_image_folder=save_folder)
            # Append to CSV file
            with open(self.detections_output_file, 'a') as f_out:
                wr = csv.writer(f_out, delimiter=';')
                wr.writerow([fn, dets])
            
            print("Processed " + str(self.seq) + " images")


#### Example 1: On single image
# test_set_folder = ""
# detections_output_file = ""
# weights = 'logs/lgsvl_2_dataset_1/000ep039-loss16.616-val_loss17.654.h5'
# anchors = 'model_data/yolo_lgsvl_2_anchors.txt'
# labels = 'model_data/yolo_lgsvl_2_labels.txt'
# score = 0.05
# iou = 0.01

# tsi = TestSetInference(test_set_folder, 
#                         detections_output_file, 
#                         weights=weights, 
#                         anchors=anchors, 
#                         labels=labels, 
#                         score=score, 
#                         iou=iou)

# test_image_folder = "test_folder"
# test_image_name = 'test4.jpg'

# test_image_path = os.path.join(test_image_folder, test_image_name)
# det_image_path = os.path.join(test_image_folder, "detections")
# # test_image = "/home/deepaktalwardt/Dropbox/SJSU/Semesters/Fall2019/CMPE256/Project/ObjectDetection/training1/keras-yolo3/test_folder/test2.jpg"

# t_image = cv2.imread(test_image_path)
# det_obj_str = tsi.detect_single_image(t_image, display=True, image_name=test_image_name, save_image_folder=det_image_path)

#### Example 2: On sample testset
# test_set_folder = "../../sample_testset/images/"
# detections_output_file = "../detections/sample_testset/detections_lgsvl_2.csv"
# weights = 'logs/lgsvl_2_dataset_3/000ep078-loss9.684-val_loss9.917.h5'
# anchors = 'model_data/yolo_no_ped_anchors.txt'
# labels = 'model_data/yolo_no_ped_labels.txt'
# score = 0.25
# iou = 0.5

# tsi = TestSetInference(test_set_folder, 
#                         detections_output_file, 
#                         weights=weights, 
#                         anchors=anchors, 
#                         labels=labels, 
#                         score=score, 
#                         iou=iou)

# detections_folder = '../detections/sample_testset/images'
# tsi.run_inference(save_folder=detections_folder)

#### Example 3: On entire Waymo testset
# test_set_folder = "../../waymo_testset/images/"
# detections_output_file = "../detections/waymo_testset/detections_lgsvl_2_waymo_testset.csv"
# weights = 'logs/lgsvl_2_dataset_3/000ep078-loss9.684-val_loss9.917.h5'
# anchors = 'model_data/yolo_no_ped_anchors.txt'
# labels = 'model_data/yolo_no_ped_labels.txt'
# test_set_images = "../../waymo_testset/waymo_testset_1000_sorted_iou_2.txt"
# score = 0.1
# iou = 0.1

# tsi1 = TestSetInference(test_set_folder, 
#                         detections_output_file,
#                         test_set_images=test_set_images,
#                         weights=weights, 
#                         anchors=anchors, 
#                         labels=labels, 
#                         score=score, 
#                         iou=iou)

# detections_folder = '../detections/waymo_testset/images'
# tsi1.run_inference(save_folder=detections_folder)


#### Example 4: On entire KITTI dataset using KITTI weights
# test_set_folder = "../../kitti/training/image_2/"
# detections_output_file = "../kitti_detections/detections_kitti_kitti_training_set.csv"
# detections_folder = '../kitti_detections/images'
# weights = 'logs/kitti_training_weights/kitti_epoch_final_30iter_unfreeze_all.h5'
# anchors = 'model_data/kitti_training_anchors.txt'
# labels = 'model_data/yolo_no_ped_labels.txt'
# score = 0.1
# iou = 0.1

# tsi2 = TestSetInference(test_set_folder, 
#                         detections_output_file, 
#                         weights=weights, 
#                         anchors=anchors, 
#                         labels=labels, 
#                         score=score, 
#                         iou=iou)

# tsi2.run_inference(save_folder=detections_folder)

#### Example 5: On top 1000 KITTI dataset using KITTI weights
test_set_folder = "../../lgsvl_2_testset/main_camera/"
# test_set_images = '../../kitti/kitti_testset_1000_iou_2.txt'
detections_output_file = "../lgsvl_1_detections/lgsvl_1_on_lgsvl_2_testset/detections_lgsvl_1_lgsvl_2_testset.csv"
detections_folder = '../lgsvl_1_detections/lgsvl_1_on_lgsvl_2_testset/images'
weights = 'logs/lgsvl_1_training_weights/lgsvl_1trained_weights_final.h5'
anchors = 'model_data/lgsvl_1_training_anchors.txt'
labels = 'model_data/yolo_no_ped_labels.txt'
score = 0.1
iou = 0.1

tsi2 = TestSetInference(test_set_folder, 
                        detections_output_file, 
                        # test_set_images=test_set_images,
                        weights=weights, 
                        anchors=anchors, 
                        labels=labels, 
                        score=score, 
                        iou=iou)

tsi2.run_inference(save_folder=detections_folder, ext='.jpg')


### Example 6: Using LGSVL 2
# test_set_folder = "../../kitti/training/image_2/"
# test_set_images = '../../kitti/kitti_testset_1000_iou_2.txt'
# detections_output_file = "../detections/lgsvl_2_ON_kitti_testset/detections_lgsvl_2_kitti_testset.csv"
# detections_folder = '../detections/lgsvl_2_ON_kitti_testset/images'
# weights = 'logs/lgsvl_2_dataset_3/000ep078-loss9.684-val_loss9.917.h5'
# anchors = 'model_data/yolo_no_ped_anchors.txt'
# labels = 'model_data/yolo_no_ped_labels.txt'
# score = 0.15
# iou = 0.2

# tsi1 = TestSetInference(test_set_folder, 
#                         detections_output_file,
#                         test_set_images=test_set_images,
#                         weights=weights, 
#                         anchors=anchors, 
#                         labels=labels, 
#                         score=score, 
#                         iou=iou)

# tsi1.run_inference(save_folder=detections_folder, ext='.png')