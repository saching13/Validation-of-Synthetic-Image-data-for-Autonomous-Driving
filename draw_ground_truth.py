import numpy as np
import os
import cv2
from PIL import Image, ImageDraw
import csv
import glob

def draw_ground_truth(image_path, boxes, save_folder):
    fn = os.path.basename(image_path)
    new_fn = 'gt_' + fn
    new_fp = os.path.join(save_folder, new_fn) 
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        box = box.split(',')

        left = int(float(box[0]))
        top = int(float(box[1]))
        right = int(float(box[2]))
        bottom = int(float(box[3]))

        draw.line((left,top, left, bottom), fill=128,width=3)
        draw.line((left,top, right,top), fill=128,width=3)
        draw.line((right,bottom, left,bottom), fill=128,width=3)
        draw.line((right,bottom, right,top), fill=128,width=3)

    image.save(new_fp, fomrat='jpg')

def draw_all_ground_truths(image_folder, annotations_file, save_folder):
    count = 0
    with open(annotations_file, 'r') as af:
        for count, line in enumerate(af):
            line = line.strip('\r\n')
            line_split = line.split(' ')
            image_path = line_split[0]
            boxes = line_split[1:-1]
            draw_ground_truth(image_path, boxes, save_folder)
            count += 1
            print("Drawn " + str(count) + " images.")

if __name__ == '__main__':
    image_folder = '../lgsvl_2_dataset_3/main_camera/'
    annotations_file = '../lgsvl_2_dataset_3/gt_2d/gt_2d_yolo3_annotations.txt'
    save_folder = '../lgsvl_2_dataset_3/gt_2d/gt_2d_visual/'
    draw_all_ground_truths(image_folder, annotations_file, save_folder)