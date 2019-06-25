# keras-yolo3

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

The original implementation comes from [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3).

Our YOLO3 detects 5 different micro-mobility vehicles from the images captured by the [lgsvl/simulator](https://github.com/lgsvl/simulator). The vehicles we are interested in detecting include scooters, hoverboards, skateboards, segways, and one-wheels.

---

## Quick Start

1. Download pretrained weights [here](https://www.dropbox.com/s/a44ly3zd6bzmssw/2d-final-weights-keras-yolo3.h5?dl=0).
2. Move the downloaded .h5 file inside [model_data](model_data)
3. Run YOLO detection.
```
python inference.py [OPTIONS...] --image, for image detection mode
```

### Usage
Use --help to see usage of inference.py:
```
usage: inference.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]

required arguments:
  --image            Image detection mode

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
```

### ROS
If you are trying to run inferences on LGSVL simulators using ROS nodes, `get_detections` function in `yolo.py` returns 2D bounding box information.

---

## Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Modify train.py and start training.  
    `python train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using `inference.py`
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

3. You can access to our datasets we trained our model on from here. These two datasets are separated because they were collected in different times. Setups and formats are identical.

[dataset1_link](https://www.dropbox.com/s/9cvsmraio6q6v0d/large_dataset_1.zip?dl=0)

[dataset2_link](https://www.dropbox.com/s/kt6hwfsa95v4hck/large_dataset_2.zip?dl=0)

---

## Results

### Image
![](docs/images/result.jpg)

### Video
[![2D Perception of Micro-mobility Vehicles on LGSVL Simulator | CMPE 297 Spring 2019](docs/images/thumbnail.jpg)](https://www.youtube.com/watch?v=DwWY89dVGEw)
