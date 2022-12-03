#https://blog.csdn.net/tzwsg/article/details/111562751



# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import numpy as 
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from mrcnn.config import Config
from datetime import datetime
from axis import get_mask_axis
# Root directory of the project
ROOT_DIR = os.getcwd()
 
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
# sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# from samples.coco import coco
 
 
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
 
# Local path to trained weights file
COCO_MODEL_PATH = "./logs/shapes20201222T0950/mask_rcnn_shapes_0040.h5"   #  模型保存目录
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("cuiwei***********************")
 
# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "loose_nut_images")  #图片所在文件夹

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
 
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # Number of classes (including background)
    NUM_CLASSES = 2 + 1  # background + 2 shapes
 
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256     #图片尺寸
 
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels
 
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100
 
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 50
 
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50
 
#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
config = InferenceConfig()
 
# model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
 
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
 
# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
 
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear's)
class_names = ['BG', 'screw', 'nut']   # 注意修改类别名称，第一个为background不可少
# Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]
# print(file_names)

######--批量测试--#######
count = os.listdir(IMAGE_DIR)   #count为图片名称列表
for i in range(0, len(count)):
    path = os.path.join(IMAGE_DIR, count[i])
    # if os.path.isfile(path)
    image = skimage.io.imread(os.path.join(IMAGE_DIR, count[i]))
    if image.ndim != 3:                         #测试灰度图，将灰度图转为rgb图
        image = skimage.color.gray2rgb(image)
    # a=datetime.now()
    # Run detection
    results = model.detect([image], verbose=1)
    # b = datetime.now()
    # Visualize results
    # print("time: ",(b-a).seconds)
    r = results[0]
    visualize.display_instances(count[i], image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
    # print(r['masks'].shape)
    get_mask_axis(count[i], r['masks'], r['class_ids'])     #获取mask像素坐标

######--单张图片测试--####### 
# image = skimage.io.imread('./nut_images/675_1.jpg')      # 你想要测试的图片
# print(image.shape)
# if image.ndim != 3:
#     image = skimage.color.gray2rgb(image)
#     print('after gray2rgb: ')
#     print(image.shape)
#     skimage.io.imsave('102_1_rgb.jpg', image)
 
# a=datetime.now()
# # Run detection
# results = model.detect([image], verbose=1)
# b = datetime.now()
# # Visualize results
# print("time: ",(b-a).seconds)
# r = results[0]
# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                             class_names, r['scores'])
# print(r['masks'].shape)

# get_mask_axis(r['masks'])
# # print(x)