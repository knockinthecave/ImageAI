# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 15:42:52 2022

@author: ionman
"""
# -*- coding: utf-8 -*-


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
from mrcnn.config import Config
from datetime import datetime 
from axis import get_mask_axis
from keras.preprocessing import image
from keras import backend as K


from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

#Before Backend Clear
K.clear_session()


# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
#from samples.coco import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs/shapes20211108T1908/")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(MODEL_DIR ,"mask_rcnn_shapes_0700.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    print("***********************")


# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

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
    NUM_CLASSES = 1 + 5 # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 384

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE =100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

#import train_tongue
#class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we‘ll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()



# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index(‘teddy bear‘)
class_names = ['BG', 'cut','outChar','effect','bubble','character']

# Load a random image from the images folder


a=datetime.now() 


"""

for k in range(1):
    file_names = next(os.walk(IMAGE_DIR))[2]
    image_name= random.choice(file_names)
    
    save_image_name=image_name.split('.')[0]
    image = skimage.io.imread(os.path.join(IMAGE_DIR, image_name))
    
"""    

#Get Today Information
def get_today():
    now = time.localtime()
    s = "%04d%02d%02d%02d%02d%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    return s

def make_folder(folder_name):
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

#images or test
root_dir = "test/"
today = get_today()
work_dir_cut = root_dir+today + "/" + "cut"
work_dir_mask = root_dir+today + "/" + "mask" 
work_dir_segmentation = root_dir+today +"/"+"segmentation"
work_dir_txt = root_dir+today + "/" + "txt"

make_folder(work_dir_cut)
make_folder(work_dir_mask)
make_folder(work_dir_segmentation)
make_folder(work_dir_txt)

img_path_list = []
txt_path_list = []


for filename in os.listdir(IMAGE_DIR):
    
    # if filename.endswith('.bmp'):  #代表结尾是bmp格式的
    img_path = IMAGE_DIR + '/' + filename
    image = cv2.imread(img_path)
    save_image_name = filename.split('.')[0]    
    
    
    
    
#    a=datetime.now() 
    # Run detection
    results = model.detect([image], verbose=1)
#    b=datetime.now() 
    # Visualize results
#    print("time",(b-a).seconds)
    res = results[0]
    rois=res['rois']
    masks=res['masks']
    #boxes=rois
    
    
    
    
    #image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    #    modellib.load_image_gt(IMAGE_DIR, config, image_name, use_mini_mask=False)
    
    #log("gt_class_id", gt_class_id)
    #log("gt_bbox", gt_bbox)
    #log("gt_mask", gt_mask)
    masked_image=visualize.cv_process(image, res['rois'], res['masks'], res['class_ids'],
                                    class_names, res['scores'])

    

    
    
    
    for img in os.listdir():
        #segmentation
        cv2.imwrite("%s/" %work_dir_segmentation+save_image_name+".png", masked_image[:,:,(2,1,0)])
        # img_path_list.append("%s/" %work_dir_segmentation+save_image_name+".png")



    
    for x in range(len(rois)):
        cut_image=image[rois[x][0]:rois[x][2],rois[x][1]:rois[x][3]]
    
        doc = open("%s/" %work_dir_txt+save_image_name+"_"+str(x)+'.txt','w')
        print([rois[x][1]],[rois[x][0]],[rois[x][3]],[rois[x][2]],file=doc)
        doc.close()
        #cut
        skimage.io.imsave("%s/" %work_dir_cut+save_image_name+"_"+str(x)+'.jpg',cut_image)
        img_path_list.append("%s/" %work_dir_cut+save_image_name+"_"+str(x)+'.jpg')
        txt_path_list.append("%s/" %work_dir_txt+save_image_name+"_"+str(x)+'.txt')




    # visualize.display_instances(image, res['rois'], res['masks'], res['class_ids'], 
    #                         class_names, res['scores'])



    mask = res['masks']
    mask = mask.astype(int)
    mask.shape

    for i in range(mask.shape[2]):
        temp = skimage.io.imread(os.path.join(IMAGE_DIR, img_path))
        for j in range(temp.shape[2]):
            temp[:,:,j] = temp[:,:,j] * mask[:,:,i]
    #    plt.figure(figsize=(64,64))       
    #    plt.imshow(temp)
    #mask
        skimage.io.imsave("%s/" %work_dir_mask+save_image_name+"_"+str(i)+'.jpg',temp)
        # img_path_list.append("%s/" %work_dir_mask+save_image_name+"_"+str(i)+'.jpg')


b=datetime.now() 
# Visualize results
print("time",(b-a).seconds)


#After clear
K.clear_session()




sys.modules.pop






