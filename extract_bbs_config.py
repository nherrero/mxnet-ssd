import os

from os.path import join

CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

IMAGES_PATH = '/Users/nherrero/Desktop/animal_images/'
OUTPUT_PATH = join(IMAGES_PATH, '__output')
JSON_PATH = join(OUTPUT_PATH, 'json')
CROPS_PATH = join(OUTPUT_PATH, 'crops')
DETECTIONS_PATH = join(OUTPUT_PATH, 'detections')
NON_DETECTIONS_PATH = join(OUTPUT_PATH, 'non_detections')

NETWORK = 'vgg16_reduced'
PREFIX = join(os.getcwd(), 'model', 'ssd')
EPOCH = 0
DATA_SHAPE = 300
MEAN_RGB = (123, 116, 104)
NMS_THRESH = 0.7
FORCE_NMS = True
DIR = None
EXTENSION = None
DETECTION_THRESH = 0.7
SHOW_TIMER = True

GPU_ENABLED = 0
GPU_ID = 0
BATCH_SIZE = 5
NUM_IMAGES = 64
# NUM_IMAGES = -1
