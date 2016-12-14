import os
import mxnet as mx

from demo import get_detector
from os.path import join
from random import shuffle

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

IMAGES_PATH = '/Users/nherrero/Desktop/animal_images/'
OUTPUT_PATH = join(IMAGES_PATH, 'output')

NETWORK = 'vgg16_reduced'
PREFIX = join(os.getcwd(), 'model', 'ssd')
EPOCH = 0
DATA_SHAPE = 300
MEAN_RGB = (123, 116, 104)
NMS_THRESH = 0.5
FORCE_NMS = True
DIR = None
EXTENSION = None
DETECTION_THRESH = 0.5
SHOW_TIMER = True

GPU_ENABLED = 0
GPU_ID = 0
BATCH_SIZE = 32

if __name__ == '__main__':

    if GPU_ENABLED:
        ctx = mx.gpu(GPU_ID)
    else:
        ctx = mx.cpu()

    image_list = [join(IMAGES_PATH, f) for f in os.listdir(IMAGES_PATH) if f.endswith('.jpg')]
    shuffle(image_list)

    #   Parse image list
    assert len(image_list) > 0, "No valid image specified to detect"

    detector = get_detector(
        NETWORK, PREFIX, EPOCH,
        DATA_SHAPE, MEAN_RGB, ctx,
        NMS_THRESH, FORCE_NMS
    )

    num_intervals = len(image_list)/BATCH_SIZE

    for i in range(num_intervals):

        #   Get batch limits
        start = i * BATCH_SIZE
        end = len(image_list[:-1]) if i == num_intervals - 1 else (i+1) * BATCH_SIZE

        #   Extract batch images
        im_batch = image_list[start : end]

        #   Run detection in batches
        dets = detector.detect_and_visualize(
            image_list,
            DIR,
            EXTENSION,
            CLASSES,
            DETECTION_THRESH,
            SHOW_TIMER
        )

        #   Store detections as json files



    print("Number of intervals: %d" % num_intervals)

    # run detection
    # dets = detector.detect_and_visualize(image_list, args.dir, args.extension,
    #                               CLASSES, args.thresh, args.show_timer)
    #
    # print dets
