import json
from random import shuffle, random

import cv2

import mxnet as mx
import os
from demo import get_detector
from detect.detection import Detection, paint_detection, crop_from_detection
from extract_bbs_config import GPU_ENABLED, GPU_ID, IMAGES_PATH, NUM_IMAGES, EPOCH, NETWORK, DATA_SHAPE, NMS_THRESH, \
    FORCE_NMS, MEAN_RGB, PREFIX, BATCH_SIZE, EXTENSION, SHOW_TIMER, DETECTION_THRESH, OUTPUT_PATH, JSON_PATH, \
    CROPS_PATH, \
    DETECTIONS_PATH, NON_DETECTIONS_PATH, CLASSES
from os.path import join

if __name__ == '__main__':

    if GPU_ENABLED:
        ctx = mx.gpu(GPU_ID)
    else:
        ctx = mx.cpu()

    # Create needed folder structure
    if not os.path.exists(OUTPUT_PATH):
        for dir in [OUTPUT_PATH, NON_DETECTIONS_PATH]:
            os.makedirs(dir)

    image_list = [f for f in os.listdir(IMAGES_PATH) if f.endswith('.jpg')]
    shuffle(image_list)
    image_list = (image_list)[:NUM_IMAGES]

    #   Parse image list
    assert len(image_list) > 0, "No valid image specified to detect"

    detector = get_detector(NETWORK, PREFIX, EPOCH, DATA_SHAPE, MEAN_RGB, ctx, NMS_THRESH, FORCE_NMS)

    num_intervals = len(image_list) / BATCH_SIZE + 1

    for i in range(num_intervals):

        #   Get batch limits
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, len(image_list))

        #   Extract batch images
        im_batch = image_list[start: end]

        #   Detect and store results
        for j, dets in enumerate(detector.im_detect(im_batch, IMAGES_PATH, EXTENSION, SHOW_TIMER)):

            img = cv2.imread(join(IMAGES_PATH, im_batch[j]))
            image_name = im_batch[j][:-4]

            if dets is not None and len(dets) > 0:

                img_dets = img.copy()

                detections = []
                c = 0

                #   Filter detections
                for det in dets:
                    det = Detection(*det)
                    if det.score > DETECTION_THRESH:
                        class_name = CLASSES[int(det.class_id)]
                        class_path = os.path.join(OUTPUT_PATH, class_name)

                        if class_name == "car":
                            detections.append(det)
                            if not os.path.exists(class_path):
                                os.makedirs(class_path)

                            cv2.imwrite(join(class_path, image_name.jpg), img)

                if len(detections) == 0:
                    cv2.imwrite(join(NON_DETECTIONS_PATH, image_name + '.jpg'), img)

            else:
                cv2.imwrite(join(NON_DETECTIONS_PATH, image_name + '.jpg'), img)
