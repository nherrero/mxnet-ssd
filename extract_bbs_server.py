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
    DETECTIONS_PATH
from os.path import join

if __name__ == '__main__':

    if GPU_ENABLED:
        ctx = mx.gpu(GPU_ID)
    else:
        ctx = mx.cpu()

    image_path = ['/home/ubuntu/speed-boat/197_0006.jpg']
    output_path = '/home/ubuntu/speed-boat/out'

    assert len(image_path) > 0, "No valid image specified to detect"

    detector = get_detector(NETWORK, PREFIX, EPOCH, DATA_SHAPE, MEAN_RGB, ctx, NMS_THRESH, FORCE_NMS)

    for j, dets in enumerate(detector.im_detect(image_path, IMAGES_PATH, EXTENSION, SHOW_TIMER)):

        img = cv2.imread(image_path[0])
        image_name = image_path[j][:-4]
        img_dets = img.copy()

        detections = []
        c = 0

        #   Filter detections
        for det in dets:

            det = Detection(*det)
            print(det.score)
            if det.score > DETECTION_THRESH:
                detections.append(det)

                #   Store crop
                cv2.imwrite(join(output_path, image_name + '_crop_%04d.jpg' % c), crop_from_detection(det, img))
                c += 1

                #   Paint detection
                # paint_detection(det, img_dets, [random() * 255 for ch in range(3)])

        # cv2.imwrite(join(output_path, image_name + '_detections.jpg'), img_dets)

        detection_info = {
            "image_name": image_name,
            "width": img.shape[1],
            "height": img.shape[0],
            "crops": [det.__dict__ for det in detections]
        }

        with open(join(output_path, image_name + '.json'), 'w') as fp:
            json.dump(detection_info, fp, sort_keys=True, indent=4)
