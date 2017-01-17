import json

from flask import Flask, request
from flask_cors import CORS

import cv2

import mxnet as mx
from demo import get_detector
from detect.detection import Detection, crop_from_detection
from extract_bbs_config import GPU_ENABLED, GPU_ID, IMAGES_PATH, EPOCH, NETWORK, DATA_SHAPE, NMS_THRESH, \
    FORCE_NMS, MEAN_RGB, PREFIX, EXTENSION, SHOW_TIMER, DETECTION_THRESH
from os.path import join
import random
import string
import requests
import shutil


UPLOAD_FOLDER = '/tmp/'



app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/test', methods=['GET'])
def test():
    return 'server responding'


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'GET':

        probabilities = []

        if 'image_url' in request.args:
            image_url = request.args.get('image_url')
            img_path = download_image(image_url)
            image_path = [img_path]
        else:
            return 'error'
        for j, dets in enumerate(app.dtr.im_detect(image_path, IMAGES_PATH, EXTENSION, SHOW_TIMER)):

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
                    crop_name = image_name + '_crop_%04d.jpg' % c
                    cv2.imwrite(image_name + crop_name, crop_from_detection(det, img))
                    c += 1

                    # call clasification API
                    url = 'http://localhost:5000/classify?image_file={path}&model_id=animals'.format(path=crop_name.replace('/tmp/', ''))
                    response = requests.get(url)
                    probabilities.append(response.text)

                    #   Paint detection
                    # paint_detection(det, img_dets, [random() * 255 for ch in range(3)])

            # cv2.imwrite(join(output_path, image_name + '_detections.jpg'), img_dets)

            # detection_info = {
            #     "image_name": image_name,
            #     "width": img.shape[1],
            #     "height": img.shape[0],
            #     "crops": [det.__dict__ for det in detections]
            # }

            # with open(join(output_path, image_name + '.json'), 'w') as fp:
            #     json.dump(detection_info, fp, sort_keys=True, indent=4)

    return probabilities


def randomword(length):
    return ''.join(random.choice(string.lowercase) for i in range(length))


def download_image(url):
    img_path = UPLOAD_FOLDER + randomword(15) + '.jpg'
    f = open(img_path, 'w')
    response = requests.get(url, stream=True)
    response.raw.decode_content = True
    shutil.copyfileobj(response.raw, f)
    f.close()
    return img_path


def start(app):

    if GPU_ENABLED:
        ctx = mx.gpu(GPU_ID)
        print('Using GPU')
    else:
        ctx = mx.cpu()
        print('Using CPU')

    detector = get_detector(NETWORK, PREFIX, EPOCH, DATA_SHAPE, MEAN_RGB, ctx, NMS_THRESH, FORCE_NMS)

    app.dtr = detector

    app.run(threaded=True, host='0.0.0.0', port=5001)


if __name__ == '__main__':

    start(app)
