import json

import cv2


class Detection:
    def __init__(self, class_id, score, xmin, ymin, xmax, ymax):
        self.class_id = float(class_id)
        self.score = float(score)
        self.xmin = float(xmin)
        self.ymin = float(ymin)
        self.xmax = float(xmax)
        self.ymax = float(ymax)


def get_transformed_coordinates(det, img):
    assert img is not None
    height, width = img.shape[0:2]
    return int(det.xmin * width), int(det.ymin * height), int(det.xmax * width), int(det.ymax * height)


def paint_detection(det, img, color=(0, 0, 255), thickness=2):
    assert img is not None

    xmin, ymin, xmax, ymax = get_transformed_coordinates(det, img)

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

    return img

def crop_from_detection(det, img):
    assert img is not None

    xmin, ymin, xmax, ymax = get_transformed_coordinates(det, img)

    return img[ ymin : ymax, xmin : xmax]


if __name__ == '__main__':

    det = Detection( 8.0, 0.9, 0.1, 9.2, 9.3, 9.4)


