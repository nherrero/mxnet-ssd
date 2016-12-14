import random
from os.path import basename, join

import cv2
import mxnet as mx
import numpy as np
from timeit import default_timer as timer
from dataset.testdb import TestDB
from dataset.iterator import DetIter


class Detector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """

    def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels, \
                 batch_size=1, ctx=None):
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        _, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
        self.mod = mx.mod.Module(symbol, context=ctx)
        self.data_shape = data_shape
        self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape, data_shape))])
        self.mod.set_params(args, auxs)
        self.data_shape = data_shape
        self.mean_pixels = mean_pixels

    def detect(self, det_iter, show_timer=False):
        """
        detect all images in iterator

        Parameters:
        ----------
        det_iter : DetIter
            iterator for all testing images
        show_timer : Boolean
            whether to print out detection exec time

        Returns:
        ----------
        list of detection results
        """
        num_images = det_iter._size
        if not isinstance(det_iter, mx.io.PrefetchingIter):
            det_iter = mx.io.PrefetchingIter(det_iter)
        start = timer()
        detections = self.mod.predict(det_iter).asnumpy()
        time_elapsed = timer() - start
        if show_timer:
            print "Detection time for {} images: {:.4f} sec".format(
                num_images, time_elapsed)
        result = []
        for i in range(detections.shape[0]):
            det = detections[i, :, :]
            res = det[np.where(det[:, 0] >= 0)[0]]
            result.append(res)
        return result

    def im_detect(self, im_list, root_dir=None, extension=None, show_timer=False):
        """
        wrapper for detecting multiple images

        Parameters:
        ----------
        im_list : list of str
            image path or list of image paths
        root_dir : str
            directory of input images, optional if image path already
            has full directory information
        extension : str
            image extension, eg. ".jpg", optional

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        test_db = TestDB(im_list, root_dir=root_dir, extension=extension)
        test_iter = DetIter(test_db, 1, self.data_shape, self.mean_pixels,
                            is_train=False)
        return self.detect(test_iter, show_timer)

    def visualize_detection(self, img, dets, classes=[], thresh=0.6):
        """
        visualize detections in one image

        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        import cv2
        import random

        height = img.shape[0]
        width = img.shape[1]
        colors = dict()

        for i in range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                score = dets[i, 1]
                if score > thresh:
                    if cls_id not in colors:
                        colors[cls_id] = (random.random(), random.random(), random.random())
                    xmin = int(dets[i, 2] * width)
                    ymin = int(dets[i, 3] * height)
                    xmax = int(dets[i, 4] * width)
                    ymax = int(dets[i, 5] * height)

                    color = [c * 255 for c in colors[cls_id]]

                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

                    if classes and len(classes) > cls_id:
                        class_name = classes[cls_id]
                        text_size, baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
                        cv2.rectangle(img, (xmin, ymin - 2), (xmin + text_size[0], ymin - 2 - text_size[1]), (0, 0, 0),
                                      -1)
                        cv2.putText(img, class_name, (xmin, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                                    1)

        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)
        cv2.imshow("img", img)
        cv2.waitKey(0)

    def detect_and_visualize(self, im_list, root_dir=None, extension=None,
                             classes=[], thresh=0.6, show_timer=False):
        """
        wrapper for im_detect and visualize_detection

        Parameters:
        ----------
        im_list : list of str or str
            image path or list of image paths
        root_dir : str or None
            directory of input images, optional if image path already
            has full directory information
        extension : str or None
            image extension, eg. ".jpg", optional

        Returns:
        ----------
        """
        dets = self.im_detect(im_list, root_dir, extension, show_timer=show_timer)
        if not isinstance(im_list, list):
            im_list = [im_list]
        assert len(dets) == len(im_list)
        for k, det in enumerate(dets):
            img = cv2.imread(im_list[k])
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            self.visualize_detection(img, det, classes, thresh)

        return dets

    def store_detections(self, img_name, dets, thresh, classes, root_dir=None):
        """
            wrapper for im_detect and visualize_detection

            Parameters:
            ----------
            im_list : list of str or str
                image path or list of image paths
            root_dir : str or None
                directory of input images, optional if image path already
                has full directory information
            extension : str or None
                image extension, eg. ".jpg", optional

            Returns:
            ----------
        """

        img = cv2.imread(img_name)
        img_p = img.copy()

        if img:

            height = img.shape[0]
            width = img.shape[1]

            colors = {}

            crops = []
            image_crops = []

            for i in range(dets.shape[0]):
                cls_id = int(dets[i, 0])
                if cls_id >= 0:
                    score = dets[i, 1]
                    if score > thresh:
                        if cls_id not in colors:
                            colors[cls_id] = (random.random(), random.random(), random.random())

                        xmin = int(dets[i, 2] * width)
                        ymin = int(dets[i, 3] * height)
                        xmax = int(dets[i, 4] * width)
                        ymax = int(dets[i, 5] * height)

                        #   Get cropped image
                        image_crops.append(img[ymin:ymax, xmin:xmax].copy())

                        #   Paint crop to image
                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[cls_id], 2)

                        #   Add crop to crops vector
                        crops.append({
                            "score": score,
                            "class": classes[cls_id],
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax})

            # Update full JSON information
            detection_info = {
                "image_name": img_name,
                "width": width,
                "height": height,
                "crops": crops
            }

            img_basename = basename(img_name)[:-3]

            #   Store json
            json_path = join(root_dir, img_basename + '.json')

            #   Store crops

            #   Store painted image
