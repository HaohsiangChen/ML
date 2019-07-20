"""Sucpicion detection class which performs various detections from frame."""
import copy

import numpy as np

from core.classifiers import YOLOClassifier
from core.services import BoxPlotter
from core.platform.async import async
import vgconf


class YOLODetection(object):
    async = async.Async()

    def __init__(self):
        self.yolo_inference_buffer = []
        self.yolo_buffer = []

        self._yolo_buffer_lock = self.async.get_lock()

        self.is_yolo_on = False
        self.yolo = None

        self.yolo_sample_rate = vgconf.DEFAULT_YOLO_SAMPLE_RATE
        self.firearm_sample_rate = vgconf.DEFAULT_FIREARM_SAMPLE_RATE
        self.inception_sample_rate = vgconf.DEFAULT_INCEPTION_SAMPLE_RATE

        self.count = 0
        self.sample_rate_lcm = self._get_sample_rate_lcm()

        self.box_plotter = BoxPlotter.BoxPlotter()
        self.is_closed = False

    def _get_gcd(self, a, b):
        while b > 0:
            a, b = b, a % b
        return a

    def _get_lcm(self, a, b):
        return (a * b) / self._get_gcd(a, b)

    def _get_sample_rate_lcm(self):
        return self._get_lcm(
            self._get_lcm(self.yolo_sample_rate, self.inception_sample_rate),
            self.firearm_sample_rate)

    def _get_random_input(self, shape):
        return np.random.uniform(size=shape)


    def _get_yolo_classifier(self):
        self.yolo = YOLOClassifier.YOLO()
        random_input = self._get_random_input((299, 299, 3))
        x = self.yolo.predict(random_input)
        self.is_yolo_on = True


    def _remove_event_detector(self):
        if not self.is_activity_detector_on:
            self._remove_inception()
        self.is_event_detector_on = False
        self.event_detector.close()
        self.event_detector_buffer=[]
        self.event_detector_inference_buffer=[]
        self.event_detector = None


    def _remove_yolo_classifier(self):
        if not self.is_yolo_on:
            return
        self.is_yolo_on = False
        self.yolo.close()
        self.yolo_inference_buffer=[]
        self.yolo_buffer=[]
        self.yolo = None


    def enable_yolo_detection(self):
        if self.is_yolo_on:
            return
        self._get_yolo_classifier()
        self.box_plotter.add_labels(self.yolo.get_labels())


    def disable_yolo_detection(self):
        if not self.is_yolo_on:
            return
        self._remove_yolo_classifier()


    def set_yolo_sample_rate(self, rate):
        self.yolo_sample_rate = rate
        self.sample_rate_lcm = self._get_sample_rate_lcm()


    @async.synchronize(lock='_yolo_buffer_lock')
    def _yolo_callback(self, result):
        if result is None or self.is_closed:
            return
        self.yolo_buffer.append(result)
        if len(self.yolo_buffer) > 1:
            self.yolo_buffer.pop(0)

    @async.async_call(callback=_yolo_callback)
    def _yolo_inference(self):
        if not self.yolo_inference_buffer or self.is_closed:
            return
        return self.yolo.predict(self.yolo_inference_buffer.pop(0))

    def perform_yolo_inference(self, frame):
        self.yolo_inference_buffer.append(frame)
        if len(self.yolo_inference_buffer) > 1:
            self.yolo_inference_buffer.pop(0)
        return self._yolo_inference()

    @async.synchronize(lock='_yolo_buffer_lock')
    def get_yolo_prediction(self):
        if not self.yolo_buffer or self.is_closed:
            return []
        return self.yolo_buffer[-1]

    def plot_objects(self, img):
        if self.is_yolo_on:
            img = self.box_plotter.plot_bboxes(img, self.get_yolo_prediction())

        # if self.is_firearm_detector_on:
        #     img = self.box_plotter.plot_bboxes(
        #         img, self.get_firearm_detector_prediction())

        return img

    def detect(self, frame):
        # If we remove following line and use 'plot_objects' method to plot
        # objects then classifiers will give wrong output. A prime example
        # of how tricky an error can be.
        frame = copy.deepcopy(frame)

        if self.is_yolo_on:
            if self.count % self.yolo_sample_rate == 0:
                self.perform_yolo_inference(frame)

        self.count += 1
        if self.count == self.sample_rate_lcm:
            self.count = 0

    def close(self):
        if not self.is_closed:
            self.is_closed = True
            self.async.close()
