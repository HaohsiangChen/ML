from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import base64
import json
import numpy as np
from io import StringIO
from timeit import default_timer as timer
from PIL import Image
import datetime as dt
from random import randint

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from kafka import KafkaProducer
import cv2

import time
import collections
import vgconf

from core.services import YOLODetection

import tensorflow as tf


gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
config=tf.ConfigProto(gpu_options=gpu_options)
session = tf.Session(config=config)
tf.logging.set_verbosity(tf.logging.ERROR)


class Spark_Object_Detector():
    """Stream WebCam Images to Kafka Endpoint."""

    def __init__(self,
                 interval=10,
                 topic_to_consume='video',
                 topic_for_produce='resultstream',
                 kafka_endpoint='master:6667'):

        """ Initialize our yolo and firearm model"""

        self.detector = YOLODetection.YOLODetection()
        self.detector.enable_yolo_detection()

        """Initialize Spark & TensorFlow environment."""

        self.topic_to_consume = topic_to_consume
        self.topic_for_produce = topic_for_produce
        self.kafka_endpoint = kafka_endpoint

        # Create Kafka Producer for sending results
        self.producer = KafkaProducer(bootstrap_servers=kafka_endpoint)

        sc = SparkContext(appName='ObjectDetection')
        self.ssc = StreamingContext(sc, interval)  # , 3)

        # Make Spark logging less extensive
        log4jLogger = sc._jvm.org.apache.log4j
        log_level = log4jLogger.Level.ERROR
        log4jLogger.LogManager.getLogger('org').setLevel(log_level)
        log4jLogger.LogManager.getLogger('akka').setLevel(log_level)
        log4jLogger.LogManager.getLogger('kafka').setLevel(log_level)
        self.logger = log4jLogger.LogManager.getLogger(__name__)
        self.objects_detector_prediction = []
        self.objects_detected_view_text=""

    def _update_predictions(self):
        self.objects_detector_prediction = self.detector.get_yolo_prediction()

        self.detected_objects = []
        if self.objects_detector_prediction:
            self.detected_objects.extend(self.objects_detector_prediction)

        if self.detected_objects:
            self._update_detected_objects(self.detected_objects)

    def _update_detected_objects(self, objects_prediction):
        parsed_objects = [p['label'] for p in objects_prediction]
        parsed_objects_dict = collections.Counter(parsed_objects)
        objects = ''

        for (obj, count) in parsed_objects_dict.items():
            objects += '%s (%d) ' % (obj, count)

        self.objects_detected_view_text = objects


    def start_processing(self):
        """Start consuming from Kafka endpoint and detect objects."""
        kvs = KafkaUtils.createDirectStream(self.ssc,
                                            [self.topic_to_consume],
                                            {'metadata.broker.list': self.kafka_endpoint}
                                            )
        kvs.foreachRDD(self.handler)
        self.ssc.start()
        self.ssc.awaitTermination()


    def detect_objects(self, event):
        """Use Yolo and Incepiton Model to detect objects."""

        decoded = base64.b64decode(event['image'])

        # TODO: Picking unique filenames or find a way to send it to kafka

        filename = 'codev1frame.jpg'  # I assume you have a way of picking unique filenames
        with open(filename, 'wb') as f:
            f.write(decoded)
        img = cv2.imread(filename)

        # Prepare object for sending to endpoint
        result = {'timestamp': event['timestamp'],
                  'camera_id': event['camera_id'],
                  'image': self.get_box_plot(img),
                  'prediction': self.objects_detected_view_text
                  }
        return json.dumps(result)

    def get_box_plot(self,img):
        self.detector.detect(img)
        frame = self.detector.plot_objects(img)
        self._update_predictions()
        img_str = cv2.imencode('.jpeg', frame)[1]
        img_as_text = base64.b64encode(img_str).decode('utf-8')
        return img_as_text

    def handler(self, timestamp, message):
        """Collect messages, detect object and send to kafka endpoint."""
        records = message.collect()
        # For performance reasons, we only want to process the newest message
        # for every camera_id
        to_process = {}
        self.logger.info( '\033[3' + str(randint(1, 7)) + ';1m' +  # Color
            '-' * 25 +
            '[ NEW MESSAGES: ' + str(len(records)) + ' ]'
            + '-' * 25 +
            '\033[0m' # End color
            )
        dt_now = dt.datetime.now()
        for record in records:
            event = json.loads(record[1])
            self.logger.info('Received Message: ' +
                             event['camera_id'] + ' - ' + event['timestamp'])
            dt_event = dt.datetime.strptime(
                event['timestamp'], '%Y-%m-%dT%H:%M:%S.%f')
            delta = dt_now - dt_event
            print("timestamp = " + str(dt_event))
            if delta.seconds > 5:
                continue
            to_process[event['camera_id']] = event

        if len(to_process) == 0:
            self.logger.info('Skipping processing...')

        for key, event in to_process.items():
            self.logger.info('Processing Message: ' +
                             event['camera_id'] + ' - ' + event['timestamp'])
            start = timer()
            detection_result = self.detect_objects(event)
            self.logger.info('prediction: ' + self.objects_detected_view_text)
            end = timer()
            delta = end - start
            self.logger.info('Done after ' + str(delta) + ' seconds.')
            #self.producer.send(self.topic_for_produce, detection_result.encode('utf-8'))
            self.logger.info('Sent image to Kafka endpoint.')
            self.producer.flush()

if __name__ == '__main__':
    sod = Spark_Object_Detector(
        interval=1,
        topic_to_consume='video',
        topic_for_produce='resultstream',
        kafka_endpoint='master:6667')
    sod.start_processing()