 # Object detection using kafka and spark-streaming
 
 ## How to use
	Download and Install Kafka and Spark
	
	https://kafka.apache.org/quickstart
	https://spark.apache.org/downloads.html
	
	The following dependencies are needed to run the tracker:

	NumPy
	sklearn
	OpenCV
	Pillow
	TensorFlow
	Kafka
    flask
	Pyspark
	
	put your yolo weight file to data/classifiers/YOLO folder, you can download from here : https://pjreddie.com/darknet/yolo/

    Run traker with cmd :

	python producer.py
	python spark-submit --jars spark-streaming-kafka-0-8-assembly_2.11-2.3.3.jar detection.py
    python consumer.py
