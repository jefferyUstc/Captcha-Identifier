docker pull tensorflow/serving:latest
docker run -p 8500:8500 --mount type=bind, source=/home/jeffery/test/tf-serving/mymodel,\
target=/models/mymodel -e MODEL_NAME=mymodel d42952c6f8a6