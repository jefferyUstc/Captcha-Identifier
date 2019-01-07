#! /usr/bin/python
# _*_ coding: utf-8 _*_
__author__ = 'Jeffery'
__date__ = '2019/1/6 20:18'

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf
import numpy as np
import grpc
import os
import cv2
from constant import characters
from utils import get_captcha

def process_img(img_path):
    """
    pre-process img to numpy
    :param img_path: path of img
    :return:
    """
    if os.path.exists(img_path):
        img = cv2.imread(img_path, 0)  # (100, 120)
    else:
        raise FileNotFoundError('img to process not exists')
    img[img < 193] = 0
    img[img >= 193] = 1
    img = np.reshape(img, [1, img.shape[0], img.shape[1], 1])
    float_img = img.astype(np.float32)
    return float_img


def request_server(img_np,
                   server_url,
                   model_name,
                   signature_name,
                   input_name,
                   output_name
                   ):
    """
    below info about model
    :param model_name:
    :param signature_name:
    :param output_name:
    :param input_name:

    :param img_np: processed img , numpy.ndarray type [h,w,c]
    :param server_url: TensorFlow Serving url,str type,e.g.'0.0.0.0:8500'
    :return: type numpy array
    """
    # connect channel
    channel = grpc.insecure_channel(server_url)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # set up request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name  # request.model_spec.version = "1"
    request.model_spec.signature_name = signature_name
    request.inputs[input_name].CopyFrom(
        tf.contrib.util.make_tensor_proto(img_np, shape=list(img_np.shape)))
    # get response
    response = stub.Predict(request, 5.0)
    # res_from_server_np = np.asarray(response.outputs[output_name].float_val)
    res_from_server_np = tf.make_ndarray(response.outputs[output_name])
    
    s = ''
    for character in res_from_server_np[0]:
        s += characters[character]
    return get_captcha(res_from_server_np, characters)


if __name__ == '__main__':
    img = process_img('./imgs/aaau_1.png')
    res = request_server(img, '0.0.0.0:8500', 'mymodel',
                         'prediction_signature', "images",
                         "result")
    print(res)
