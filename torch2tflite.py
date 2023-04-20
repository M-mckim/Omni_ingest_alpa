"""
    ref:
    https://github.com/sithu31296/PyTorch-ONNX-TFLite
    https://velog.io/@kcw4875/Pytorch%EC%97%90%EC%84%9C-TFLite%EB%A1%9C-%EB%B3%80%ED%99%98%ED%95%98%EA%B8%B0
"""
import torch
import keras
import tensorflow as tf


class torch2tflite:
    def __init__(self, model):
        self.model = model
        
    def conver_to_tflite(model):
        # Convert the model.
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # Save the TF Lite model.
        with open('model.tflite', 'wb') as f:
            f.write(tflite_model)