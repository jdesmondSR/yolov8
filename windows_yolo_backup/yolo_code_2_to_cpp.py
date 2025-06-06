# -*- coding: utf-8 -*-
"""

training data path


"""

from ultralytics import YOLO
#import torch


if __name__ == "__main__":
    #data_path = "C:\Users\jdesm\OneDrive\Desktop\Yolo_Domumentation\yolo_work\training_data_1\rescueLineImages100.v1i.yolov8\data.yaml"
    data_path = "C:/Users/jdesm/OneDrive/Desktop/Yolo_Domumentation/yolo_work/training_data_1/rescueLineImages100.v1i.yolov8/data.yaml"

    print("a")
    # using a model from yolo
    #model = YOLO('yolov8n.yaml') #blank yolo from scratch...
    #model = YOLO("model100_NO_pretrain.pt")
    model = YOLO("majic_modelv1.pt")

    

    # can use tflite, onnx format doesnt seem to work...
    #model.export(format = "tflite") #creates an onnx file for use in cpp
    model.export(format = "tflite", imgsz = 320)


    print("b")

    # device = 0 means use the GPU
    #results = model.train(data = data_path, epochs = 100, imgsz = 320, device = 0)
    print("c")
    #model.save("model100.pt")
    print("d")
