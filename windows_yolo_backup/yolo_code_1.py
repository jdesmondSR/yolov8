# -*- coding: utf-8 -*-
"""

training data path


"""

#import torch
#import torchvision
#print("PyTorch Version:", torch.__version__)
#print("CUDA Available:", torch.cuda.is_available())
#print("CUDA Version:", torch.version.cuda)
#print("torchvision Version:", torchvision.__version__)


from ultralytics import YOLO
import torch


if __name__ == "__main__":
    #data_path = "C:\Users\jdesm\OneDrive\Desktop\Yolo_Domumentation\yolo_work\training_data_1\rescueLineImages100.v1i.yolov8\data.yaml"
    #data_path = "C:/Users/jdesm/OneDrive/Desktop/Yolo_Domumentation/yolo_work/training_data_1/rescueLineImages100.v2i.yolov8/data.yaml"
    data_path = "C:/Users/jdesm/OneDrive/Desktop/Yolo_Domumentation/yolo_work/training_data_1/MAJic_images.v2i.yolov8/data.yaml"
    #"C:\Users\jdesm\OneDrive\Desktop\Yolo_Domumentation\yolo_work\training_data_1\MAJic_images.v2i.yolov8\data.yaml"
    
    #"C:\Users\jdesm\OneDrive\Desktop\Yolo_Domumentation\Rescue_Line_test.v2i.yolov8\data.yaml"

    print("a")
    # using a model from yolo
    model = YOLO('yolov8n.yaml') #blank yolo from scratch...


    print("b")

    # device = 0 means use the GPU
    results = model.train(data = data_path,
                          imgsz = 320,
                          device = 0,
                          pretrained=False)
    print("c")
    model.save("majic_modelv1.pt")
    print("d")
