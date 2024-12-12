from sympy import simplify
from ultralytics import YOLO

# Load a model
model = YOLO("../yolo11_128.pt")  # load an official model

# Export the model
model.export(format="ncnn",imgsz=160,int8=True,data="face.yaml")