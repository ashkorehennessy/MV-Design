from sympy import simplify
from ultralytics import YOLO

# Load a model
model = YOLO("./best.pt")  # load an official model

# Export the model
model.export(format="ncnn",imgsz=160,int8=True,data="yolo11n_train/face.yaml")