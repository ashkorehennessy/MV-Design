from sympy import simplify
from ultralytics import YOLO

# Load a model
model = YOLO("./yolo11_640.pt")  # load an official model

# Export the model
model.export(format="onnx",data="yolo11n_train/face.yaml")