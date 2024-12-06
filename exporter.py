from sympy import simplify
from ultralytics import YOLO

# Load a model
model = YOLO("./yolo11n.pt")  # load an official model

# Export the model
model.export(format="openvino",int8=True,batch=4)