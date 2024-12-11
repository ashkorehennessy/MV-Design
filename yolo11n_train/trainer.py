from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="yolo11n_train/face.yaml",
                      epochs=100,
                      imgsz=224,
                      amp=False,
                      batch=128,
                      int8=True)