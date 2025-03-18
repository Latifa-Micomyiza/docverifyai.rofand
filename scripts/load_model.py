from roboflow import Roboflow
from ultralytics import YOLO


from roboflow import Roboflow
rf = Roboflow(api_key="YLkMrPxIwMqDTL0dxyCh")
project = rf.workspace("rofand").project("rofandv1")
version = project.version(4)
dataset = version.download("yolov8")

# Load the model
model = YOLO("yolov8s.pt")

# Train the model
model.train(data=f"{dataset.location}/data.yaml", epochs=25, imgsz=800,batch=8, plots=True)
