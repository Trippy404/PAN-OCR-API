
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data='dataset/data.yaml',   # path to your yaml
    epochs=30,
    imgsz=640,
    batch=16,
    name='pan_card_model'
)
