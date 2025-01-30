import torch
import os
from PIL import Image

# YOLO model path
YOLO_MODEL_PATH = r'models/yolov5s.pt'

# Download YOLO model if it does not exist
def download_yolo_model():
    if not os.path.exists(YOLO_MODEL_PATH):
        print("YOLO model not found. Downloading...")
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        torch.save(yolo_model.state_dict(), YOLO_MODEL_PATH)
        print("YOLO model downloaded and saved.")
    else:
        print("YOLO model found. Using the existing model.")


def load_yolo_model():
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo_model.load_state_dict(torch.load(YOLO_MODEL_PATH, map_location=torch.device('cpu')))
    return yolo_model

def detect_cat_with_yolo(image_path, yolo_model):

    img = Image.open(image_path)
    img = img.convert("RGB")

    results = yolo_model(img)

    labels = results.names
    detected_classes = results.xywh[0][:, -1].cpu().numpy()

    # Check if 'cat' /YOLO label for cat is 15
    if 15 in detected_classes:
        return True
    else:
        return False
