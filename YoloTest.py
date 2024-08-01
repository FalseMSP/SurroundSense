from ultralytics import YOLO
import torch

try:
    # Force CPU usage if GPU might be an issue
    device = torch.device('cpu')
    model = YOLO("yolov8n.pt").to(device)
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")

