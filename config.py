MODEL_PATH = "yolov8n.pt"
CONFIDENCE = 0.55
CAMERA_INDEX = 0

ALERT_COOLDOWN = 30  # seconds before re-alerting same class after it leaves + returns
ALERT_CLASSES = {
    "person", "car", "truck", "dog", "cat", "bicycle", "motorcycle", "bus",
    "bottle", "cup", "cell phone", "laptop", "keyboard", "mouse", "remote",
    "book", "backpack", "handbag", "suitcase", "umbrella", "chair",
    "scissors", "knife", "clock", "vase",
}

LOG_FILE = "detections.csv"
