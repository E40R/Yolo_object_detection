import cv2
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.names = self.model.names

    def detect(self, frame):
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({
                    "label": self.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": (x1, y1, x2, y2),
                })
        return detections

    def annotate(self, frame, detections):
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            label = d["label"]
            conf = d["confidence"]
            color = (0, 0, 220) if label == "person" else (0, 200, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = f"{label} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, tag, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return frame
