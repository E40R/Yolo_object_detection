import csv
import os
from datetime import datetime


class DetectionLogger:
    def __init__(self, filepath="detections.csv"):
        self.filepath = filepath
        self._ensure_header()

    def _ensure_header(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "Time", "Event", "Object", "Confidence"])

    def log_event(self, event_type, label, confidence=0.0):
        """Log a single event: 'entered' or 'left'."""
        now = datetime.now()
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                now.strftime("%Y-%m-%d"),
                now.strftime("%H:%M:%S"),
                event_type,
                label,
                f"{confidence:.3f}" if confidence else "",
            ])

    def read_all(self):
        rows = []
        if not os.path.exists(self.filepath):
            return rows
        with open(self.filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows
