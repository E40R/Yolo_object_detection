import csv
import os
import shutil
from datetime import datetime


HISTORY_DIR = "history"


class DetectionLogger:
    def __init__(self, filepath="detections.csv"):
        self.filepath = filepath
        self.session_start = datetime.now()
        self._archive_previous()
        self._start_fresh()

    def _archive_previous(self):
        """Move the old CSV into history/ before starting a new session."""
        if not os.path.exists(self.filepath):
            return
        os.makedirs(HISTORY_DIR, exist_ok=True)
        # name it by the first entry's date or fallback to modified time
        ts = datetime.fromtimestamp(os.path.getmtime(self.filepath))
        name = f"session_{ts.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        dest = os.path.join(HISTORY_DIR, name)
        if not os.path.exists(dest):
            shutil.move(self.filepath, dest)

    def _start_fresh(self):
        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Time", "Event", "Object", "Confidence"])

    def log_event(self, event_type, label, confidence=0.0):
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

    def read_current(self):
        return self._read_csv(self.filepath)

    def list_history(self):
        if not os.path.exists(HISTORY_DIR):
            return []
        files = sorted(os.listdir(HISTORY_DIR), reverse=True)
        return [f for f in files if f.endswith(".csv")]

    def read_history(self, filename):
        path = os.path.join(HISTORY_DIR, filename)
        return self._read_csv(path)

    @staticmethod
    def _read_csv(path):
        rows = []
        if not os.path.exists(path):
            return rows
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows
