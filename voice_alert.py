import threading
import time
import pyttsx3


class VoiceAlert:
    def __init__(self, cooldown=30):
        self.cooldown = cooldown
        self._last_alert = {}
        self._present = set()
        self._absent_count = {}
        self._lock = threading.Lock()
        self.GONE_FRAMES = 15  # frames without seeing it = gone

    def update(self, detected_labels):
        """Call once per frame with labels currently visible.
        Returns (entered, left) — lists of labels that just appeared or disappeared."""
        now = time.time()
        current = set(detected_labels)
        entered = []
        left = []

        with self._lock:
            # new objects entering frame
            for label in current - self._present:
                last = self._last_alert.get(label, 0)
                if now - last >= self.cooldown:
                    self._last_alert[label] = now
                    threading.Thread(target=self._speak, args=(f"{label} detected",), daemon=True).start()
                entered.append(label)

            # track objects leaving — need N consecutive absent frames
            for label in list(self._present):
                if label not in current:
                    self._absent_count[label] = self._absent_count.get(label, 0) + 1
                    if self._absent_count[label] >= self.GONE_FRAMES:
                        self._present.discard(label)
                        self._absent_count.pop(label, None)
                        left.append(label)
                else:
                    self._absent_count.pop(label, None)

            self._present = self._present | current

        return entered, left

    @staticmethod
    def _speak(text):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
