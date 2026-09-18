import streamlit as st
import cv2
import time
from datetime import datetime
from collections import deque, Counter
from PIL import Image

from config import MODEL_PATH, CONFIDENCE, CAMERA_INDEX, ALERT_COOLDOWN, ALERT_CLASSES, LOG_FILE
from detector import ObjectDetector
from voice_alert import VoiceAlert
from detection_logger import DetectionLogger

st.set_page_config(page_title="AI Surveillance Robot", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ AI Surveillance Robot")
st.caption("Real-time object detection — logs only when something changes")

# -- sidebar --
with st.sidebar:
    st.header("Controls")
    confidence = st.slider("Detection confidence", 0.1, 1.0, CONFIDENCE, 0.05)
    all_classes = sorted([
        "person", "car", "truck", "dog", "cat", "bicycle", "motorcycle", "bus",
        "bottle", "cup", "cell phone", "laptop", "keyboard", "mouse", "remote",
        "book", "backpack", "handbag", "suitcase", "umbrella", "chair",
        "scissors", "knife", "clock", "vase", "bird", "teddy bear",
        "sports ball", "banana", "apple", "orange", "pizza",
    ])
    alert_classes = st.multiselect("Alert on these objects", all_classes,
                                   default=sorted(ALERT_CLASSES))
    voice_on = st.toggle("Voice alerts", value=True)
    cooldown = st.slider("Alert cooldown (sec)", 1, 60, ALERT_COOLDOWN)
    cam_idx = st.number_input("Camera index", 0, 10, CAMERA_INDEX)
    st.divider()
    run = st.toggle("▶ START SURVEILLANCE", value=False)
    st.divider()
    if st.button("Clear detection log"):
        import os
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
            st.success("Log cleared")

# -- metrics --
c1, c2, c3, c4 = st.columns(4)
ph_frames = c1.empty()
ph_objects = c2.empty()
ph_alerts = c3.empty()
ph_fps = c4.empty()

# -- video + live alerts --
col_vid, col_log = st.columns([3, 1])
video_ph = col_vid.empty()
log_ph = col_log.empty()

# -- detection log section --
st.divider()
log_header = st.columns([2, 1, 1])
log_header[0].subheader("Detection Log")
log_summary_ph = log_header[1].empty()
log_download_ph = log_header[2].empty()

tab_table, tab_stats = st.tabs(["Event Log", "Summary Stats"])
table_ph = tab_table.empty()
stats_ph = tab_stats.empty()


def show_log_ui(logger):
    rows = logger.read_all()
    if not rows:
        table_ph.info("No events recorded yet. Start surveillance to begin.")
        return

    recent = list(reversed(rows[-100:]))
    log_summary_ph.caption(f"{len(rows)} events recorded")

    try:
        with open(LOG_FILE, "r") as f:
            csv_data = f.read()
        log_download_ph.download_button("Download CSV", csv_data, "detections.csv", "text/csv",
                                        use_container_width=True)
    except Exception:
        pass

    table_ph.dataframe(
        recent,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn("Date"),
            "Time": st.column_config.TextColumn("Time"),
            "Event": st.column_config.TextColumn("Event"),
            "Object": st.column_config.TextColumn("Object"),
            "Confidence": st.column_config.TextColumn("Confidence"),
        },
    )

    counts = Counter(r["Object"] for r in rows if r.get("Event") == "entered")
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])

    stats_col1, stats_col2 = stats_ph.columns(2)
    with stats_col1:
        st.markdown("**Objects by frequency**")
        for obj, count in sorted_counts:
            st.markdown(f"- **{obj}**: {count} times")
    with stats_col2:
        st.markdown("**Activity by date**")
        dates = Counter(r["Date"] for r in rows)
        for date, count in sorted(dates.items(), reverse=True)[:7]:
            st.markdown(f"- {date}: **{count}** events")


if run:
    detector = ObjectDetector(MODEL_PATH, confidence)
    voice = VoiceAlert(cooldown) if voice_on else None
    logger = DetectionLogger(LOG_FILE)

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        st.error("Could not open camera. Check your camera index.")
        st.stop()

    frame_count = 0
    alert_count = 0
    recent_alerts = deque(maxlen=12)
    t_start = time.time()
    last_log_refresh = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera feed lost.")
            break

        detections = detector.detect(frame)
        frame = detector.annotate(frame, detections)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, ts, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        frame_count += 1

        # only get labels that are in our alert list
        frame_labels = [d["label"] for d in detections if d["label"] in alert_classes]
        best_conf = {}
        for d in detections:
            if d["label"] in alert_classes:
                old = best_conf.get(d["label"], 0)
                if d["confidence"] > old:
                    best_conf[d["label"]] = d["confidence"]

        if voice:
            entered, left = voice.update(frame_labels)
        else:
            entered, left = [], []

        # log only changes to CSV and UI
        for label in entered:
            conf = best_conf.get(label, 0)
            logger.log_event("entered", label, conf)
            alert_count += 1
            recent_alerts.appendleft(
                f"**{label}** detected ({conf:.0%}) — {datetime.now().strftime('%H:%M:%S')}"
            )

        for label in left:
            logger.log_event("left", label)
            recent_alerts.appendleft(
                f"*{label} left frame* — {datetime.now().strftime('%H:%M:%S')}"
            )

        # metrics
        elapsed = time.time() - t_start
        fps = frame_count / elapsed if elapsed > 0 else 0
        ph_frames.metric("Frames", frame_count)
        ph_objects.metric("In Frame", len(frame_labels))
        ph_alerts.metric("Events", alert_count)
        ph_fps.metric("FPS", f"{fps:.1f}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_ph.image(Image.fromarray(frame_rgb), use_container_width=True)

        if recent_alerts:
            log_ph.markdown("**Live Feed**\n\n" + "\n\n".join(recent_alerts))

        if time.time() - last_log_refresh > 3:
            show_log_ui(logger)
            last_log_refresh = time.time()

    cap.release()
    st.info("Surveillance stopped.")
else:
    video_ph.info("Toggle **START SURVEILLANCE** in the sidebar to begin.")
    logger = DetectionLogger(LOG_FILE)
    show_log_ui(logger)
