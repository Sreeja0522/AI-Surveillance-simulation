
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from ultralytics import YOLO
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Surveillance Monitoring System",
    layout="wide"
)

# ============================================================
# DARK SURVEILLANCE CONSOLE THEME
# ============================================================
st.markdown("""
<style>
    /* Global Dark Theme */
    .stApp {
        background-color: #0B1120;
        color: #E5E7EB;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1F2937;
    }

    /* Typography */
    .system-title {
        font-size: 1.35rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
    }

    .system-subtitle {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-bottom: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Streamlit Metrics */
    div[data-testid="stMetricValue"] {
        color: #F8FAFC;
        font-size: 1.5rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #94A3B8;
    }

    /* Buttons */
    .stButton > button {
        background-color: #172033;
        color: #E5E7EB;
        border: 1px solid #334155;
    }

    .stButton > button:hover {
        background-color: #1E293B;
        color: #F8FAFC;
        border-color: #3B82F6;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background-color: #111827;
        border-color: #334155;
    }

    /* Slider */
    .stSlider label {
        color: #CBD5E1;
    }

    /* Horizontal Divider */
    hr {
        border-color: #1F2937;
    }

    /* Fixed Corner Toast Overlay */
    #toast-container {
        position: fixed;
        top: 24px;
        right: 24px;
        z-index: 999999;
        display: flex;
        flex-direction: column;
        gap: 8px;
        pointer-events: none;
    }

    .hud-toast {
        min-width: 250px;
        max-width: 320px;
        background: #111827;
        color: #F8FAFC;
        padding: 10px 14px;
        border-radius: 6px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
        border: 1px solid #1F2937;
        border-left-width: 4px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: 0.82rem;
        line-height: 1.35;
        animation: fadeIn 0.25s ease-out, fadeOut 0.5s ease-in 3.5s forwards;
    }

    .hud-toast.vehicle {
        border-left-color: #F43F5E;
    }

    .hud-toast.system {
        border-left-color: #22C55E;
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(-8px);
        }

        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeOut {
        from {
            opacity: 1;
        }

        to {
            opacity: 0;
        }
    }
</style>

<div id="toast-container"></div>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="system-title">Automated Surveillance Analysis Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="system-subtitle">Real-Time Ingestion & Edge Detection Console</div>',
    unsafe_allow_html=True
)


# ============================================================
# YOLO MODEL
# ============================================================
@st.cache_resource
def load_yolo():
    return YOLO("yolov8n.pt")


model = load_yolo()


# ============================================================
# DETECTION CLASSES
# ============================================================
ROAD_VEHICLE_NAMES = {
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle"
}

HUMAN_NAMES = {
    "person"
}

ANIMAL_NAMES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe"
}

TARGET_NAMES = (
    ROAD_VEHICLE_NAMES
    .union(HUMAN_NAMES)
    .union(ANIMAL_NAMES)
)

ALLOWED_CLASS_IDS = [
    cls_id
    for cls_id, name in model.names.items()
    if name in TARGET_NAMES and name != "train"
]


# ============================================================
# VEHICLE COLOR DETECTION
# ============================================================
def get_dominant_color(bgr_crop):

    if (
        bgr_crop.size == 0
        or bgr_crop.shape[0] < 5
        or bgr_crop.shape[1] < 5
    ):
        return "vehicle"

    h, w, _ = bgr_crop.shape

    center_crop = bgr_crop[
        int(h * 0.25):int(h * 0.75),
        int(w * 0.25):int(w * 0.75)
    ]

    if center_crop.size == 0:
        center_crop = bgr_crop

    hsv = cv2.cvtColor(
        center_crop,
        cv2.COLOR_BGR2HSV
    )

    mean_s = np.mean(hsv[:, :, 1])
    mean_v = np.mean(hsv[:, :, 2])

    if mean_v < 45:
        return "blue"

    if mean_s < 38 and mean_v > 175:
        return "white"

    if mean_s < 45:
        return "gray"

    mean_h = np.mean(hsv[:, :, 0])

    if mean_h < 10 or mean_h > 165:
        return "red"

    elif 10 <= mean_h < 25:
        return "orange"

    elif 25 <= mean_h < 35:
        return "yellow"

    elif 35 <= mean_h < 85:
        return "green"

    elif 85 <= mean_h < 135:
        return "blue"

    else:
        return "purple"


# ============================================================
# EVENT / TOAST SYSTEM
# ============================================================
def dispatch_event(
    message: str,
    category: str = "system",
    speak: bool = False
):

    speech_js = f"""
        const utterance = new SpeechSynthesisUtterance("{message}");
        utterance.rate = 1.0;
        window.speechSynthesis.speak(utterance);
    """ if speak else ""

    js_code = f"""
    <script>
        const container =
            window.parent.document.getElementById("toast-container");

        if (container) {{
            const toast =
                window.parent.document.createElement("div");

            toast.className = "hud-toast {category}";
            toast.innerText = "{message}";

            container.appendChild(toast);

            setTimeout(() => {{
                if (toast.parentNode) {{
                    toast.parentNode.removeChild(toast);
                }}
            }}, 4000);
        }}

        {speech_js}
    </script>
    """

    components.html(
        js_code,
        height=0,
        width=0
    )


# ============================================================
# TIME FORMATTER
# ============================================================
def format_time_str(seconds: float) -> str:

    mins = int(seconds // 60)
    secs = int(seconds % 60)
    frac = int(
        (seconds - int(seconds)) * 10
    )

    return f"{mins:02d}:{secs:02d}.{frac}"


# ============================================================
# SESSION STATE
# ============================================================
if "is_playing" not in st.session_state:
    st.session_state.is_playing = True

if "current_time_sec" not in st.session_state:
    st.session_state.current_time_sec = 0.0

if "last_video_name" not in st.session_state:
    st.session_state.last_video_name = ""

if "alerted_vehicle_ids" not in st.session_state:
    st.session_state.alerted_vehicle_ids = set()


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(
    "**Source Media & Parameters**"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Target Video",
    type=[
        "mp4",
        "avi",
        "mov",
        "mkv"
    ]
)

conf_thresh = st.sidebar.slider(
    "Confidence Threshold",
    0.20,
    0.90,
    0.40,
    0.05
)


# ============================================================
# RESET ALERT REGISTRY
# ============================================================
if st.sidebar.button(
    "Reset Alert Registry",
    use_container_width=True
):

    st.session_state.alerted_vehicle_ids.clear()

    dispatch_event(
        "Alert registry cleared",
        category="system",
        speak=False
    )


# ============================================================
# VIDEO FRAME WINDOW
# ============================================================
frame_window = st.empty()


# ============================================================
# CONTROLLER TOOLBAR
# ============================================================
c1, c2, c3, c4, c5 = st.columns(
    [1, 1, 1, 1, 4]
)

with c1:

    if st.button(
        "Play",
        use_container_width=True
    ):
        st.session_state.is_playing = True


with c2:

    if st.button(
        "Pause",
        use_container_width=True
    ):
        st.session_state.is_playing = False


with c3:

    if st.button(
        "-5s",
        use_container_width=True
    ):
        st.session_state.current_time_sec = max(
            0.0,
            st.session_state.current_time_sec - 5.0
        )


with c4:

    if st.button(
        "+5s",
        use_container_width=True
    ):
        st.session_state.current_time_sec += 5.0


with c5:

    slider_slot = st.empty()


# ============================================================
# METRICS
# ============================================================
st.markdown("---")

col_h, col_v, col_a = st.columns(3)

with col_h:
    h_metric = st.empty()

with col_v:
    v_metric = st.empty()

with col_a:
    a_metric = st.empty()


# ============================================================
# FRAME PROCESSING
# ============================================================
def process_frame(frame):

    results = model.track(
        frame,
        conf=conf_thresh,
        classes=ALLOWED_CLASS_IDS,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    humans_found = 0
    vehicles_found = 0
    animals_found = 0

    new_vehicles_detected = []


    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------
    if results[0].boxes:

        boxes = (
            results[0]
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )

        classes = (
            results[0]
            .boxes
            .cls
            .cpu()
            .numpy()
        )

        track_ids = (
            results[0]
            .boxes
            .id
            .cpu()
            .numpy()
            if results[0].boxes.id is not None
            else [None] * len(boxes)
        )


        for box, cls_id, track_id in zip(
            boxes,
            classes,
            track_ids
        ):

            label = model.names[int(cls_id)]

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # =================================================
            # PERSON
            # =================================================
            if label in HUMAN_NAMES:

                humans_found += 1

                # Cyan detection box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 180, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "person",
                    (
                        x1,
                        max(18, y1 - 6)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 180, 255),
                    1,
                    cv2.LINE_AA
                )


            # =================================================
            # ANIMAL
            # =================================================
            elif label in ANIMAL_NAMES:

                animals_found += 1

                # Neutral gray detection box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (180, 180, 180),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (
                        x1,
                        max(18, y1 - 6)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA
                )


            # =================================================
            # ROAD VEHICLE
            # =================================================
            elif label in ROAD_VEHICLE_NAMES:

                vehicles_found += 1

                vehicle_crop = frame[
                    max(0, y1):max(0, y2),
                    max(0, x1):max(0, x2)
                ]

                color_name = get_dominant_color(
                    vehicle_crop
                )

                track_label = (
                    f"{color_name} {label}"
                )


                if track_id is not None:

                    track_id = int(track_id)

                    display_tag = (
                        f"ID:{track_id} {track_label}"
                    )

                    if (
                        track_id
                        not in st.session_state.alerted_vehicle_ids
                    ):

                        st.session_state.alerted_vehicle_ids.add(
                            track_id
                        )

                        new_vehicles_detected.append(
                            track_label
                        )

                else:

                    display_tag = track_label


                # Green vehicle detection box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 220, 120),
                    2
                )

                cv2.putText(
                    frame,
                    display_tag,
                    (
                        x1,
                        max(18, y1 - 6)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 220, 120),
                    2,
                    cv2.LINE_AA
                )


    # =========================================================
    # UPDATE METRICS
    # =========================================================
    h_metric.metric(
        "Humans Detected",
        humans_found
    )

    v_metric.metric(
        "Road Vehicles",
        vehicles_found
    )

    a_metric.metric(
        "Animals",
        animals_found
    )


    # =========================================================
    # VEHICLE ALERTS
    # =========================================================
    if new_vehicles_detected:

        for vehicle_desc in new_vehicles_detected:

            dispatch_event(
                f"Vehicle Approaching: {vehicle_desc}",
                category="vehicle",
                speak=True
            )


    # =========================================================
    # DISPLAY FRAME
    # =========================================================
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frame_window.image(
        frame_rgb,
        channels="RGB",
        use_container_width=True
    )


# ============================================================
# VIDEO INPUT
# ============================================================
if uploaded_file is not None:

    # --------------------------------------------------------
    # Detect new video
    # --------------------------------------------------------
    if (
        st.session_state.last_video_name
        != uploaded_file.name
    ):

        st.session_state.last_video_name = (
            uploaded_file.name
        )

        st.session_state.current_time_sec = 0.0

        st.session_state.is_playing = True

        st.session_state.alerted_vehicle_ids.clear()

        dispatch_event(
            "Video stream initialized",
            category="system",
            speak=False
        )


    # --------------------------------------------------------
    # Temporary video file
    # --------------------------------------------------------
    temp_path = os.path.join(
        tempfile.gettempdir(),
        f"surveillance_{uploaded_file.name}"
    )

    with open(
        temp_path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )


    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------
    cap = cv2.VideoCapture(
        temp_path
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    fps = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 25.0
    )

    duration_sec = (
        total_frames / fps
    )


    # --------------------------------------------------------
    # Validate video
    # --------------------------------------------------------
    if (
        total_frames > 0
        and duration_sec > 0
    ):

        st.session_state.current_time_sec = min(
            max(
                0.0,
                st.session_state.current_time_sec
            ),
            duration_sec
        )


        # =====================================================
        # PLAYBACK SCRUBBER
        # =====================================================
        seek_sec = slider_slot.slider(
            "Playback Timeline",
            min_value=0.0,
            max_value=round(
                duration_sec,
                2
            ),
            value=round(
                st.session_state.current_time_sec,
                2
            ),
            step=0.1,
            format="%.1fs",
            key="time_scrubber",
            label_visibility="collapsed"
        )


        if (
            abs(
                seek_sec
                - st.session_state.current_time_sec
            ) > 0.15
        ):

            st.session_state.current_time_sec = seek_sec


        frame_duration = 1.0 / fps


        # =====================================================
        # PLAYING
        # =====================================================
        if st.session_state.is_playing:

            start_frame = int(
                st.session_state.current_time_sec
                * fps
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                start_frame
            )


            while (
                cap.isOpened()
                and st.session_state.is_playing
            ):

                loop_start = time.time()

                ret, frame = cap.read()


                # ------------------------------------------------
                # Pause at video end
                # ------------------------------------------------
                if (
                    not ret
                    or st.session_state.current_time_sec
                    >= duration_sec
                ):

                    st.session_state.is_playing = False

                    st.session_state.current_time_sec = (
                        duration_sec
                    )

                    dispatch_event(
                        "Video playback complete",
                        category="system",
                        speak=False
                    )

                    break


                # ------------------------------------------------
                # Process current frame
                # ------------------------------------------------
                process_frame(frame)


                # ------------------------------------------------
                # Advance timestamp
                # ------------------------------------------------
                st.session_state.current_time_sec += (
                    frame_duration
                )


                # ------------------------------------------------
                # Real-time FPS synchronization
                # ------------------------------------------------
                elapsed = (
                    time.time()
                    - loop_start
                )

                sleep_time = (
                    frame_duration
                    - elapsed
                )

                if sleep_time > 0:
                    time.sleep(
                        sleep_time
                    )


        # =====================================================
        # PAUSED
        # =====================================================
        else:

            target_frame = int(
                st.session_state.current_time_sec
                * fps
            )

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                target_frame
            )

            ret, frame = cap.read()

            if ret:
                process_frame(frame)


    # --------------------------------------------------------
    # Invalid video
    # --------------------------------------------------------
    else:

        st.error(
            "Invalid media container or unsupported codec format."
        )


    cap.release()


# ============================================================
# NO VIDEO
# ============================================================
else:

    frame_window.info(
        "Upload video file via the sidebar to initiate analysis."
    )

