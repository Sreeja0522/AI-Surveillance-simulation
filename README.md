# Automated Surveillance Analysis Platform

An intelligent surveillance system and dashboard built with Python, YOLOv8, OpenCV, and Streamlit. The application performs real-time edge processing on ingested CCTV footage to track targets, extract visual attributes, and emit browser-level auditory alerts.

---

## Key Features

- **Real-Time Video Ingestion:** Direct file upload support for large `.mp4`, `.avi`, and `.mkv` files with automatic FPS synchronization to match native playback speed.
- **Selective YOLOv8 Tracking:** Tracks humans, road vehicles, and animals using ByteTrack. Class filters exclude non-relevant entities (such as trains) at the inference level.
- **Color Extraction & Deduplication:** Uses HSV color space analysis on vehicle bounding boxes to detect dominant colors (e.g., *white car*, *red truck*). Tracks assigned object IDs so each approaching vehicle triggers a voice alert only once.
- **Browser-Level Audio Notifications:** Dispatches client-side voice alerts via the Web Speech API without relying on host-machine audio drivers.
- **HUD Toast Overlays:** Non-blocking, vanishing toast notifications in the top corner that keep the video feed visible.
- **Playback Controls:** Interactive controls including Play, Pause, ±5s jump buttons, auto-pause on video completion, and a timeline scrubber calibrated in seconds (`0.0s` to end).

---

## Project Structure

```text
├── app.py              # Main Streamlit application and inference pipeline
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation

---

## Installation & Setup
Clone the repository and enter the directory:

Create and activate a virtual environment (recommended):

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
Running the Application
Launch the Streamlit dashboard with an expanded upload limit to support large CCTV clips:

Run the cmd: 

streamlit run app.py --server.maxUploadSize=1000