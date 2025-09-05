# Driver Drowsiness Detection and Alert System  

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)  
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)  
![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-orange.svg)  
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  

A **real-time computer vision system** that detects driver drowsiness using a webcam.  
It monitors **eye closure (EAR)** and **head posture (pitch)**, and triggers an **audible alarm** when drowsiness is detected, helping prevent accidents and improve road safety.  

---

## 🚀 Features
- Real-time **face landmark detection** (MediaPipe Face Mesh)  
- **Eye Aspect Ratio (EAR)**–based eye-closure detection  
- **Head pose estimation (pitch)** via solvePnP  
- **Audible alarm** using `pygame` when drowsiness is detected  
- On-screen overlays: landmarks, EAR, pitch, FPS, and alarm status  
- Configurable thresholds for different users/environments  
- Includes **single-file script** for easy testing/demo  

---

🛠 Tech Stack
Language: Python 3.9+

Libraries:
OpenCV
MediaPipe
NumPy
pygame

---

📂 Project Structure

driver-drowsiness-detection/
│
├── main.py # Entry point (multi-file version)
├── drowsiness_detector.py # Detection logic
├── alarm_system.py # Audio alert system
├── single_file_drowsiness.py # All-in-one version
├── requirements.txt # Dependencies
└── README.md # Documentation

---

⚙️ How It Works
Video Capture – Opens webcam at 640×480 resolution.
Face & Landmark Detection – MediaPipe Face Mesh detects facial landmarks.
Eye Closure Detection – EAR < threshold → eyes considered closed.
Head Pose Detection – Pitch angle from solvePnP → detect head-down posture.
Alarm – Audible beep plays continuously when drowsy state detected.
Visualization – Overlays EAR, pitch, FPS, and alarm status.

---

📋 Requirements
Python 3.9+
A working webcam
Windows, macOS, or Linux

Dependencies (see requirements.txt):
opencv-python
mediapipe
numpy
pygame

---

⚡ Installation (Windows Example)
1.(Optional) Create a virtual environment :
python -m venv .venv
.venv\Scripts\activate

2.Install dependencies :
pip install -r requirements.txt

---

▶️ Usage
Run the project:
python main.py

Or use the single-file version:
python single_file_drowsiness.py

➡️ Press q to quit.

---

🔧 Configuration
Tune thresholds in drowsiness_detector.py (or in single_file_drowsiness.py):
Eye Closure:
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 10
Head Pose:
HEAD_DOWN_THRESH = 15

---

🛠 Troubleshooting
Webcam not opening → Ensure no other app is using it, try cv2.VideoCapture(1).
pygame sound not working → Check audio device; not supported in headless mode.

Installation errors → Upgrade pip and reinstall:
pip install --upgrade pip
pip install --force-reinstall mediapipe opencv-python

---

