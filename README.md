# 🌌 Zenith AI - Video Intelligence Pro
> **High-Performance Multimodal Video Analysis & Narrative Synthesis**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Bsh54/Zenith-AI/blob/main/main.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### 🌐 Language / Langue
**English** | [🇫🇷 Voir le README en Français](./README_FR.md)

---

## ✨ Overview
**Zenith AI** is a cutting-edge multimodal intelligence system designed to "understand" video content like a human would. By combining Computer Vision (YOLOv8), Speech-to-Text (Whisper), and Large Language Models (LLM), it transforms any video or URL into a structured, professional narrative report.

### 🚀 Key Features
- **🎥 Universal Input**: Upload local files or paste links (YouTube, TikTok, Twitter, etc.).
- **👁️ Vision Intelligence**: Real-time object detection and scene analysis using YOLOv8.
- **🎙️ Audio Transcription**: High-fidelity speech-to-text with automatic language detection.
- **🧠 Narrative Synthesis**: Generates a deep, contextual analysis report in French (or your preferred language).
- **💎 Luxury UI**: A sleek, dark-mode dashboard built with Gradio.

---

## 🛠️ How to run on Google Colab

Follow these simple steps to get Zenith AI running in seconds:

### 1. Open a New Notebook
Go to [Google Colab](https://colab.research.google.com/) and create a new Python 3 notebook.

### 2. Configure GPU Acceleration (Recommended)
For maximum performance:
- Go to `Runtime` > `Change runtime type`
- Select **T4 GPU** (or any available GPU)
- Click **Save**

### 3. Copy and Paste the Code
Copy the entire content of [main.ipynb](./main.ipynb) into a cell.

### 4. Setup your API
Before running the cell, locate the `API_CONFIG` section at the top of the script and enter your credentials:
```python
API_CONFIG = {
    "url": "YOUR_API_ENDPOINT",
    "key": "YOUR_API_KEY",
    "model": "YOUR_MODEL_NAME"
}
```

### 5. Run & Launch
- Execute the cell (Ctrl + Enter).
- Wait for the dependencies to install.
- Click the **public URL** (ending in `.gradio.live`) to open the dashboard.

---



---

## 📦 Dependencies
- `gradio`: Web Interface
- `ultralytics`: YOLOv8 Vision
- `faster-whisper`: Audio Transcription
- `yt-dlp`: Video Downloader
- `decord`: High-speed frame extraction

---

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.

---
Built with ❤️ By Shadrak BESSANH 