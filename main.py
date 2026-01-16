
!apt-get update && apt-get install -y ffmpeg > /dev/null 2>&1
!pip install -q gradio requests numpy opencv-python-headless moviepy scenedetect ultralytics sentence-transformers markdown2 psutil pymediainfo faster-whisper open-clip-torch nvidia-ml-py decord imageio-ffmpeg pydub yt-dlp

#  Zenith AI 
import os, json, logging, shutil, time, threading, base64, hashlib, pickle, gc, traceback, asyncio, concurrent.futures
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
from collections import Counter
from datetime import datetime
from dataclasses import dataclass, field
import cv2, numpy as np, torch, requests, gradio as gr, psutil
from PIL import Image

# AI Tools Engine
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Configuration & Directories
BASE_DIR = Path("/content/video_analysis_pro")
OUTPUT_DIR, CACHE_DIR, REPORTS_DIR = BASE_DIR/"output", BASE_DIR/"cache", BASE_DIR/"reports"
for d in [BASE_DIR, OUTPUT_DIR, CACHE_DIR, REPORTS_DIR]: d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VisionPro")

# API Configuration - TO BE CONFIGURED BY USER
API_CONFIG = {
    "url": "YOUR_API_ENDPOINT_HERE",
    "key": "YOUR_API_KEY_HERE",
    "model": "YOUR_MODEL_NAME_HERE"
}

@dataclass
class Frame:
    path: Path
    timestamp: float
    metrics: Dict[str, float] = None
    vision_content: str = ""

# --- PROCESSING LOGIC ---
def get_frame_metrics(frame: np.ndarray) -> dict:
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return {"brightness": float(np.mean(gray)), "contrast": float(np.std(gray)),
                "saturation": float(np.mean(hsv[:, :, 1])), "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var())}
    except: return {"brightness": 0, "contrast": 0, "saturation": 0, "sharpness": 0}

class VideoProcessor:
    def __init__(self, video_path: Path, output_dir: Path):
        self.video_path, self.output_dir = video_path, output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_keyframes(self, max_frames: int = 50) -> List[Frame]:
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(str(self.video_path), ctx=cpu(0))
            total = len(vr)
            step = max(1, total // max_frames)
            indices = range(0, total, step)[:max_frames]

            # Récupération rapide des images via Decord
            frames_data = vr.get_batch(indices).asnumpy()
            fps = vr.get_avg_fps()

            extracted = []
            for i, idx in enumerate(indices):
                img = cv2.cvtColor(frames_data[i], cv2.COLOR_RGB2BGR)
                ts = idx / fps
                p = self.output_dir / f"f_{idx}.jpg"
                cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                extracted.append(Frame(path=p, timestamp=ts, metrics=get_frame_metrics(img)))
            return extracted
        except Exception as e:
            logger.warning(f"Decord faster extraction failed, falling back to CV2: {e}")
            # Fallback OpenCV
            cap = cv2.VideoCapture(str(self.video_path))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1000
            step = max(1, total // max_frames)
            extracted = []
            for idx in range(0, total, step):
                if len(extracted) >= max_frames: break
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, img = cap.read()
                if ret:
                    ts = idx / fps
                    p = self.output_dir / f"f_{idx}.jpg"
                    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    extracted.append(Frame(path=p, timestamp=ts, metrics=get_frame_metrics(img)))
            cap.release()
            return extracted

class AudioProcessor:
    def __init__(self): self.model = None
    def initialize(self):
        if WHISPER_AVAILABLE and self.model is None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = WhisperModel("base", device=device, compute_type="int8")
            except: pass
    def transcribe(self, p: Path) -> str:
        self.initialize()
        if not self.model: return "Transcription indisponible"
        try:
            segments, info = self.model.transcribe(str(p), beam_size=5)
            transcript = " ".join([s.text for s in segments])
            return f"[Langue source détectée: {info.language.upper()}] {transcript}"
        except: return "Erreur transcription"

class VideoDownloader:
    @staticmethod
    def download(url: str, output_dir: Path) -> Optional[Path]:
        import yt_dlp
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': str(output_dir / 'downloaded_video.%(ext)s'),
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return Path(ydl.prepare_filename(info))
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None

class APIGatewayClient:
    def __init__(self, key: str, url: str): self.key, self.url = key, url.strip() if "/chat/completions" in url else url.rstrip('/') + "/chat/completions"
    def chat_stream(self, model: str, prompt: str, images: List[str] = None) -> Iterator[str]:
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        content = [{"type": "text", "text": prompt}]
        if images:
            for p in images[:3]:
                with open(p, "rb") as f: b64 = base64.b64encode(f.read()).decode('utf-8')
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        try:
            r = requests.post(self.url, headers=headers, json={"model": model, "messages": [{"role": "user", "content": content}], "stream": True}, stream=True, timeout=300)
            for line in r.iter_lines():
                if line.startswith(b"data: "):
                    d = line[6:].decode('utf-8')
                    if d == "[DONE]": break
                    try:
                        c = json.loads(d)
                        if 'choices' in c: yield c['choices'][0].get('delta', {}).get('content', '')
                    except: continue
        except Exception as e: yield f"Connection Error: {str(e)}"

# --- INTERFACE ENGINE ---
audio_proc = AudioProcessor()
current_data = {"frames": [], "transcript": "", "video": None, "report": ""}

def process_video(video_path, video_url=None):
    if not video_path and (not video_url or not video_url.strip()):
        return gr.update(visible=False), [], " Média ou URL requis", gr.update(visible=False)

    current_data["frames"] = []
    current_data["report"] = ""
    session_dir = OUTPUT_DIR / f"session_{int(time.time())}"
    session_dir.mkdir(parents=True, exist_ok=True)

    if video_url and video_url.strip():
        yield gr.update(visible=False), [], " Téléchargement de la vidéo en cours...", gr.update(visible=False)
        downloaded_path = VideoDownloader.download(video_url.strip(), session_dir)
        if not downloaded_path:
            yield gr.update(visible=False), [], " Échec du téléchargement. Vérifiez l'URL.", gr.update(visible=False)
            return
        current_data["video"] = downloaded_path
    elif video_path:
        current_data["video"] = Path(video_path)
    else:
        return gr.update(visible=False), [], " Média ou URL requis", gr.update(visible=False)

    proc = VideoProcessor(current_data["video"], session_dir)
    yield gr.update(visible=False), [], " Échantillonnage  en cours...", gr.update(visible=False)
    current_data["frames"] = proc.extract_keyframes()
    gallery = [(str(f.path), f"{f.timestamp:.1f}s") for f in current_data["frames"][:12]]

    yield (
        gr.update(visible=True),
        gallery,
        " Importation terminée. Système prêt pour l'analyse .",
        gr.update(visible=True)
    )

def run_analysis():
    if not current_data["frames"]: yield "Veuillez charger une vidéo", gr.update(visible=False); return
    yield " Initialisation de l'analyse  en cours...", gr.update(visible=False)

    if API_CONFIG["url"] == "YOUR_API_ENDPOINT_HERE":
        yield "❌ Erreur : Veuillez configurer votre API Endpoint dans le code.", gr.update(visible=False)
        return

    client = APIGatewayClient(API_CONFIG["key"], API_CONFIG["url"])
    yield " Analyse multimodale en cours  ...", gr.update(visible=False)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        audio_task = executor.submit(audio_proc.transcribe, current_data["video"])
        def analyze_vision():
            yolo = YOLO("yolov8n.pt") if YOLO_AVAILABLE else None
            if not yolo: return
            for f in current_data["frames"]:
                res = yolo(str(f.path), verbose=False, imgsz=320)
                objs = [res[0].names[int(b.cls[0])] for b in res[0].boxes if b.conf > 0.25]
                f.vision_content = ", ".join([f"{v}x {k}" for k,v in Counter(objs).items()])
        vision_task = executor.submit(analyze_vision)
        concurrent.futures.wait([audio_task, vision_task])
        current_data["transcript"] = audio_task.result()

    yield " Génération de la synthèse narrative, Veuillez Patienter ...", gr.update(visible=False)
    v_info = "\n".join([f"[{f.timestamp:.1f}s] {f.vision_content}" for f in current_data["frames"][:15]])

    prompt = f"""Tu es l'unité d'Zenith AI. Ton rôle est d'analyser ce flux vidéo et sa bande sonore.

    DONNÉES DE TRANSCRIPTION:
    {current_data['transcript']}

    DONNÉES VISUELLES:
    {v_info}

    CONSIGNES CRUCIALES :
    1. Si la transcription est dans une langue autre que le français, TRADUIS-LA parfaitement.
    2. Produis un rapport d'analyse COMPLET et NARRATIF en FRANÇAIS uniquement.
    3. Structure : Introduction contextuelle, Synthèse audio/discursive, Conclusion .
    4. Ton : Professionnel, analytique."""

    report = ""
    for chunk in client.chat_stream(API_CONFIG["model"], prompt, [str(f.path) for f in current_data["frames"][:3]]):
        report += chunk
        yield report, gr.update(visible=False)

    current_data["report"] = report
    h_path = REPORTS_DIR / f"Report_{int(time.time())}.html"
    with open(h_path, "w") as f: f.write(f"<html><body style='font-family:sans-serif;padding:40px;line-height:1.6'><h2>Vision Pro Analysis</h2><hr>{report.replace(chr(10), '<br>')}</body></html>")
    yield report, gr.update(value=str(h_path), visible=True)

# --- LUXURY CSS & THEME ---
theme_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
:root { --primary: #6366f1; --bg: #0b0f1a; --card: #161b2a; --border: #2d364f; }
.gradio-container { background-color: var(--bg) !important; font-family: 'Inter', sans-serif !important; color: #e2e8f0 !important; }
.dashboard-header { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-bottom: 1px solid var(--border); padding: 40px; border-radius: 20px; text-align: center; margin-bottom: 30px; }
.glass-card { background: var(--card); border: 1px solid var(--border); border-radius: 16px; padding: 24px; }
.report-view { background: #0f172a; border-radius: 12px; border: 1px solid #1e293b; padding: 30px; line-height: 1.8; color: #cbd5e1; font-size: 1.05rem; min-height: 600px; }
.primary-btn { background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important; color: white !important; font-weight: 600 !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo"), css=theme_css) as demo:
    with gr.Column(elem_id="main-app"):
        gr.HTML("""<div class="dashboard-header"><h1>Zenith AI</h1><p>Analyse et synthèse narrative de flux vidéo</p></div>""")
        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("### ÉTAPE 1 : IMPORTATION")
                    video_input = gr.Video(label="Uploader une vidéo", height=200)
                    gr.Markdown("<p style='text-align:center;'>— OU —</p>")
                    url_input = gr.Textbox(label="Lien Vidéo", placeholder="URL (YouTube, TikTok...)")
                    status_msg = gr.Markdown("*Prêt pour l'importation...*")
                with gr.Column(elem_classes="glass-card", visible=False) as vision_panel:
                    gr.Markdown("### ÉTAPE 2 : ÉCHANTILLONNAGE")
                    gallery = gr.Gallery(columns=3, height=400, preview=True)
                    action_btn = gr.Button("Lancer l'Analyse", variant="primary", elem_classes="primary-btn", visible=False)
            with gr.Column(scale=6, visible=False) as report_panel:
                with gr.Column(elem_classes="glass-card"):
                    gr.Markdown("### ÉTAPE 3 : SYNTHÈSE ANALYTIQUE")
                    report_area = gr.Markdown(value="*Initialisation du moteur...*", elem_classes="report-view")
                    download_btn = gr.File(label="Télécharger le Rapport", visible=False)

    video_input.change(fn=process_video, inputs=[video_input, url_input], outputs=[vision_panel, gallery, status_msg, action_btn])
    url_input.submit(fn=process_video, inputs=[video_input, url_input], outputs=[vision_panel, gallery, status_msg, action_btn])
    action_btn.click(fn=lambda: gr.update(visible=True), outputs=[report_panel]).then(fn=run_analysis, outputs=[report_area, download_btn])

demo.launch(share=True)
