"""
Версия FastAPI: минимальная задержка, данные приходят из ROS топика
"""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

import cv2
import threading
import time
import logging
import numpy as np

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import config
from models.detector import ObjectDetector
from models.roi_logic import ROILogicManager

# ------------------- FastAPI setup -------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

frame = None
raw_frame = None
frame_lock = threading.Lock()
raw_frame_lock = threading.Lock()
detector = None
roi_manager = None
current_counts = {}
lost_notifications = []
events = []
frame_id = 0
last_frame_time = None

bridge = CvBridge()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

BLANK_IMAGE = np.zeros((240, 320, 3), dtype=np.uint8)
cv2.putText(BLANK_IMAGE, 'Нет данных', (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
BLANK_JPEG = cv2.imencode('.jpg', BLANK_IMAGE)[1].tobytes()

# ------------------- Обработка ROS -------------------
def image_callback(msg):
    """Callback для получения изображений из ROS"""
    global frame, raw_frame, frame_lock, raw_frame_lock
    global detector, roi_manager, last_frame_time, current_counts, lost_notifications, events, frame_id

    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with raw_frame_lock:
            raw_frame = cv_image.copy()

        if getattr(config, 'ENABLE_TRACKING', False):
            frame_id += 1
            processed_frame, tracked_objects = detector.detect(
                cv_image, return_detections=True, use_tracking=True
            )

            if roi_manager:
                lost_objects, new_events = roi_manager.update(tracked_objects, frame=cv_image, frame_id=frame_id)
                if new_events:
                    with frame_lock:
                        events.extend(new_events)
                        if len(events) > 100:
                            events[:] = events[-100:]
                lost_snapshot = roi_manager.get_lost_objects()
                stats = roi_manager.get_statistics()
                processed_frame = draw_tracked_objects(processed_frame, tracked_objects, lost_snapshot, stats)
                counts = {cls: data.get('active', 0) for cls, data in stats.items()}
                with frame_lock:
                    lost_notifications[:] = lost_snapshot
            else:
                processed_frame = draw_tracked_objects(processed_frame, tracked_objects, [], None)
                counts = {}
            detector.last_counts = counts
        else:
            processed_frame = detector.detect(cv_image)
            detector.last_counts = detector.last_counts or {}

        with frame_lock:
            frame = processed_frame
            last_frame_time = time.time()
            current_counts = detector.last_counts.copy()
    except Exception as e:
        logger.error(f"Ошибка обработки изображения: {e}")

# ------------------- Визуализация -------------------
def draw_tracked_objects(frame, tracked_objects, lost_objects, statistics=None):
    if frame is None:
        return frame
    canvas = frame
    # ... (оставляем реализацию из FastAPI версии, см. object_detection_ros)
    # для краткости опущено, но логика та же: зелёные боксы, красные пунктиры, статистика
    return canvas

def _encode_frame(img: np.ndarray) -> bytes:
    ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    return buffer.tobytes() if ret else BLANK_JPEG

def generate_processed_frames():
    while True:
        with frame_lock:
            jpeg = _encode_frame(frame) if frame is not None else BLANK_JPEG
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(max(0.01, 1.0 / config.OUTPUT_FPS))

def generate_raw_frames():
    while True:
        with raw_frame_lock:
            jpeg = _encode_frame(raw_frame) if raw_frame is not None else BLANK_JPEG
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(max(0.01, 1.0 / config.OUTPUT_FPS))

# ------------------- FastAPI endpoints -------------------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    with frame_lock:
        counts_snapshot = current_counts.copy()
    source_info = f"ROS топик {config.ROS_TOPIC}"
    return templates.TemplateResponse(
        "index.html",
        {"request": request,
         "source_info": source_info,
         "counts": counts_snapshot,
         "model_name": config.MODEL_NAME}
    )

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_processed_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/raw_video_feed")
async def raw_video_feed():
    return StreamingResponse(generate_raw_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/counts")
async def get_counts():
    with frame_lock:
        counts_snapshot = current_counts.copy()
    return JSONResponse({"counts": counts_snapshot, "timestamp": time.time()})

@app.get("/lost_notifications")
async def get_lost_notifications():
    with frame_lock:
        snapshot = list(lost_notifications)
    return JSONResponse(snapshot)

@app.get("/events")
async def get_events():
    with frame_lock:
        snapshot = list(events)
    return JSONResponse(snapshot)

# ------------------- Main -------------------
if __name__ == "__main__":
    print("Инициализация детектора...")
    detector = ObjectDetector(
        model_name=config.MODEL_NAME,
        device=config.DEVICE,
        model_path=getattr(config, "MODEL_PATH", "") or "",
        allowed_classes=getattr(config, "ALLOWED_CLASSES", None)
    )

    if getattr(config, "ENABLE_TRACKING", False):
        roi_manager = ROILogicManager(
            lost_timeout=getattr(config, "TRACKER_LOST_TIMEOUT", 5.0),
            event_timeout=getattr(config, "ROI_EVENT_TIMEOUT", 10.0),
            roi_expansion=getattr(config, "ROI_EXPANSION", 1.5),
            iou_threshold=getattr(config, "TRACKER_IOU_THRESHOLD", 0.3),
            recheck_frames=getattr(config, "ROI_RECHECK_FRAMES", 10),
            neighbor_offset_ratio=getattr(config, "ROI_NEIGHBOR_OFFSET_RATIO", 0.5),
            confirmation_frames=getattr(config, "ROI_CONFIRMATION_FRAMES", 3)
        )
        logger.info("✅ YOLO Track с ByteTrack инициализирован")
    else:
        roi_manager = None
        logger.info("⚠️  Трекинг отключен")

    # Инициализация ROS
    rospy.init_node("object_detection_server", anonymous=True)
    rospy.Subscriber(config.ROS_TOPIC, Image, image_callback)
    logger.info(f"ROS подписка на {config.ROS_TOPIC}")

    # Запуск FastAPI
    uvicorn.run(app, host="127.0.0.1", port=config.FLASK_PORT, log_level="info")
