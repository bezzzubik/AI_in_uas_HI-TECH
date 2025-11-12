"""
Система детекции объектов в видеопотоке с трансляцией через FastAPI
Использует YOLOv8 для детекции объектов (людей, машин и т.д.)
"""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import threading
import time
import numpy as np

import config
from models.detector import ObjectDetector
from models.roi_logic import ROILogicManager
from utils.network import get_local_ip, print_server_info
from utils.camera import get_camera_source
import logging
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

frame = None
raw_frame = None
frame_lock = threading.Lock()
raw_frame_lock = threading.Lock()
last_frame_time = None
raw_last_frame_time = None
detector = None
roi_manager = None
actual_camera_source = None
current_counts = {}
lost_notifications = []
events = []
frame_id = 0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tracking.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


BLANK_IMAGE = np.zeros((240, 320, 3), dtype=np.uint8)
cv2.putText(BLANK_IMAGE, 'Нет данных', (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
BLANK_JPEG = cv2.imencode('.jpg', BLANK_IMAGE)[1].tobytes()


def draw_tracked_objects(frame, tracked_objects, lost_objects, statistics=None):
    if frame is None:
        return frame

    canvas = frame
    for obj in tracked_objects:
        x1, y1, x2, y2 = [int(coord) for coord in obj['bbox']]
        track_id = obj.get('track_id')
        class_name = obj['class']
        confidence = obj['confidence']
        color = (0, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        label = f"ID:{track_id} {class_name} {confidence:.2f}" if track_id is not None else f"{class_name} {confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(canvas, (x1, y1 - th - baseline - 5), (x1 + tw + 5, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    for obj in lost_objects:
        x1, y1, x2, y2 = [int(coord) for coord in obj['bbox']]
        track_id = obj['track_id']
        class_name = obj['class']
        status = obj.get('status', 'lost')
        time_lost = obj.get('time_lost', 0)
        color = (255, 165, 0) if status == 'reappeared' else (0, 0, 255)
        dash_length = 10
        gap_length = 5
        for x in range(x1, x2, dash_length + gap_length):
            cv2.line(canvas, (x, y1), (min(x + dash_length, x2), y1), color, 2)
            cv2.line(canvas, (x, y2), (min(x + dash_length, x2), y2), color, 2)
        for y in range(y1, y2, dash_length + gap_length):
            cv2.line(canvas, (x1, y), (x1, min(y + dash_length, y2)), color, 2)
            cv2.line(canvas, (x2, y), (x2, min(y + dash_length, y2)), color, 2)
        status_text = "REAPPEARED" if status == 'reappeared' else f"LOST ({time_lost:.1f}s)"
        label = f"ID:{track_id} {class_name} ({status_text})"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - baseline - 5), (x1 + tw + 5, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        roi = obj.get('roi')
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (255, 165, 0), 1)

    if statistics:
        y_offset = 30
        x_offset = 10
        summary_lines = [f"{cls}: {data.get('active', 0)} активных / {data.get('total_detected', 0)} всего"
                         for cls, data in statistics.items()]
        if summary_lines:
            overlay = canvas.copy()
            text_height = len(summary_lines) * 25 + 10
            max_line_width = max(len(line) for line in summary_lines)
            cv2.rectangle(overlay, (x_offset, y_offset - 20),
                          (x_offset + max_line_width * 9, y_offset + text_height),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
            for i, line in enumerate(summary_lines):
                cv2.putText(canvas, line, (x_offset + 5, y_offset + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas


def video_capture_thread(source):
    global frame, raw_frame, frame_lock, raw_frame_lock, last_frame_time, raw_last_frame_time
    global detector, roi_manager, actual_camera_source, current_counts, lost_notifications, events, frame_id

    actual_source = get_camera_source(source)
    actual_camera_source = actual_source
    cap = cv2.VideoCapture(actual_source)
    if not cap.isOpened():
        logger.error("❌ Ошибка: не удалось открыть источник видео %s", actual_source)
        return

    logger.info("✅ Видеопоток открыт: %s", actual_source)
    if isinstance(actual_source, int):
        cap.set(cv2.CAP_PROP_FPS, config.CAPTURE_FPS)
        logger.info("📹 Частота захвата: %s FPS", config.CAPTURE_FPS)

    capture_delay = 1.0 / config.CAPTURE_FPS
    processing_delay = 1.0 / config.PROCESSING_FPS
    frame_count = 0
    last_processing_time = time.time()

    while True:
        ret, captured_frame = cap.read()
        if not ret:
            logger.warning("Ошибка чтения кадра")
            time.sleep(0.1)
            continue

        frame_count += 1
        current_time = time.time()
        with raw_frame_lock:
            raw_frame = captured_frame.copy()
            raw_last_frame_time = current_time

        should_process = (current_time - last_processing_time) >= processing_delay
        if should_process and detector is not None:
            if getattr(config, 'ENABLE_TRACKING', False):
                frame_id += 1
                processed_frame, tracked_objects = detector.detect(
                    captured_frame,
                    return_detections=True,
                    use_tracking=True
                )
                if roi_manager is not None:
                    lost_objects, new_events = roi_manager.update(
                        tracked_objects,
                        frame=captured_frame,
                        frame_id=frame_id
                    )
                    if new_events:
                        with frame_lock:
                            events.extend(new_events)
                            if len(events) > 100:
                                events[:] = events[-100:]
                    lost_snapshot = roi_manager.get_lost_objects()
                    stats = roi_manager.get_statistics()
                    processed_frame = draw_tracked_objects(
                        processed_frame, tracked_objects, lost_snapshot, stats
                    )
                    counts = {cls: data.get('active', 0) for cls, data in stats.items()}
                    with frame_lock:
                        lost_notifications[:] = lost_snapshot
                else:
                    processed_frame = draw_tracked_objects(processed_frame, tracked_objects, [], None)
                    counts = {}
                detector.last_counts = counts
            else:
                processed_frame = detector.detect(captured_frame)
                detector.last_counts = detector.last_counts or {}
            last_processing_time = current_time
        else:
            processed_frame = captured_frame

        with frame_lock:
            frame = processed_frame
            last_frame_time = current_time
            current_counts = detector.last_counts.copy() if detector and detector.last_counts else {}

        time.sleep(capture_delay)

    cap.release()


def _encode_frame(img: np.ndarray) -> bytes:
    ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
    if not ret:
        return BLANK_JPEG
    return buffer.tobytes()


def generate_processed_frames():
    while True:
        with frame_lock:
            if frame is not None:
                jpeg = _encode_frame(frame)
            else:
                jpeg = BLANK_JPEG
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(max(0.01, 1.0 / config.OUTPUT_FPS))


def generate_raw_frames():
    while True:
        with raw_frame_lock:
            if raw_frame is not None:
                jpeg = _encode_frame(raw_frame)
            else:
                jpeg = BLANK_JPEG
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(max(0.01, 1.0 / config.OUTPUT_FPS))


@app.get("/")
async def index(request: Request):
    with frame_lock:
        counts_snapshot = current_counts.copy()
    source_info = f"Источник: {actual_camera_source if actual_camera_source is not None else config.VIDEO_SOURCE}"
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "source_info": source_info,
            "counts": counts_snapshot,
            "model_name": config.MODEL_NAME
        }
    )


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        generate_processed_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/raw_video_feed")
async def raw_video_feed():
    return StreamingResponse(
        generate_raw_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/counts")
async def get_counts():
    with frame_lock:
        counts_snapshot = current_counts.copy()
    return JSONResponse({
        "counts": counts_snapshot,
        "timestamp": time.time()
    })


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


if __name__ == '__main__':
    print("Инициализация детектора...")
    print(f"Модель: {config.MODEL_NAME}")
    print(f"Устройство: {config.DEVICE}")
    detector = ObjectDetector(
        model_name=config.MODEL_NAME,
        device=config.DEVICE,
        model_path=getattr(config, 'MODEL_PATH', '') or '',
        allowed_classes=getattr(config, 'ALLOWED_CLASSES', None)
    )

    if getattr(config, 'ENABLE_TRACKING', False):
        roi_manager = ROILogicManager(
            lost_timeout=getattr(config, 'TRACKER_LOST_TIMEOUT', 5.0),
            event_timeout=getattr(config, 'ROI_EVENT_TIMEOUT', 10.0),
            roi_expansion=getattr(config, 'ROI_EXPANSION', 1.5),
            iou_threshold=getattr(config, 'TRACKER_IOU_THRESHOLD', 0.3),
            recheck_frames=getattr(config, 'ROI_RECHECK_FRAMES', 10),
            neighbor_offset_ratio=getattr(config, 'ROI_NEIGHBOR_OFFSET_RATIO', 0.5),
            confirmation_frames=getattr(config, 'ROI_CONFIRMATION_FRAMES', 3)
        )
        logger.info("✅ YOLO Track с ByteTrack инициализирован")
    else:
        roi_manager = None
        logger.info("⚠️  Трекинг отключен")

    time.sleep(0.5)
    video_thread = threading.Thread(target=video_capture_thread, args=(config.VIDEO_SOURCE,), daemon=True)
    video_thread.start()
    time.sleep(0.5)

    local_ip = get_local_ip()
    print_server_info(local_ip, config.FLASK_PORT)

    uvicorn.run(
        app,
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        log_level="debug" if config.FLASK_DEBUG else "info"
    )
