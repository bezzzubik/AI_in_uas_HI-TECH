"""
Версия с поддержкой ROS для детекции объектов в видеопотоке
Интегрируется с существующим ROS кодом для дрона
"""

from flask import Flask, Response, render_template, jsonify
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import time

# Импорты из модулей проекта
import config
from models.detector import ObjectDetector
from models.roi_logic import ROILogicManager, ObjectStatus
from utils.network import get_local_ip, print_server_info
import logging

app = Flask(__name__)
bridge = CvBridge()
frame = None
frame_lock = threading.Lock()
detector = None  # Объект детектора
roi_manager = None  # Менеджер ROI-логики
last_frame_time = None  # Время последнего полученного кадра
current_counts = {}  # Текущие подсчитанные объекты
lost_notifications = []  # Уведомления о потерянных объектах
events = []  # События о длительной потере
frame_id = 0  # Счетчик кадров

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tracking_ros.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def draw_tracked_objects(frame, tracked_objects, lost_objects, statistics=None):
    """
    Рисует отслеживаемые и потерянные объекты на кадре с улучшенной визуализацией
    
    Args:
        frame: кадр для рисования
        tracked_objects: список активных объектов от YOLO Track
        lost_objects: список потерянных объектов от ROI-логики
        statistics: статистика по классам
    """
    h, w = frame.shape[:2]
    
    # Рисуем отслеживаемые объекты (зеленые боксы с ID)
    for obj in tracked_objects:
        x1, y1, x2, y2 = [int(coord) for coord in obj['bbox']]
        track_id = obj['track_id']
        class_name = obj['class']
        confidence = obj['confidence']
        
        # Зеленый бокс для активных объектов
        color = (0, 255, 0)  # Зеленый
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Метка с ID, классом и уверенностью
        label = f"ID:{track_id} {class_name} {confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 5), (x1 + tw + 5, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Рисуем потерянные объекты (красные пунктирные боксы)
    for obj in lost_objects:
        x1, y1, x2, y2 = [int(coord) for coord in obj['bbox']]
        track_id = obj['track_id']
        class_name = obj['class']
        status = obj.get('status', 'lost')
        time_lost = obj.get('time_lost', 0)
        
        # Цвет в зависимости от статуса
        if status == 'reappeared':
            color = (255, 165, 0)  # Оранжевый для появившихся снова
        else:
            color = (0, 0, 255)  # Красный для потерянных
        dash_length = 10
        gap_length = 5
        
        # Верхняя линия
        x = x1
        while x < x2:
            cv2.line(frame, (x, y1), (min(x + dash_length, x2), y1), color, 2)
            x += dash_length + gap_length
        
        # Нижняя линия
        x = x1
        while x < x2:
            cv2.line(frame, (x, y2), (min(x + dash_length, x2), y2), color, 2)
            x += dash_length + gap_length
        
        # Левая линия
        y = y1
        while y < y2:
            cv2.line(frame, (x1, y), (x1, min(y + dash_length, y2)), color, 2)
            y += dash_length + gap_length
        
        # Правая линия
        y = y1
        while y < y2:
            cv2.line(frame, (x2, y), (x2, min(y + dash_length, y2)), color, 2)
            y += dash_length + gap_length
        
        # Метка для потерянного объекта
        status_text = "REAPPEARED" if status == 'reappeared' else f"LOST ({time_lost:.1f}s)"
        label = f"ID:{track_id} {class_name} ({status_text})"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 5), (x1 + tw + 5, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Рисуем ROI для поиска если есть
        if obj.get('roi') is not None:
            roi_x1, roi_y1, roi_x2, roi_y2 = obj['roi']
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 165, 0), 1)  # Оранжевый для ROI
    
    # Рисуем текстовую сводку статистики
    if statistics:
        y_offset = 30
        x_offset = 10
        
        # Фон для текста
        summary_lines = []
        for class_name, stats in statistics.items():
            active = stats.get('active', 0)
            total = stats.get('total_detected', 0)
            summary_lines.append(f"{class_name}: {active} активных / {total} всего")
        
        if summary_lines:
            max_line_width = max(len(line) for line in summary_lines)
            text_height = len(summary_lines) * 25 + 10
            
            # Полупрозрачный фон
            overlay = frame.copy()
            cv2.rectangle(overlay, (x_offset, y_offset - 20), 
                         (x_offset + max_line_width * 8, y_offset + text_height), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Текст статистики
            for i, line in enumerate(summary_lines):
                cv2.putText(frame, line, (x_offset + 5, y_offset + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def image_callback(msg):
    """Callback для получения изображений из ROS"""
    global frame, frame_lock, detector, roi_manager, last_frame_time, current_counts, lost_notifications, events, frame_id
    
    try:
        # Конвертируем ROS сообщение в OpenCV изображение
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Детектируем объекты с трекингом YOLO Track
        if getattr(config, 'ENABLE_TRACKING', False):
            frame_id += 1
            processed_frame, tracked_objects = detector.detect(
                cv_image, 
                return_detections=True, 
                use_tracking=True
            )
            
            # Обновляем ROI-логику
            if roi_manager is not None:
                lost_objects, new_events = roi_manager.update(tracked_objects, frame=cv_image, frame_id=frame_id)
                
                # Обновляем ROI для потерянных объектов
                frame_shape = cv_image.shape
                for lost_obj in lost_objects:
                    if lost_obj.get('roi') is None:
                        lost_obj_data = roi_manager.lost_objects.get(lost_obj['track_id'])
                        if lost_obj_data:
                            roi = lost_obj_data.get_search_roi(frame_shape, roi_manager.roi_expansion)
                            lost_obj['roi'] = roi
                
                # Сохраняем новые события
                if new_events:
                    with frame_lock:
                        events.extend(new_events)
                        if len(events) > 100:
                            events = events[-100:]
                
                # Обновляем уведомления
                with frame_lock:
                    lost_notifications = roi_manager.get_lost_objects()
            
            # Рисуем отслеживаемые и потерянные объекты
            if roi_manager is not None:
                lost_objects = roi_manager.get_lost_objects()
                statistics = roi_manager.get_statistics()
                processed_frame = draw_tracked_objects(
                    processed_frame, tracked_objects, lost_objects, statistics
                )
                counts = {cls: data.get('active', 0) for cls, data in statistics.items()}
            else:
                processed_frame = draw_tracked_objects(
                    processed_frame, tracked_objects, [], None
                )
                counts = {}
            
            detector.last_counts = counts
        else:
            # Обычная детекция без трекинга
            processed_frame = detector.detect(cv_image)
        
        # Обновляем глобальный кадр и время
        with frame_lock:
            frame = processed_frame
            last_frame_time = time.time()  # Обновляем время последнего кадра
            current_counts = detector.last_counts.copy() if detector and detector.last_counts else {}
    except Exception as e:
        logger.error(f"Ошибка обработки изображения: {e}")


def generate_frames():
    """Генерирует кадры для трансляции"""
    global frame, frame_lock, last_frame_time
    
    while True:
        start_func = time.time()
        with frame_lock:
            current_time = time.time()
            # Проверяем, прошло ли более указанного времени с последнего кадра
            if frame is not None and last_frame_time is not None:
                time_since_last_frame = current_time - last_frame_time
                if time_since_last_frame <= config.NO_DATA_TIMEOUT:
                    ret, buffer = cv2.imencode('.jpg', frame, 
                                               [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        end_func = time.time()
        if end_func - start_func > 1 / config.OUTPUT_FPS:
            time.sleep(1 / config.OUTPUT_FPS - (end_func - start_func))


@app.route('/')
def index():
    """Главная страница"""
    global current_counts
    with frame_lock:
        counts_snapshot = current_counts.copy() if current_counts else {}
    source_info = f"ROS топик {config.ROS_TOPIC}"
    return render_template('index.html', source_info=source_info, counts=counts_snapshot, timestamp=int(time.time() * 1000))


@app.route('/video_feed')
def video_feed():
    """Эндпоинт для видеопотока"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/counts')
def get_counts():
    """Возвращает текущие подсчитанные объекты"""
    with frame_lock:
        counts_snapshot = current_counts.copy() if current_counts else {}
    return jsonify({
        'counts': counts_snapshot,
        'timestamp': time.time()
    })


@app.route('/lost_notifications')
def get_lost_notifications():
    """API для получения уведомлений о потерянных объектах"""
    global lost_notifications
    with frame_lock:
        notifications_snapshot = lost_notifications.copy() if lost_notifications else []
    return jsonify(notifications_snapshot)

@app.route('/events')
def get_events():
    """API для получения событий о длительной потере объектов"""
    global events
    with frame_lock:
        events_snapshot = events.copy() if events else []
    return jsonify(events_snapshot)


def init_ros():
    """Инициализирует ROS"""
    rospy.init_node('object_detection_server', anonymous=True)
    rospy.Subscriber(config.ROS_TOPIC, Image, image_callback)
    print(f"ROS инициализирован. Подписка на {config.ROS_TOPIC}")


if __name__ == '__main__':
    # Инициализируем детектор с настройками из config
    # detector уже объявлена как глобальная переменная на уровне модуля
    print(f"Инициализация детектора...")
    print(f"Модель: {config.MODEL_NAME}")
    print(f"Устройство: {config.DEVICE}")
    detector = ObjectDetector(
        model_name=config.MODEL_NAME,
        device=config.DEVICE,
        model_path=getattr(config, 'MODEL_PATH', '') or '',
        allowed_classes=getattr(config, 'ALLOWED_CLASSES', None)
    )
    
    # Инициализируем ROI-менеджер если включен трекинг
    if getattr(config, 'ENABLE_TRACKING', False):
        lost_timeout = getattr(config, 'TRACKER_LOST_TIMEOUT', 5.0)
        event_timeout = getattr(config, 'ROI_EVENT_TIMEOUT', 10.0)
        roi_expansion = getattr(config, 'ROI_EXPANSION', 1.5)
        iou_threshold = getattr(config, 'TRACKER_IOU_THRESHOLD', 0.3)
        recheck_frames = getattr(config, 'ROI_RECHECK_FRAMES', 10)
        neighbor_offset_ratio = getattr(config, 'ROI_NEIGHBOR_OFFSET_RATIO', 0.5)
        confirmation_frames = getattr(config, 'ROI_CONFIRMATION_FRAMES', 3)
        
        roi_manager = ROILogicManager(
            lost_timeout=lost_timeout,
            event_timeout=event_timeout,
            roi_expansion=roi_expansion,
            iou_threshold=iou_threshold,
            recheck_frames=recheck_frames,
            neighbor_offset_ratio=neighbor_offset_ratio,
            confirmation_frames=confirmation_frames
        )
        logger.info("✅ YOLO Track с ByteTrack инициализирован")
        logger.info(f"   Lost timeout: {lost_timeout} сек")
        logger.info(f"   Event timeout: {event_timeout} сек")
        logger.info(f"   ROI expansion: {roi_expansion}")
        logger.info(f"   IoU threshold: {iou_threshold}")
        logger.info(f"   ROI recheck frames: {recheck_frames}")
        logger.info(f"   ROI neighbor offset ratio: {neighbor_offset_ratio}")
        logger.info(f"   Confirmation frames: {confirmation_frames}")
    else:
        roi_manager = None
        logger.info("⚠️  Трекинг отключен")
    
    # Инициализируем ROS
    init_ros()
    
    # Получаем локальный IP-адрес и выводим информацию
    local_ip = get_local_ip()
    print_server_info(local_ip, config.FLASK_PORT)
    print("Ожидание изображений из ROS топика...")
    print("="*50 + "\n")
    
    # Запускаем Flask в отдельном потоке
    flask_thread = threading.Thread(
        target=lambda: app.run(
            host=config.FLASK_HOST, 
            port=config.FLASK_PORT, 
            debug=config.FLASK_DEBUG, 
            threaded=True, 
            use_reloader=False
        ),
        daemon=True
    )
    flask_thread.start()
    
    # Основной цикл ROS
    rospy.spin()
