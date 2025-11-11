"""
Версия с поддержкой ROS для детекции объектов в видеопотоке
Интегрируется с существующим ROS кодом для дрона
"""

from flask import Flask, Response, render_template
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import time

# Импорты из модулей проекта
import config
from models.detector import ObjectDetector
from utils.network import get_local_ip, print_server_info

app = Flask(__name__)
bridge = CvBridge()
frame = None
frame_lock = threading.Lock()
detector = None  # Объект детектора
last_frame_time = None  # Время последнего полученного кадра


def image_callback(msg):
    """Callback для получения изображений из ROS"""
    global frame, frame_lock, detector, last_frame_time
    
    try:
        # Конвертируем ROS сообщение в OpenCV изображение
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Детектируем объекты
        processed_frame = detector.detect(cv_image)
        
        # Обновляем глобальный кадр и время
        with frame_lock:
            frame = processed_frame
            last_frame_time = time.time()  # Обновляем время последнего кадра
    except Exception as e:
        print(f"Ошибка обработки изображения: {e}")


def generate_frames():
    """Генерирует кадры для трансляции"""
    global frame, frame_lock, last_frame_time
    
    while True:
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
        time.sleep(0.03)


@app.route('/')
def index():
    """Главная страница"""
    source_info = f"ROS топик {config.ROS_TOPIC}"
    return render_template('index.html', source_info=source_info)


@app.route('/video_feed')
def video_feed():
    """Эндпоинт для видеопотока"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


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
    detector = ObjectDetector(model_name=config.MODEL_NAME, device=config.DEVICE)
    
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
