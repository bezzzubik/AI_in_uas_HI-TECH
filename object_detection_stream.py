"""
Система детекции объектов в видеопотоке с трансляцией через Flask
Использует YOLOv8 для детекции объектов (людей, машин и т.д.)
"""

from flask import Flask, Response, render_template
import cv2
import threading
import time

# Импорты из модулей проекта
import config
from models.detector import ObjectDetector
from utils.network import get_local_ip, print_server_info
from utils.camera import get_camera_source

app = Flask(__name__)

# Глобальные переменные
frame = None
frame_lock = threading.Lock()
detector = None  # Объект детектора
last_frame_time = None  # Время последнего полученного кадра
actual_camera_source = None  # Фактически используемый источник камеры


def video_capture_thread(source):
    """Поток для захвата видео с камеры"""
    global frame, frame_lock, last_frame_time, detector, actual_camera_source
    
    # Получаем доступный источник (автоматически выбирает камеру, если нужно)
    actual_source = get_camera_source(source)
    actual_camera_source = actual_source  # Сохраняем для отображения в веб-интерфейсе
    
    # source может быть:
    # - 0 или 1 для веб-камеры
    # - путь к видео файлу
    # - URL для IP-камеры
    cap = cv2.VideoCapture(actual_source)
    
    if not cap.isOpened():
        print(f"❌ Ошибка: не удалось открыть источник видео {actual_source}")
        return
    
    print(f"✅ Видеопоток открыт: {actual_source}")
    
    # Настраиваем FPS если возможно (только для веб-камер)
    if isinstance(actual_source, int):
        cap.set(cv2.CAP_PROP_FPS, config.CAPTURE_FPS)
        print(f"📹 Частота захвата: {config.CAPTURE_FPS} FPS")
    
    # Вычисляем задержки
    capture_delay = 1.0 / config.CAPTURE_FPS
    processing_delay = 1.0 / config.PROCESSING_FPS
    
    # Счетчики для контроля частоты обработки
    frame_count = 0
    last_processing_time = time.time()
    
    print(f"🔍 Частота обработки: {config.PROCESSING_FPS} FPS")
    print(f"📤 Частота вывода: {config.OUTPUT_FPS} FPS")
    
    while True:
        ret, captured_frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра")
            break
        
        frame_count += 1
        current_time = time.time()
        
        # Проверяем, нужно ли обрабатывать этот кадр
        time_since_last_processing = current_time - last_processing_time
        should_process = time_since_last_processing >= processing_delay
        
        if should_process and detector is not None:
            # Детектируем объекты
            processed_frame = detector.detect(captured_frame)
            last_processing_time = current_time
        else:
            # Используем оригинальный кадр без обработки
            processed_frame = captured_frame
        
        # Обновляем глобальный кадр и время
        with frame_lock:
            frame = processed_frame
            last_frame_time = time.time()  # Обновляем время последнего кадра
        
        # Задержка для контроля частоты захвата
        time.sleep(capture_delay)
    
    cap.release()


def generate_frames():
    """Генерирует кадры для трансляции"""
    global frame, frame_lock, last_frame_time
    
    frames_sent = 0
    output_delay = 1.0 / config.OUTPUT_FPS
    last_output_time = time.time()
    
    while True:
        with frame_lock:
            current_time = time.time()
            # Проверяем, прошло ли более указанного времени с последнего кадра
            if frame is not None and last_frame_time is not None:
                time_since_last_frame = current_time - last_frame_time
                if time_since_last_frame <= config.NO_DATA_TIMEOUT:
                    # Контролируем частоту вывода
                    time_since_last_output = current_time - last_output_time
                    if time_since_last_output >= output_delay:
                        # Кодируем кадр в JPEG
                        try:
                            ret, buffer = cv2.imencode('.jpg', frame, 
                                                       [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
                            if ret:
                                frame_bytes = buffer.tobytes()
                                frames_sent += 1
                                last_output_time = current_time
                                if frames_sent % 100 == 0:  # Логируем каждые 100 кадров
                                    print(f"Отправлено кадров: {frames_sent}")
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        except Exception as e:
                            print(f"Ошибка кодирования кадра: {e}")
            else:
                # Отладочная информация при отсутствии кадров
                if frames_sent == 0:
                    if frame is None:
                        print("⚠️  Кадр еще не получен...")
                    elif last_frame_time is None:
                        print("⚠️  Время последнего кадра не установлено...")
                    else:
                        time_since_last = current_time - last_frame_time
                        if time_since_last > config.NO_DATA_TIMEOUT:
                            print(f"⚠️  Превышено время ожидания: {time_since_last:.2f} сек")
        
        # Задержка для контроля частоты вывода
        time.sleep(output_delay)


@app.route('/')
def index():
    """Главная страница"""
    # Определяем источник видео для отображения
    global actual_camera_source
    import time as time_module
    
    # Используем фактический источник, если он уже определен
    if actual_camera_source is not None:
        if isinstance(actual_camera_source, int):
            source_info = f"Веб-камера #{actual_camera_source}"
        else:
            source_info = str(actual_camera_source)
    else:
        # Если камера еще не выбрана, показываем запрошенный источник
        if isinstance(config.VIDEO_SOURCE, int):
            source_info = f"Веб-камера #{config.VIDEO_SOURCE}"
        else:
            source_info = str(config.VIDEO_SOURCE)
    
    # Добавляем timestamp для предотвращения кэширования
    timestamp = int(time_module.time() * 1000)
    
    return render_template('index.html', source_info=source_info, timestamp=timestamp)


@app.route('/video_feed')
def video_feed():
    """Эндпоинт для видеопотока"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )


if __name__ == '__main__':
    # Инициализируем детектор с настройками из config
    # detector уже объявлена как глобальная переменная на уровне модуля
    print(f"Инициализация детектора...")
    print(f"Модель: {config.MODEL_NAME}")
    print(f"Устройство: {config.DEVICE}")
    detector = ObjectDetector(model_name=config.MODEL_NAME, device=config.DEVICE)
    
    # Небольшая задержка, чтобы детектор точно инициализировался
    time.sleep(0.5)
    
    # Запускаем поток захвата видео
    video_thread = threading.Thread(
        target=video_capture_thread, 
        args=(config.VIDEO_SOURCE,), 
        daemon=True
    )
    video_thread.start()
    
    # Небольшая задержка перед запуском сервера, чтобы поток успел начать работу
    time.sleep(1.0)
    
    # Получаем локальный IP-адрес и выводим информацию
    local_ip = get_local_ip()
    print_server_info(local_ip, config.FLASK_PORT)
    
    # Запускаем Flask сервер
    app.run(
        host=config.FLASK_HOST, 
        port=config.FLASK_PORT, 
        debug=config.FLASK_DEBUG, 
        threaded=True
    )
