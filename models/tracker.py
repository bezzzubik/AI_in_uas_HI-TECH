"""
Модуль трекинга объектов с фильтром Калмана, компенсацией движения камеры,
Re-ID признаками и Hungarian Algorithm
"""

import numpy as np
import cv2
from collections import defaultdict
import time
from enum import Enum
from scipy.optimize import linear_sum_assignment

from .motion_compensation import MotionCompensator
from .reid_features import ReIDFeatureExtractor, ReIDTracker


class TrackStatus(Enum):
    """Статусы трека объекта"""
    ACTIVE = "active"      # Объект активно отслеживается
    LOST = "lost"         # Объект потерян, но предсказывается
    DELETED = "deleted"   # Объект удален из трекинга


class KalmanTracker:
    """Трекер одного объекта с фильтром Калмана, ускорением и сглаживанием"""
    
    def __init__(self, track_id, bbox, class_name, confidence, frame=None):
        """
        Инициализация трекера для объекта
        
        Args:
            track_id: уникальный ID трека
            bbox: [x1, y1, x2, y2] - bounding box
            class_name: имя класса объекта
            confidence: уверенность детекции
            frame: кадр для извлечения Re-ID признаков
        """
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.last_seen = time.time()
        self.hit_streak = 1  # Количество последовательных обнаружений
        self.status = TrackStatus.ACTIVE
        self.time_since_update = 0
        self.age = 0  # Возраст трека в кадрах
        
        # Конвертируем bbox в формат [cx, cy, w, h]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        
        # Инициализируем фильтр Калмана с расширенным состоянием
        # Состояние: [cx, cy, w, h, vx, vy, ax, ay, cam_dx, cam_dy] 
        # - центр, размер, скорость, ускорение объекта, движение камеры
        self.kf = cv2.KalmanFilter(10, 4)  # 10 состояний, 4 измерения
        
        # Матрица перехода (модель движения с ускорением и учетом камеры)
        dt = 1.0  # временной шаг
        dt2 = dt * dt / 2.0  # для ускорения
        
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, dt, 0,  dt2, 0,   dt, 0],   # cx = cx + vx*dt + ax*dt²/2 + cam_dx*dt
            [0, 1, 0, 0, 0,  dt, 0,   dt2, 0,   dt],  # cy = cy + vy*dt + ay*dt²/2 + cam_dy*dt
            [0, 0, 1, 0, 0,  0,  0,   0,   0,   0],   # w = w
            [0, 0, 0, 1, 0,  0,  0,   0,   0,   0],   # h = h
            [0, 0, 0, 0, 1,  0,  dt,  0,   0,   0],  # vx = vx + ax*dt
            [0, 0, 0, 0, 0,  1,  0,   dt,  0,   0],   # vy = vy + ay*dt
            [0, 0, 0, 0, 0,  0,  1,   0,   0,   0],   # ax = ax
            [0, 0, 0, 0, 0,  0,  0,   1,   0,   0],   # ay = ay
            [0, 0, 0, 0, 0,  0,  0,   0,   1,   0],   # cam_dx = cam_dx
            [0, 0, 0, 0, 0,  0,  0,   0,   0,   1]    # cam_dy = cam_dy
        ], dtype=np.float32)
        
        # Матрица измерения (что мы измеряем)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # измеряем cx
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # измеряем cy
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # измеряем w
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]   # измеряем h
        ], dtype=np.float32)
        
        # Базовые ковариации (будут адаптироваться)
        base_process_noise = 0.03
        base_measurement_noise = 0.1
        
        # Ковариация процесса (неопределенность модели)
        self.kf.processNoiseCov = np.eye(10, dtype=np.float32) * base_process_noise
        # Больше шума для ускорения
        self.kf.processNoiseCov[6, 6] = base_process_noise * 2.0  # ax
        self.kf.processNoiseCov[7, 7] = base_process_noise * 2.0  # ay
        
        # Ковариация измерения (шум измерений)
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * base_measurement_noise
        
        # Ковариация ошибки
        self.kf.errorCovPost = np.eye(10, dtype=np.float32)
        
        # Инициализация состояния
        self.kf.statePre = np.array([cx, cy, w, h, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        self.kf.statePost = np.array([cx, cy, w, h, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        
        # Сглаживание координат (экспоненциальное сглаживание)
        self.smoothing_alpha = 0.7  # Коэффициент сглаживания
        self.smoothed_bbox = bbox.copy()
        
        # История позиций для ROI-проверки и адаптивного шума
        self.position_history = [(cx, cy)]
        self.max_history_length = 10
        
        # Адаптивный шум (на основе неопределенности)
        self.uncertainty = 1.0  # Начальная неопределенность
        self.min_uncertainty = 0.5
        self.max_uncertainty = 3.0
        
        # ROI для проверки при потере
        self.search_roi = None
        self.lost_since = None
        
        # Re-ID трекер признаков
        self.reid_tracker = ReIDTracker()
        if frame is not None:
            feature_extractor = ReIDFeatureExtractor()
            features = feature_extractor.extract_features(frame, bbox)
            self.reid_tracker.update(features)
    
    def _update_adaptive_noise(self):
        """Обновляет адаптивный шум на основе неопределенности трека"""
        # Вычисляем неопределенность на основе ковариации ошибки
        error_cov = self.kf.errorCovPost
        position_uncertainty = np.sqrt(error_cov[0, 0] + error_cov[1, 1])
        
        # Адаптируем неопределенность
        self.uncertainty = np.clip(position_uncertainty, self.min_uncertainty, self.max_uncertainty)
        
        # Обновляем ковариацию процесса на основе неопределенности
        base_noise = 0.03
        adaptive_noise = base_noise * (1.0 + self.uncertainty)
        
        self.kf.processNoiseCov = np.eye(10, dtype=np.float32) * adaptive_noise
        self.kf.processNoiseCov[6, 6] = adaptive_noise * 2.0  # ax
        self.kf.processNoiseCov[7, 7] = adaptive_noise * 2.0  # ay
    
    def update(self, bbox, confidence=None, camera_motion=None, frame=None):
        """
        Обновление трекера новым обнаружением
        
        Args:
            bbox: новый bounding box
            confidence: уверенность детекции
            camera_motion: (dx, dy, rotation) - движение камеры
            frame: кадр для извлечения Re-ID признаков
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        
        # Сглаживание координат (экспоненциальное сглаживание)
        smoothed_x1 = self.smoothing_alpha * x1 + (1 - self.smoothing_alpha) * self.smoothed_bbox[0]
        smoothed_y1 = self.smoothing_alpha * y1 + (1 - self.smoothing_alpha) * self.smoothed_bbox[1]
        smoothed_x2 = self.smoothing_alpha * x2 + (1 - self.smoothing_alpha) * self.smoothed_bbox[2]
        smoothed_y2 = self.smoothing_alpha * y2 + (1 - self.smoothing_alpha) * self.smoothed_bbox[3]
        
        self.smoothed_bbox = [smoothed_x1, smoothed_y1, smoothed_x2, smoothed_y2]
        
        # Используем сглаженные координаты для фильтра Калмана
        smoothed_cx = (smoothed_x1 + smoothed_x2) / 2.0
        smoothed_cy = (smoothed_y1 + smoothed_y2) / 2.0
        smoothed_w = smoothed_x2 - smoothed_x1
        smoothed_h = smoothed_y2 - smoothed_y1
        
        # Обновляем движение камеры в состоянии
        if camera_motion is not None:
            dx, dy, _ = camera_motion
            state = self.kf.statePost.copy()
            state[8] = dx  # cam_dx
            state[9] = dy  # cam_dy
            self.kf.statePost = state
        
        # Измерение (используем сглаженные значения)
        measurement = np.array([smoothed_cx, smoothed_cy, smoothed_w, smoothed_h], dtype=np.float32)
        
        # Обновление фильтра Калмана
        self.kf.correct(measurement)
        
        # Обновляем адаптивный шум
        self._update_adaptive_noise()
        
        # Обновляем историю позиций
        self.position_history.append((float(smoothed_cx), float(smoothed_cy)))
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        
        # Обновляем Re-ID признаки
        if frame is not None:
            feature_extractor = ReIDFeatureExtractor()
            features = feature_extractor.extract_features(frame, self.smoothed_bbox)
            self.reid_tracker.update(features)
        
        self.last_seen = time.time()
        self.hit_streak += 1
        self.time_since_update = 0
        self.status = TrackStatus.ACTIVE
        self.age += 1
        self.lost_since = None
        
        if confidence is not None:
            self.confidence = confidence
    
    def predict(self, camera_motion=None):
        """
        Предсказание следующей позиции с учетом движения камеры
        
        Args:
            camera_motion: (dx, dy, rotation) - движение камеры
        """
        # Обновляем движение камеры перед предсказанием
        if camera_motion is not None:
            dx, dy, _ = camera_motion
            state = self.kf.statePost.copy()
            state[8] = dx  # cam_dx
            state[9] = dy  # cam_dy
            self.kf.statePost = state
        
        prediction = self.kf.predict()
        cx, cy, w, h = prediction[0], prediction[1], prediction[2], prediction[3]
        
        # Конвертируем обратно в формат [x1, y1, x2, y2]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return [x1, y1, x2, y2]
    
    def get_state(self):
        """Получить текущее состояние (предсказанное)"""
        state = self.kf.statePost
        cx, cy, w, h = state[0], state[1], state[2], state[3]
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return [x1, y1, x2, y2]
    
    def get_smoothed_bbox(self):
        """Получить сглаженный bounding box"""
        return self.smoothed_bbox.copy()
    
    def get_search_roi(self, frame_shape, expansion_factor=1.5):
        """
        Получить ROI для поиска потерянного объекта
        
        Args:
            frame_shape: (height, width) форма кадра
            expansion_factor: коэффициент расширения области поиска
            
        Returns:
            (x1, y1, x2, y2) или None если объект вне кадра
        """
        bbox = self.get_state()
        x1, y1, x2, y2 = bbox
        
        # Вычисляем центр и размер
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = (x2 - x1) * expansion_factor
        h = (y2 - y1) * expansion_factor
        
        # Расширяем область поиска
        roi_x1 = max(0, int(cx - w / 2))
        roi_y1 = max(0, int(cy - h / 2))
        roi_x2 = min(frame_shape[1], int(cx + w / 2))
        roi_y2 = min(frame_shape[0], int(cy + h / 2))
        
        # Проверяем, что ROI валидна
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return None
        
        return (roi_x1, roi_y1, roi_x2, roi_y2)
    
    def is_in_frame(self, frame_shape):
        """Проверяет, находится ли объект в пределах кадра"""
        bbox = self.get_state()
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        
        # Проверяем пересечение с кадром
        return not (x2 < 0 or x1 > w or y2 < 0 or y1 > h)


class ObjectTracker:
    """Трекер множества объектов с фильтром Калмана, Re-ID и Hungarian Algorithm"""
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3, lost_timeout=5.0, 
                 enable_motion_compensation=True, motion_method='optical_flow',
                 reid_weight=0.3, iou_weight=0.7):
        """
        Инициализация трекера
        
        Args:
            max_age: максимальное время жизни трека без обновления (в кадрах)
            min_hits: минимальное количество попаданий для подтверждения трека
            iou_threshold: порог IoU для сопоставления детекций с треками
            lost_timeout: время в секундах до отправки сигнала о потере объекта
            enable_motion_compensation: включить компенсацию движения камеры
            motion_method: метод компенсации ('optical_flow' или 'homography')
            reid_weight: вес Re-ID признаков в стоимости сопоставления (0-1)
            iou_weight: вес IoU в стоимости сопоставления (0-1)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.lost_timeout = lost_timeout
        self.reid_weight = reid_weight
        self.iou_weight = iou_weight
        
        self.trackers = {}  # track_id -> KalmanTracker
        self.next_id = 1
        self.lost_objects = []  # Список потерянных объектов для уведомлений
        
        # Компенсатор движения камеры
        self.motion_compensator = None
        if enable_motion_compensation:
            self.motion_compensator = MotionCompensator(method=motion_method)
            print(f"✅ Компенсация движения камеры включена (метод: {motion_method})")
        
        # Re-ID экстрактор признаков
        self.reid_extractor = ReIDFeatureExtractor()
        
        # Статистика по классам
        self.class_statistics = defaultdict(lambda: {'active': 0, 'total_detected': 0})
    
    def _iou(self, box1, box2):
        """Вычисление IoU между двумя боксами"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Пересечение
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Объединение
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _associate_detections_to_trackers(self, detections, trackers, camera_motion=None, frame=None):
        """
        Сопоставление детекций с трекерами используя Hungarian Algorithm
        
        Args:
            detections: список детекций
            trackers: словарь трекеров
            camera_motion: движение камеры
            frame: текущий кадр для Re-ID
            
        Returns:
            matched_indices: список пар (det_idx, track_id)
            unmatched_detections: список индексов несопоставленных детекций
            unmatched_trackers: список ID несопоставленных трекеров
        """
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(trackers.keys())
        
        # Вычисляем матрицу стоимости
        cost_matrix = np.zeros((len(detections), len(trackers)))
        track_ids_list = list(trackers.keys())
        
        for d, det in enumerate(detections):
            for t, track_id in enumerate(track_ids_list):
                tracker = trackers[track_id]
                
                # Предсказанная позиция
                predicted_box = tracker.predict(camera_motion)
                
                # IoU стоимость (чем больше IoU, тем меньше стоимость)
                iou = self._iou(det['bbox'], predicted_box)
                iou_cost = 1.0 - iou  # Преобразуем в стоимость
                
                # Re-ID стоимость (если доступен кадр)
                reid_cost = 0.0
                if frame is not None:
                    # Извлекаем признаки из детекции
                    det_features = self.reid_extractor.extract_features(frame, det['bbox'])
                    # Вычисляем схожесть с трекером
                    similarity = tracker.reid_tracker.compute_similarity(det_features)
                    reid_cost = 1.0 - similarity  # Преобразуем в стоимость
                
                # Комбинированная стоимость
                cost = self.iou_weight * iou_cost + self.reid_weight * reid_cost
                cost_matrix[d, t] = cost
        
        # Hungarian Algorithm для оптимального сопоставления
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Фильтруем сопоставления по порогу
        matched_indices = []
        matched_detections = set()
        matched_trackers = set()
        
        for row_idx, col_idx in zip(row_indices, col_indices):
            cost = cost_matrix[row_idx, col_idx]
            track_id = track_ids_list[col_idx]
            
            # Проверяем порог (IoU должен быть достаточным)
            predicted_box = trackers[track_id].predict(camera_motion)
            iou = self._iou(detections[row_idx]['bbox'], predicted_box)
            
            if iou > self.iou_threshold and cost < 1.0:  # Допустимая стоимость
                matched_indices.append((row_idx, track_id))
                matched_detections.add(row_idx)
                matched_trackers.add(track_id)
        
        # Находим несопоставленные
        unmatched_detections = [d for d in range(len(detections)) if d not in matched_detections]
        unmatched_trackers = [track_id for track_id in track_ids_list if track_id not in matched_trackers]
        
        return matched_indices, unmatched_detections, unmatched_trackers
    
    def update(self, detections, frame=None):
        """
        Обновление трекера новыми детекциями
        
        Args:
            detections: список словарей с ключами 'bbox', 'class', 'confidence'
                      bbox в формате [x1, y1, x2, y2]
            frame: текущий кадр для компенсации движения и Re-ID (опционально)
        
        Returns:
            tracked_objects: список отслеживаемых объектов с ID
            predicted_objects: список предсказанных объектов (потерянных)
            lost_notifications: список уведомлений о потерянных объектах
            statistics: статистика по классам
        """
        current_time = time.time()
        
        # Вычисляем движение камеры если включена компенсация
        camera_motion = None
        if self.motion_compensator is not None and frame is not None:
            transform = self.motion_compensator.compute_transform(frame)
            camera_motion = self.motion_compensator.get_camera_motion()
        
        # Предсказываем позиции для всех существующих трекеров
        for track_id, tracker in self.trackers.items():
            tracker.predict(camera_motion)
            tracker.time_since_update += 1
            tracker.age += 1
        
        # Сопоставляем детекции с трекерами используя Hungarian Algorithm
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(
            detections, self.trackers, camera_motion, frame
        )
        
        # Обновляем совпавшие трекеры
        for det_idx, track_id in matched:
            det = detections[det_idx]
            self.trackers[track_id].update(
                det['bbox'], 
                det.get('confidence'), 
                camera_motion,
                frame
            )
            
            # Обновляем статус
            if self.trackers[track_id].status == TrackStatus.LOST:
                self.trackers[track_id].status = TrackStatus.ACTIVE
                self.trackers[track_id].lost_since = None
            
            # Обновляем статистику
            class_name = det['class']
            if self.trackers[track_id].status == TrackStatus.ACTIVE:
                self.class_statistics[class_name]['active'] = max(
                    self.class_statistics[class_name]['active'],
                    sum(1 for t in self.trackers.values() 
                        if t.class_name == class_name and t.status == TrackStatus.ACTIVE)
                )
        
        # Создаем новые трекеры для несопоставленных детекций
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            track_id = self.next_id
            self.next_id += 1
            self.trackers[track_id] = KalmanTracker(
                track_id, det['bbox'], det['class'], det.get('confidence', 0.5), frame
            )
            
            # Обновляем статистику
            class_name = det['class']
            self.class_statistics[class_name]['total_detected'] += 1
        
        # Обрабатываем несопоставленные трекеры (потерянные объекты)
        tracked_objects = []
        predicted_objects = []
        lost_notifications = []
        
        trackers_to_remove = []
        frame_shape = frame.shape if frame is not None else None
        
        for track_id, tracker in self.trackers.items():
            if track_id in unmatched_trks:
                # Объект потерян
                if tracker.status == TrackStatus.ACTIVE:
                    tracker.status = TrackStatus.LOST
                    tracker.lost_since = current_time if tracker.lost_since is None else tracker.lost_since
                
                time_lost = current_time - tracker.lost_since if tracker.lost_since else tracker.time_since_update
                
                # ROI-проверка: проверяем, находится ли объект в кадре
                in_frame = True
                if frame_shape is not None:
                    in_frame = tracker.is_in_frame(frame_shape)
                    if in_frame and tracker.search_roi is None:
                        tracker.search_roi = tracker.get_search_roi(frame_shape)
                
                # Проверяем, не пора ли отправить уведомление
                if time_lost >= self.lost_timeout and tracker.status != TrackStatus.DELETED:
                    if track_id not in [n['track_id'] for n in self.lost_objects]:
                        notification = {
                            'track_id': track_id,
                            'class': tracker.class_name,
                            'message': f"Объект с ID {track_id} потерян",
                            'roi': tracker.search_roi if in_frame else None,
                            'time_lost': time_lost
                        }
                        self.lost_objects.append(notification)
                        lost_notifications.append(notification)
                
                # Предсказываем позицию потерянного объекта
                if in_frame:
                    predicted_bbox = tracker.get_state()
                    predicted_objects.append({
                        'track_id': track_id,
                        'bbox': predicted_bbox,
                        'class': tracker.class_name,
                        'confidence': tracker.confidence,
                        'is_predicted': True,
                        'roi': tracker.search_roi
                    })
                
                # Удаляем старые трекеры
                if tracker.hit_streak < self.min_hits or tracker.time_since_update > self.max_age:
                    tracker.status = TrackStatus.DELETED
                    trackers_to_remove.append(track_id)
            else:
                # Объект отслеживается
                if tracker.hit_streak >= self.min_hits and tracker.status == TrackStatus.ACTIVE:
                    bbox = tracker.get_smoothed_bbox()  # Используем сглаженные координаты
                    tracked_objects.append({
                        'track_id': track_id,
                        'bbox': bbox,
                        'class': tracker.class_name,
                        'confidence': tracker.confidence,
                        'is_predicted': False,
                        'status': 'active'
                    })
        
        # Удаляем старые трекеры
        for track_id in trackers_to_remove:
            class_name = self.trackers[track_id].class_name
            del self.trackers[track_id]
            # Обновляем статистику активных объектов
            self.class_statistics[class_name]['active'] = sum(
                1 for t in self.trackers.values() 
                if t.class_name == class_name and t.status == TrackStatus.ACTIVE
            )
        
        # Формируем статистику
        statistics = {}
        for class_name, stats in self.class_statistics.items():
            active_count = sum(
                1 for t in self.trackers.values() 
                if t.class_name == class_name and t.status == TrackStatus.ACTIVE
            )
            statistics[class_name] = {
                'active': active_count,
                'total_detected': stats['total_detected']
            }
        
        return tracked_objects, predicted_objects, lost_notifications, statistics
    
    def get_lost_objects(self):
        """Получить список всех потерянных объектов"""
        return self.lost_objects.copy()
    
    def get_statistics(self):
        """Получить статистику по классам"""
        stats = {}
        for class_name, data in self.class_statistics.items():
            active_count = sum(
                1 for t in self.trackers.values() 
                if t.class_name == class_name and t.status == TrackStatus.ACTIVE
            )
            stats[class_name] = {
                'active': active_count,
                'total_detected': data['total_detected']
            }
        return stats
