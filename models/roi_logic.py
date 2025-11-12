"""
Модуль ROI-логики для отслеживания потерянных объектов
Отслеживает потерю объектов, прогнозирует позицию, формирует ROI и проверяет повторное появление
"""

import numpy as np
import cv2
import time
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ObjectStatus(Enum):
    """Статусы объекта"""
    ACTIVE = "active"      # Объект активно отслеживается
    LOST = "lost"         # Объект потерян, но предсказывается
    REAPPEARED = "reappeared"  # Объект появился снова
    DELETED = "deleted"   # Объект удален из трекинга


class LostObject:
    """Класс для отслеживания потерянного объекта"""
    
    def __init__(self, track_id: int, bbox: List[float], class_name: str, 
                 confidence: float, frame_id: int = 0):
        """
        Инициализация потерянного объекта
        
        Args:
            track_id: ID трека от YOLO Track
            bbox: [x1, y1, x2, y2] - последний известный bounding box
            class_name: имя класса объекта
            confidence: уверенность детекции
            frame_id: ID кадра, когда объект был потерян
        """
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.lost_frame_id = frame_id
        self.last_seen_bbox = bbox.copy()
        self.status = ObjectStatus.LOST
        
        # История позиций для прогнозирования
        self.position_history = [self._bbox_to_center(bbox)]
        self.max_history_length = 10
        
        # Прогнозируемая позиция
        self.predicted_bbox = bbox.copy()
        self.predicted_roi = None
        
        # Время потери
        self.lost_time = time.time()
        self.reappeared_time = None
        
        # Счетчики
        self.frames_lost = 0
        self.frames_since_reappearance = 0
        
        # Простая модель движения (линейная экстраполяция)
        self.velocity = [0.0, 0.0]  # [vx, vy]
        self._update_velocity()
    
    def _bbox_to_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Конвертирует bbox в центр"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    
    def _center_to_bbox(self, center: Tuple[float, float], w: float, h: float) -> List[float]:
        """Конвертирует центр обратно в bbox"""
        cx, cy = center
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
    
    def _update_velocity(self):
        """Обновляет скорость на основе истории позиций"""
        if len(self.position_history) < 2:
            self.velocity = [0.0, 0.0]
            return
        
        # Вычисляем среднюю скорость за последние кадры
        recent_positions = self.position_history[-min(5, len(self.position_history)):]
        if len(recent_positions) < 2:
            self.velocity = [0.0, 0.0]
            return
        
        # Вычисляем среднюю скорость
        velocities = []
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            velocities.append([dx, dy])
        
        if velocities:
            avg_velocity = np.mean(velocities, axis=0)
            self.velocity = [float(avg_velocity[0]), float(avg_velocity[1])]
    
    def update_prediction(self, frame_id: int, decay_factor: float = 0.95):
        """
        Обновляет прогнозируемую позицию объекта
        
        Args:
            frame_id: текущий ID кадра
            decay_factor: фактор затухания скорости (0-1)
        """
        self.frames_lost = frame_id - self.lost_frame_id
        
        if len(self.position_history) == 0:
            return
        
        # Получаем последнюю позицию и размер
        last_center = self.position_history[-1]
        last_bbox = self.last_seen_bbox
        w = last_bbox[2] - last_bbox[0]
        h = last_bbox[3] - last_bbox[1]
        
        # Обновляем скорость с затуханием
        self.velocity[0] *= decay_factor
        self.velocity[1] *= decay_factor
        
        # Прогнозируем новую позицию
        predicted_center = (
            last_center[0] + self.velocity[0],
            last_center[1] + self.velocity[1]
        )
        
        # Обновляем прогнозируемый bbox
        self.predicted_bbox = self._center_to_bbox(predicted_center, w, h)
    
    def get_search_roi(self, frame_shape: Tuple[int, int], expansion_factor: float = 1.5) -> Optional[Tuple[int, int, int, int]]:
        """
        Получает ROI для поиска объекта
        
        Args:
            frame_shape: (height, width) форма кадра
            expansion_factor: коэффициент расширения области поиска
            
        Returns:
            (x1, y1, x2, y2) или None если объект вне кадра
        """
        x1, y1, x2, y2 = self.predicted_bbox
        h, w = frame_shape[:2]
        
        # Вычисляем центр и размер
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        box_w = (x2 - x1) * expansion_factor
        box_h = (y2 - y1) * expansion_factor
        
        # Расширяем область поиска
        roi_x1 = max(0, int(cx - box_w / 2))
        roi_y1 = max(0, int(cy - box_h / 2))
        roi_x2 = min(w, int(cx + box_w / 2))
        roi_y2 = min(h, int(cy + box_h / 2))
        
        # Проверяем, что ROI валидна
        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return None
        
        self.predicted_roi = (roi_x1, roi_y1, roi_x2, roi_y2)
        return self.predicted_roi
    
    def is_in_frame(self, frame_shape: Tuple[int, int]) -> bool:
        """Проверяет, находится ли объект в пределах кадра"""
        x1, y1, x2, y2 = self.predicted_bbox
        h, w = frame_shape[:2]
        
        # Проверяем пересечение с кадром
        return not (x2 < 0 or x1 > w or y2 < 0 or y1 > h)
    
    def check_reappearance(self, detected_bbox: List[float], iou_threshold: float = 0.3) -> bool:
        """
        Проверяет, появился ли объект снова
        
        Args:
            detected_bbox: обнаруженный bounding box
            iou_threshold: порог IoU для сопоставления
            
        Returns:
            True если объект появился снова
        """
        iou = self._calculate_iou(self.predicted_bbox, detected_bbox)
        return iou > iou_threshold
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Вычисляет IoU между двумя боксами"""
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
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def mark_reappeared(self, new_bbox: List[float], frame_id: int):
        """Отмечает объект как появившийся снова"""
        self.status = ObjectStatus.REAPPEARED
        self.reappeared_time = time.time()
        self.last_seen_bbox = new_bbox.copy()
        self.position_history.append(self._bbox_to_center(new_bbox))
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
        self._update_velocity()
        logger.info(f"Объект ID {self.track_id} ({self.class_name}) появился снова на кадре {frame_id}")


class ROILogicManager:
    """Менеджер ROI-логики для отслеживания потерянных объектов"""

    def __init__(
        self,
        lost_timeout: float = 5.0,
        event_timeout: float = 10.0,
        roi_expansion: float = 1.5,
        iou_threshold: float = 0.3,
        recheck_frames: int = 10,
        neighbor_offset_ratio: float = 0.5,
        confirmation_frames: int = 3,
    ):
        """
        Инициализация менеджера ROI-логики

        Args:
            lost_timeout: время в секундах до отправки сигнала о потере объекта
            event_timeout: время в секундах до генерации события о длительной потере
            roi_expansion: коэффициент расширения ROI
            iou_threshold: порог IoU для сопоставления при повторном появлении
            recheck_frames: количество кадров без объекта до повторной проверки ROI
            neighbor_offset_ratio: смещение соседних областей при поиске, доля ширины/высоты ROI
            confirmation_frames: минимальное число последовательных кадров для подтверждения объекта
        """
        self.lost_timeout = lost_timeout
        self.event_timeout = event_timeout
        self.roi_expansion = roi_expansion
        self.iou_threshold = iou_threshold
        self.recheck_frames = recheck_frames
        self.neighbor_offset_ratio = neighbor_offset_ratio
        self.confirmation_frames = max(1, confirmation_frames)

        # Словари состояний объектов
        self.lost_objects: Dict[int, LostObject] = {}
        self.deleted_objects: Dict[int, LostObject] = {}
        self.active_objects: Dict[int, Dict] = {}

        # События и статистика
        self.events: List[Dict] = []
        self.stats = {
            'total_lost': 0,
            'total_reappeared': 0,
            'total_events': 0,
        }
        # Статистика по подтверждённым объектам: class_name -> {'total_detected': count}
        self.class_statistics = defaultdict(lambda: {'total_detected': 0})

        self.current_frame_id = 0

    def update(
        self,
        tracked_objects: List[Dict],
        frame: Optional[np.ndarray] = None,
        frame_id: Optional[int] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Обновляет состояние ROI-логики.

        Args:
            tracked_objects: текущие объекты от YOLO Track
            frame: исходный кадр (для ROI-проверок)
            frame_id: внешний ID кадра (если не указан, увеличивается автоматически)

        Returns:
            lost_objects_list: список потерянных объектов для визуализации
            new_events: события, сформированные на этом шаге
        """
        if frame_id is not None:
            self.current_frame_id = frame_id
        else:
            self.current_frame_id += 1

        frame_shape = frame.shape if frame is not None else None

        # Сохраняем предыдущее состояние активных объектов для анализа потерь
        old_active_objects = self.active_objects
        self.active_objects = {}
        current_track_ids = set()

        # Обновляем активные объекты и счётчики подтверждения
        for obj in tracked_objects:
            track_id = obj.get('track_id')
            if track_id is None:
                continue

            current_track_ids.add(track_id)
            class_name = obj.get('class', 'unknown')
            prev_state = old_active_objects.get(track_id, {})

            streak = prev_state.get('streak', 0) + 1
            confirmed_before = prev_state.get('confirmed', False)
            confirmation_logged = prev_state.get('confirmation_logged', False)
            confirmed_now = confirmed_before or streak >= self.confirmation_frames

            if confirmed_now and not confirmation_logged:
                self.class_statistics[class_name]['total_detected'] += 1
                confirmation_logged = True

            self.active_objects[track_id] = {
                'bbox': obj['bbox'],
                'class': class_name,
                'confidence': obj.get('confidence', 0.0),
                'frame_id': self.current_frame_id,
                'streak': streak,
                'confirmed': confirmed_now,
                'confirmation_logged': confirmation_logged,
            }

            # Если объект был потерян или удалён, фиксируем возвращение
            if track_id in self.lost_objects:
                lost_obj = self.lost_objects.pop(track_id)
                lost_obj.mark_reappeared(obj['bbox'], self.current_frame_id)
                self.stats['total_reappeared'] += 1
            if track_id in self.deleted_objects:
                self.deleted_objects.pop(track_id, None)

        # Обрабатываем объекты, которые исчезли из активного списка
        for track_id, prev_obj in old_active_objects.items():
            if track_id in current_track_ids:
                continue

            class_name = prev_obj.get('class', 'unknown')
            bbox = prev_obj.get('bbox')
            confidence = prev_obj.get('confidence', 0.0)
            frame_id_prev = prev_obj.get('frame_id', self.current_frame_id)
            confirmed = prev_obj.get('confirmed', False)
            streak = prev_obj.get('streak', 0)

            # Если объект не был подтверждён, статистику не обновляем
            if not confirmed:
                continue

            lost_obj = self.lost_objects.get(track_id)
            if lost_obj is None:
                lost_obj = LostObject(
                    track_id=track_id,
                    bbox=bbox,
                    class_name=class_name,
                    confidence=confidence,
                    frame_id=frame_id_prev,
                )
                self.lost_objects[track_id] = lost_obj
                self.stats['total_lost'] += 1
                logger.info(
                    "Объект ID %s (%s) потерян на кадре %s",
                    track_id,
                    class_name,
                    self.current_frame_id,
                )
            else:
                lost_obj.last_seen_bbox = bbox.copy()

            center = (
                (bbox[0] + bbox[2]) / 2.0,
                (bbox[1] + bbox[3]) / 2.0,
            )
            lost_obj.position_history.append(center)
            if len(lost_obj.position_history) > lost_obj.max_history_length:
                lost_obj.position_history.pop(0)

        # Обновляем прогнозы потерянных объектов и выполняем ROI-проверки
        deleted_ids = set()
        new_events: List[Dict] = []
        roi_candidates: List[Tuple[int, Tuple[int, int, int, int]]] = []

        for track_id, lost_obj in list(self.lost_objects.items()):
            lost_obj.update_prediction(self.current_frame_id)

            time_lost = time.time() - lost_obj.lost_time
            if time_lost >= self.event_timeout:
                event = {
                    'track_id': track_id,
                    'class': lost_obj.class_name,
                    'message': (
                        f"Объект с ID {track_id} потерян на длительное время "
                        f"({time_lost:.1f} сек)"
                    ),
                    'time_lost': time_lost,
                    'frames_lost': lost_obj.frames_lost,
                    'timestamp': time.time(),
                }
                new_events.append(event)
                self.events.append(event)
                self.stats['total_events'] += 1

            if frame_shape is not None and lost_obj.frames_lost >= self.recheck_frames:
                roi = lost_obj.get_search_roi(frame_shape, self.roi_expansion)
                if roi is not None:
                    roi_candidates.append((track_id, roi))

        # Группируем пересекающиеся ROI и выполняем поиск объектов
        if frame_shape is not None and roi_candidates:
            roi_groups = self._group_roi_candidates(roi_candidates)
            for group in roi_groups:
                roi = group['roi']
                found = self._find_objects_in_roi(roi, tracked_objects)

                if not found:
                    for neighbor_roi in self._generate_neighbor_rois(roi, frame_shape):
                        if self._find_objects_in_roi(neighbor_roi, tracked_objects):
                            found = True
                            break

                if not found:
                    # Объект не найден — переносим в удалённые
                    for track_id in group['members']:
                        if track_id in self.lost_objects:
                            lost_obj = self.lost_objects.pop(track_id)
                            lost_obj.status = ObjectStatus.DELETED
                            self.deleted_objects[track_id] = lost_obj
                            deleted_ids.add(track_id)
                            logger.warning(
                                "Объект ID %s (%s) помечен как удалён после ROI проверки",
                                track_id,
                                lost_obj.class_name,
                            )
                            event = {
                                'track_id': track_id,
                                'class': lost_obj.class_name,
                                'message': f"Объект с ID {track_id} удалён после проверки ROI",
                                'time_lost': time.time() - lost_obj.lost_time,
                                'frames_lost': lost_obj.frames_lost,
                                'timestamp': time.time(),
                            }
                            new_events.append(event)
                            self.events.append(event)
                            self.stats['total_events'] += 1

        # Формируем список потерянных объектов для визуализации
        lost_objects_list: List[Dict] = []
        current_time = time.time()
        for track_id, lost_obj in self.lost_objects.items():
            time_lost = current_time - lost_obj.lost_time
            lost_objects_list.append({
                'track_id': track_id,
                'bbox': lost_obj.predicted_bbox,
                'class': lost_obj.class_name,
                'confidence': lost_obj.confidence,
                'status': lost_obj.status.value,
                'time_lost': time_lost,
                'frames_lost': lost_obj.frames_lost,
                'roi': lost_obj.predicted_roi,
            })

        # Ограничиваем количество событий
        if len(self.events) > 100:
            self.events = self.events[-100:]
        if len(new_events) > 0 and len(new_events) > 100:
            new_events = new_events[-100:]

        return lost_objects_list, new_events

    def _find_objects_in_roi(self, roi: Tuple[int, int, int, int], tracked_objects: List[Dict]) -> bool:
        x1, y1, x2, y2 = roi
        for obj in tracked_objects:
            bbox = obj.get('bbox')
            if bbox and self._intersect(roi, bbox):
                return True
        return False

    @staticmethod
    def _intersect(roi: Tuple[int, int, int, int], bbox: List[float], min_overlap: float = 0.1) -> bool:
        rx1, ry1, rx2, ry2 = roi
        bx1, by1, bx2, by2 = bbox
        ix1 = max(rx1, int(bx1))
        iy1 = max(ry1, int(by1))
        ix2 = min(rx2, int(bx2))
        iy2 = min(ry2, int(by2))
        if ix2 <= ix1 or iy2 <= iy1:
            return False
        roi_area = (rx2 - rx1) * (ry2 - ry1)
        inter_area = (ix2 - ix1) * (iy2 - iy1)
        if roi_area <= 0:
            return False
        return inter_area / roi_area >= min_overlap

    def _group_roi_candidates(
        self,
        candidates: List[Tuple[int, Tuple[int, int, int, int]]],
    ) -> List[Dict]:
        groups: List[Dict] = []
        for track_id, roi in candidates:
            merged = False
            for group in groups:
                if self._roi_overlap(group['roi'], roi):
                    group['roi'] = self._merge_roi(group['roi'], roi)
                    group['members'].append(track_id)
                    merged = True
                    break
            if not merged:
                groups.append({'roi': roi, 'members': [track_id]})
        return groups

    @staticmethod
    def _roi_overlap(roi1: Tuple[int, int, int, int], roi2: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = roi1
        a1, b1, a2, b2 = roi2
        return not (x2 <= a1 or a2 <= x1 or y2 <= b1 or b2 <= y1)

    @staticmethod
    def _merge_roi(roi1: Tuple[int, int, int, int], roi2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = roi1
        a1, b1, a2, b2 = roi2
        return min(x1, a1), min(y1, b1), max(x2, a2), max(y2, b2)

    def _generate_neighbor_rois(
        self,
        roi: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int, int],
    ) -> List[Tuple[int, int, int, int]]:
        rx1, ry1, rx2, ry2 = roi
        width = rx2 - rx1
        height = ry2 - ry1
        offset_x = int(width * self.neighbor_offset_ratio)
        offset_y = int(height * self.neighbor_offset_ratio)

        candidates = []
        candidates.append(self._clamp_roi((rx1 - offset_x, ry1, rx2 - offset_x, ry2), frame_shape))  # left
        candidates.append(self._clamp_roi((rx1 + offset_x, ry1, rx2 + offset_x, ry2), frame_shape))  # right
        candidates.append(self._clamp_roi((rx1, ry1 - offset_y, rx2, ry2 - offset_y), frame_shape))  # up
        candidates.append(self._clamp_roi((rx1, ry1 + offset_y, rx2, ry2 + offset_y), frame_shape))  # down
        return [c for c in candidates if c is not None]

    @staticmethod
    def _clamp_roi(roi: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = roi
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def get_lost_objects(self) -> List[Dict]:
        """Возвращает текущие потерянные объекты"""
        current_time = time.time()
        return [
            {
                'track_id': obj.track_id,
                'bbox': obj.predicted_bbox,
                'class': obj.class_name,
                'status': obj.status.value,
                'time_lost': current_time - obj.lost_time,
                'frames_lost': obj.frames_lost,
            }
            for obj in self.lost_objects.values()
        ]

    def get_events(self) -> List[Dict]:
        """Возвращает список всех событий"""
        return self.events.copy()

    def get_statistics(self) -> Dict:
        """
        Возвращает статистику по подтверждённым объектам.
        Формат: {class_name: {'active': count, 'total_detected': count}}
        """
        active_by_class = defaultdict(int)
        for data in self.active_objects.values():
            if not data.get('confirmed'):
                continue
            class_name = data.get('class', 'unknown')
            active_by_class[class_name] += 1

        result = {}
        for class_name, active_count in active_by_class.items():
            result[class_name] = {
                'active': active_count,
                'total_detected': self.class_statistics[class_name]['total_detected'],
            }

        for class_name, stats in self.class_statistics.items():
            if class_name not in result:
                result[class_name] = {
                    'active': 0,
                    'total_detected': stats['total_detected'],
                }

        return result

    def reset(self):
        """Сбрасывает состояние менеджера"""
        self.lost_objects.clear()
        self.deleted_objects.clear()
        self.active_objects.clear()
        self.events.clear()
        self.class_statistics.clear()
        self.current_frame_id = 0
        logger.info("ROI-логика сброшена")

