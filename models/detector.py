"""
Модуль для детекции объектов с помощью YOLOv8
"""

from ultralytics import YOLO
import torch
import cv2
from typing import List, Optional, Set, Dict
from collections import Counter


class ObjectDetector:
    """Класс для детекции объектов с помощью YOLOv8"""
    
    def __init__(self, model_name='yolov8n.pt', device='cpu', model_path: str = '', allowed_classes: Optional[List[str]] = None):
        """
        Инициализация детектора
        
        Args:
            model_name: имя модели YOLO (yolov8n.pt, yolov8s.pt и т.д.)
            device: устройство для выполнения ('cpu', 'cuda', 'mps')
            model_path: путь к пользовательским весам (если указан — используется он)
            allowed_classes: список разрешенных классов по именам (пусто/None = все)
        """
        self.model_name = model_name
        self.model_path = model_path.strip() if model_path else ''
        self.device = self._validate_device(device)
        self.allowed_classes: Optional[Set[str]] = None
        if allowed_classes:
            self.allowed_classes = {name.strip().lower() for name in allowed_classes if name and isinstance(name, str)}
        self.model = None
        self.last_counts: Dict[str, int] = {}
        self._load_model()
    
    def _validate_device(self, device):
        """Проверяет и валидирует устройство"""
        device = device.lower()
        
        if device == 'cuda':
            if not torch.cuda.is_available():
                print("⚠️  CUDA недоступен, переключаюсь на CPU")
                return 'cpu'
            print(f"✅ Используется GPU: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        elif device == 'mps':
            if not torch.backends.mps.is_available():
                print("⚠️  MPS недоступен, переключаюсь на CPU")
                return 'cpu'
            print("✅ Используется Apple Silicon GPU (MPS)")
            return 'mps'
        else:
            print("✅ Используется CPU")
            return 'cpu'
    
    def _load_model(self):
        """Загружает модель YOLOv8"""
        chosen = self.model_path if self.model_path else self.model_name
        print(f"Загрузка модели {chosen}...")
        if self.model_path and self.model_path.lower().endswith('.pth'):
            print("ℹ️  Задан .pth. Он должен быть совместим с Ultralytics YOLO. "
                  "Если это чистый PyTorch state_dict, экспортируйте в .pt/ONNX.")
        try:
            self.model = YOLO(chosen)
            print(f"Модель {chosen} загружена успешно!")
            print(f"Устройство: {self.device.upper()}")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            raise
    
    def detect(self, frame, verbose=False, return_detections=False, use_tracking=False):
        """
        Детектирует объекты на кадре с опциональным трекингом
        
        Args:
            frame: кадр изображения (numpy array)
            verbose: выводить ли подробную информацию
            return_detections: если True, возвращает также список детекций/треков
            use_tracking: если True, использует YOLO Track с ByteTrack
            
        Returns:
            Обработанный кадр с нарисованными bounding boxes
            Если return_detections=True: (frame, tracked_objects_list)
            tracked_objects_list содержит: track_id, bbox, class, confidence
        """
        if self.model is None or frame is None:
            return (frame, []) if return_detections else frame
        
        try:
            self.last_counts = {}
            
            if use_tracking:
                # Используем YOLO Track с ByteTrack
                results = self.model.track(
                    frame, 
                    device=self.device, 
                    verbose=verbose,
                    persist=True,  # Сохраняем трекер между кадрами
                    conf=0.1,  # Низкий порог для ByteTrack
                    iou=0.5
                )
            else:
                # Обычная детекция
                results = self.model(frame, device=self.device, verbose=verbose)
            
            res = results[0]

            # Получаем имена классов
            names = res.names if hasattr(res, 'names') else getattr(self.model, 'names', {})
            if isinstance(names, dict):
                idx_to_name = names
            else:
                idx_to_name = {i: n for i, n in enumerate(names)} if names else {}

            # Подготовка списка отслеживаемых объектов
            tracked_objects_list = []
            
            if res.boxes is not None and len(res.boxes) > 0:
                boxes = res.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy() if boxes.conf is not None else None
                cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None
                track_ids = boxes.id.cpu().numpy().astype(int) if hasattr(boxes, 'id') and boxes.id is not None else None
                
                for i in range(len(xyxy)):
                    class_id = int(cls[i]) if cls is not None else -1
                    class_name = str(idx_to_name.get(class_id, str(class_id))).lower()
                    display_name = idx_to_name.get(class_id, str(class_id))
                    score = float(conf[i]) if conf is not None else 0.0
                    x1, y1, x2, y2 = xyxy[i]
                    track_id = int(track_ids[i]) if track_ids is not None else None
                    
                    # Фильтруем по разрешенным классам
                    if not self.allowed_classes or class_name in self.allowed_classes:
                        tracked_objects_list.append({
                            'track_id': track_id,
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'class': display_name,
                            'confidence': score
                        })

            # Если фильтр не задан, используем стандартную визуализацию
            if not self.allowed_classes or len(self.allowed_classes) == 0:
                annotated = res.plot()
                counts = Counter()
                if res.boxes is not None and len(res.boxes) > 0:
                    cls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else None
                    if cls is not None:
                        for class_id in cls:
                            display_name = str(idx_to_name.get(int(class_id), str(class_id))) if isinstance(idx_to_name, dict) else str(class_id)
                            counts[display_name] += 1
                self.last_counts = dict(counts)
                return (annotated, tracked_objects_list) if return_detections else annotated

            # Рисуем только разрешенные классы
            annotated = frame.copy()
            counts = Counter()

            for obj in tracked_objects_list:
                x1, y1, x2, y2 = [int(coord) for coord in obj['bbox']]
                display_name = obj['class']
                score = obj['confidence']
                track_id = obj.get('track_id')
                
                # Рисуем бокс
                color = (0, 255, 0) if track_id is not None else (0, 200, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Формируем метку
                if track_id is not None:
                    label = f"ID:{track_id} {display_name} {score:.2f}"
                else:
                    label = f"{display_name} {score:.2f}"
                
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
                cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                counts[display_name] += 1

            self.last_counts = dict(counts)
            return (annotated, tracked_objects_list) if return_detections else annotated
        except Exception as e:
            print(f"Ошибка детекции: {e}")
            return (frame, []) if return_detections else frame
    
    def get_device_info(self):
        """Возвращает информацию об используемом устройстве"""
        if self.device == 'cuda':
            return f"GPU: {torch.cuda.get_device_name(0)}"
        elif self.device == 'mps':
            return "Apple Silicon GPU (MPS)"
        else:
            return "CPU"

