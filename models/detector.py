"""
Модуль для детекции объектов с помощью YOLOv8
"""

from ultralytics import YOLO
import torch


class ObjectDetector:
    """Класс для детекции объектов с помощью YOLOv8"""
    
    def __init__(self, model_name='yolov8n.pt', device='cpu'):
        """
        Инициализация детектора
        
        Args:
            model_name: имя модели YOLO (yolov8n.pt, yolov8s.pt и т.д.)
            device: устройство для выполнения ('cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.device = self._validate_device(device)
        self.model = None
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
        print(f"Загрузка модели {self.model_name}...")
        try:
            self.model = YOLO(self.model_name)
            print(f"Модель {self.model_name} загружена успешно!")
            print(f"Устройство: {self.device.upper()}")
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            raise
    
    def detect(self, frame, verbose=False):
        """
        Детектирует объекты на кадре
        
        Args:
            frame: кадр изображения (numpy array)
            verbose: выводить ли подробную информацию
            
        Returns:
            Обработанный кадр с нарисованными bounding boxes
        """
        if self.model is None or frame is None:
            return frame
        
        try:
            # Запускаем детекцию на указанном устройстве
            results = self.model(frame, device=self.device, verbose=verbose)
            
            # Рисуем результаты на кадре
            annotated_frame = results[0].plot()
            
            return annotated_frame
        except Exception as e:
            print(f"Ошибка детекции: {e}")
            return frame
    
    def get_device_info(self):
        """Возвращает информацию об используемом устройстве"""
        if self.device == 'cuda':
            return f"GPU: {torch.cuda.get_device_name(0)}"
        elif self.device == 'mps':
            return "Apple Silicon GPU (MPS)"
        else:
            return "CPU"

