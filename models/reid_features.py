"""
Модуль для извлечения Re-ID признаков объектов
Используется для различения похожих объектов одного класса
"""

import cv2
import numpy as np
from collections import deque


class ReIDFeatureExtractor:
    """Извлекает визуальные признаки для Re-ID"""
    
    def __init__(self, feature_dim=128, history_size=10):
        """
        Инициализация экстрактора признаков
        
        Args:
            feature_dim: размерность признакового вектора
            history_size: размер истории признаков для сглаживания
        """
        self.feature_dim = feature_dim
        self.history_size = history_size
    
    def extract_features(self, frame, bbox):
        """
        Извлекает признаки из области объекта
        
        Args:
            frame: кадр (BGR)
            bbox: [x1, y1, x2, y2] - bounding box
            
        Returns:
            features: нормализованный вектор признаков
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        h, w = frame.shape[:2]
        
        # Обрезаем координаты до границ кадра
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        # Извлекаем область объекта
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        # Комбинируем несколько типов признаков
        features = []
        
        # 1. Цветовые гистограммы (HSV)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv_roi], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv_roi], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv_roi], [2], None, [16], [0, 256])
        features.extend(hist_h.flatten())
        features.extend(hist_s.flatten())
        features.extend(hist_v.flatten())
        
        # 2. Текстура (LBP-like признаки через градиенты)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_hist = np.histogram(gradient_magnitude.flatten(), bins=16, range=(0, 255))[0]
        features.extend(gradient_hist)
        
        # 3. Геометрические признаки (отношение сторон, площадь)
        aspect_ratio = (x2 - x1) / max(y2 - y1, 1.0)
        area = (x2 - x1) * (y2 - y1)
        normalized_area = area / (w * h) if w * h > 0 else 0
        features.extend([aspect_ratio, normalized_area])
        
        # 4. Центральные моменты (упрощенная версия)
        moments = cv2.moments(gray_roi)
        if moments['m00'] > 0:
            cx_norm = moments['m10'] / moments['m00'] / max(x2 - x1, 1)
            cy_norm = moments['m01'] / moments['m00'] / max(y2 - y1, 1)
        else:
            cx_norm = cy_norm = 0.5
        features.extend([cx_norm, cy_norm])
        
        # Преобразуем в массив и нормализуем
        features = np.array(features, dtype=np.float32)
        
        # Обрезаем или дополняем до нужной размерности
        if len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        elif len(features) < self.feature_dim:
            # Дополняем нулями
            padding = np.zeros(self.feature_dim - len(features), dtype=np.float32)
            features = np.concatenate([features, padding])
        
        # Нормализация L2
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features.astype(np.float32)
    
    def compute_similarity(self, features1, features2):
        """
        Вычисляет схожесть между двумя векторами признаков
        
        Args:
            features1: первый вектор признаков
            features2: второй вектор признаков
            
        Returns:
            similarity: значение от 0 до 1 (1 = идентичны)
        """
        if features1 is None or features2 is None:
            return 0.0
        
        # Косинусное сходство
        dot_product = np.dot(features1, features2)
        return max(0.0, min(1.0, (dot_product + 1.0) / 2.0))  # Нормализуем к [0, 1]


class ReIDTracker:
    """Трекер Re-ID признаков для объекта"""
    
    def __init__(self, feature_dim=128, history_size=10):
        """
        Инициализация трекера признаков
        
        Args:
            feature_dim: размерность признаков
            history_size: размер истории признаков
        """
        self.feature_dim = feature_dim
        self.feature_history = deque(maxlen=history_size)
        self.current_features = None
    
    def update(self, features):
        """Обновляет историю признаков"""
        if features is not None:
            self.feature_history.append(features)
            self.current_features = features
    
    def get_averaged_features(self):
        """Возвращает усредненные признаки из истории"""
        if len(self.feature_history) == 0:
            return self.current_features
        
        # Усредняем по истории
        avg_features = np.mean(list(self.feature_history), axis=0)
        
        # Нормализация
        norm = np.linalg.norm(avg_features)
        if norm > 0:
            avg_features = avg_features / norm
        
        return avg_features.astype(np.float32)
    
    def compute_similarity(self, other_features):
        """Вычисляет схожесть с другими признаками"""
        if self.current_features is None or other_features is None:
            return 0.0
        
        # Используем усредненные признаки
        avg_features = self.get_averaged_features()
        if avg_features is None:
            return 0.0
        
        # Косинусное сходство
        dot_product = np.dot(avg_features, other_features)
        return max(0.0, min(1.0, (dot_product + 1.0) / 2.0))

