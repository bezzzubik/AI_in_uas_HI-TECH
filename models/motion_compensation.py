"""
Модуль компенсации движения камеры для трекинга объектов
Использует оптический поток и homography для учета движения камеры дрона
"""

import cv2
import numpy as np


class MotionCompensator:
    """Компенсатор движения камеры"""
    
    def __init__(self, method='homography'):
        """
        Инициализация компенсатора движения
        
        Args:
            method: метод компенсации ('optical_flow' или 'homography')
        """
        self.method = method
        self.prev_frame = None
        self.prev_gray = None
        self.transform_matrix = None
        self.prev_dx = 0.0
        self.prev_dy = 0.0
        self.prev_rotation = 0.0
        self.alpha = 0.85  # коэффициент сглаживания

        
        # Параметры для оптического потока
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Параметры для детектора углов (для оптического потока)
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Параметры для ORB (для homography)
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def compute_transform(self, current_frame):
        """
        Вычисляет трансформацию между текущим и предыдущим кадром
        
        Args:
            current_frame: текущий кадр (BGR)
            
        Returns:
            transform_matrix: матрица трансформации (3x3 для homography) или None
        """
        if self.prev_frame is None:
            self.prev_frame = current_frame.copy()
            self.prev_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            return np.eye(3, dtype=np.float32)
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        if self.method == 'optical_flow':
            transform = self._compute_optical_flow_transform(self.prev_gray, current_gray)
        else:  # homography
            transform = self._compute_homography_transform(self.prev_gray, current_gray)
        
        self.prev_frame = current_frame.copy()
        self.prev_gray = current_gray
        self.transform_matrix = transform
        
        return transform
    
    def _compute_optical_flow_transform(self, prev_gray, current_gray):
        """Вычисляет трансформацию через оптический поток"""
        # Находим углы на предыдущем кадре
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)
        
        if p0 is None or len(p0) < 4:
            return np.eye(3, dtype=np.float32)
        
        # Вычисляем оптический поток
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0, None, **self.lk_params)
        
        # Фильтруем хорошие точки
        good_old = p0[st == 1]
        good_new = p1[st == 1]
        
        if len(good_old) < 4:
            return np.eye(3, dtype=np.float32)
        
        # Вычисляем аффинную трансформацию
        transform, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
        
        if transform is None:
            return np.eye(3, dtype=np.float32)
        
        # Конвертируем в матрицу 3x3
        transform_3x3 = np.eye(3, dtype=np.float32)
        transform_3x3[:2, :] = transform
        
        return transform_3x3
    
    def _compute_homography_transform(self, prev_gray, current_gray):
        """Вычисляет трансформацию через homography"""
        # Находим ключевые точки и дескрипторы
        kp1, des1 = self.orb.detectAndCompute(prev_gray, None)
        kp2, des2 = self.orb.detectAndCompute(current_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return np.eye(3, dtype=np.float32)
        
        # Сопоставляем дескрипторы
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Фильтруем хорошие совпадения (Lowe's ratio test)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            return np.eye(3, dtype=np.float32)
        
        # Извлекаем точки
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Вычисляем homography
        transform, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if transform is None:
            return np.eye(3, dtype=np.float32)
        
        return transform.astype(np.float32)
    
    def transform_bbox(self, bbox, transform):
        """
        Трансформирует bounding box с учетом движения камеры
        
        Args:
            bbox: [x1, y1, x2, y2]
            transform: матрица трансформации 3x3
            
        Returns:
            transformed_bbox: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        
        # Точки углов бокса
        points = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1]
        ], dtype=np.float32).T
        
        # Применяем трансформацию
        transformed_points = transform @ points
        transformed_points = transformed_points[:2, :] / transformed_points[2, :]
        
        # Находим новый bounding box
        new_x1 = float(np.min(transformed_points[0, :]))
        new_y1 = float(np.min(transformed_points[1, :]))
        new_x2 = float(np.max(transformed_points[0, :]))
        new_y2 = float(np.max(transformed_points[1, :]))
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    def get_camera_motion(self):
        """
        Возвращает оценку движения камеры
        
        Returns:
            (dx, dy, rotation): смещение и поворот камеры
        """
        if self.transform_matrix is None:
            return (0.0, 0.0, 0.0)
        
        # Извлекаем смещение из матрицы трансформации
        dx = float(self.transform_matrix[0, 2])
        dy = float(self.transform_matrix[1, 2])
        
        # Вычисляем угол поворота
        rotation = float(np.arctan2(self.transform_matrix[1, 0], self.transform_matrix[0, 0]))
        
        dx = self.alpha * dx + (1 - self.alpha) * self.prev_dx
        dy = self.alpha * dy + (1 - self.alpha) * self.prev_dy
        rotation = self.alpha * rotation + (1 - self.alpha) * self.prev_rotation

        self.prev_dx = dx
        self.prev_dy = dy
        self.prev_rotation = rotation

        return (dx, dy, rotation)

