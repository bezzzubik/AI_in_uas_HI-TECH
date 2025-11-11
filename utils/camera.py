"""
Утилиты для работы с камерами
"""

import cv2


def find_available_cameras(max_cameras=10):
    """
    Находит все доступные веб-камеры в системе
    
    Args:
        max_cameras: максимальное количество камер для проверки
        
    Returns:
        list: список индексов доступных камер
    """
    available_cameras = []
    
    print("Поиск доступных веб-камер...")
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Пытаемся прочитать кадр, чтобы убедиться, что камера действительно работает
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"  ✓ Камера #{i} найдена и доступна")
            cap.release()
        else:
            cap.release()
    
    if available_cameras:
        print(f"Найдено доступных камер: {len(available_cameras)}")
        return available_cameras
    else:
        print("⚠️  Доступные веб-камеры не найдены")
        return []


def get_camera_source(requested_source):
    """
    Получает доступный источник камеры
    
    Args:
        requested_source: запрошенный источник (индекс камеры, путь к файлу, URL и т.д.)
        
    Returns:
        int или str: доступный источник камеры
    """
    # Если это не число (файл, URL и т.д.), возвращаем как есть
    if not isinstance(requested_source, int):
        return requested_source
    
    # Если это индекс камеры, проверяем доступность
    available_cameras = find_available_cameras()
    
    if not available_cameras:
        print("⚠️  Доступные веб-камеры не найдены!")
        print("   Попытка использовать запрошенную камеру #{}...".format(requested_source))
        return requested_source
    
    # Проверяем, доступна ли запрошенная камера
    if requested_source in available_cameras:
        print(f"✅ Используется камера #{requested_source}")
        return requested_source
    else:
        # Используем первую доступную камеру
        first_available = available_cameras[0]
        print(f"⚠️  Камера #{requested_source} недоступна")
        print(f"✅ Автоматически выбрана камера #{first_available}")
        return first_available

