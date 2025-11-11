"""
Утилиты для системы детекции объектов
"""

from .network import get_local_ip, print_server_info
from .camera import find_available_cameras, get_camera_source

__all__ = ['get_local_ip', 'print_server_info', 'find_available_cameras', 'get_camera_source']

