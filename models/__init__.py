"""
Модули для детекции объектов
"""

from .detector import ObjectDetector
from .roi_logic import ROILogicManager, LostObject, ObjectStatus

__all__ = ['ObjectDetector', 'ROILogicManager', 'LostObject', 'ObjectStatus']

