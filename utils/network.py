"""
Утилиты для работы с сетью
"""

import socket


def get_local_ip():
    """
    Получает локальный IP-адрес компьютера в локальной сети
    
    Returns:
        str: IP-адрес или "не определен" в случае ошибки
    """
    try:
        # Создаем временное соединение для определения IP
        # Подключаемся к внешнему адресу (не устанавливая реальное соединение)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            # Альтернативный способ через hostname
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            return ip
        except Exception:
            return "не определен"


def print_server_info(local_ip, port=5000):
    """
    Выводит информацию о сервере для доступа
    
    Args:
        local_ip: локальный IP-адрес
        port: порт сервера
    """
    print("\n" + "="*50)
    print("Сервер запущен!")
    print("="*50)
    print("Доступ к серверу:")
    print(f"  Локально:    http://localhost:{port}")
    print(f"  В сети:      http://{local_ip}:{port}")
    print("="*50)
    print("Для доступа с другого устройства в той же сети:")
    print(f"  Откройте в браузере: http://{local_ip}:{port}")
    print("="*50)

