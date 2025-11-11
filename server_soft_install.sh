#!/bin/bash

# предонозначени для debian-based uix систем

# обновляем информацию о репозиториях
sudo apt update

# Устанавливаем sshpass для удаленного выполнения команд по ssh
sudo apt-get install sshpass

# Добавляем репозиторий ROS
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl -y
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# Устанавливаем ROS Noetic
sudo apt update
sudo apt install ros-noetic-desktop-full -y

# Настраиваем окружение
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Установка инструментов работы с ROS

sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential -y

# Инициализация rosdep
sudo rosdep init
rosdep update

# Установка пакетов компьютерного зрения
sudo apt install ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-vision-opencv -y

# Python‑библиотеки для работы с видео

sudo apt install python3-pip -y
pip3 install flask opencv-python opencv-contrib-python numpy


# Проверка списка пакетов

echo "ROS nodes list"

roscore
echo "________________"
rostopiclist
