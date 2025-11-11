#!/bin/bash

# параметры подключения
set -e
echo "Введите IP БВС"
read RASPIP
echo "Вы ввели IP = $RASPIP, подключаемся"

SERVER="ip@{$RASPIP}"
PASSWORD="raspberry"
PORT="22"

# отправка команд на сервер
sshpass -p "$PASSWORD" scp "clover.launch" "${SERVER}:~/catkin_ws/src/clover/clover/launch/clover.launch"

sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $PORT $SERVER "sudo systemctl restart clover"

sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $PORT $SERVER "echo "export ROS_MASTER_URI=http://${RASPIP}:11311" >>~/.bashrc"
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $PORT $SERVER "echo "export ROS_IP=${RASPIP}" >>~/.bashrc"
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $PORT $SERVER "source ~/.bashrc"


MAINIP=`hostname -I | cut -d " " -f1`
# на стороне сревера
echo "export ROS_MASTER_URI=http://${RASPIP}:11311" >>~/.bashrc
echo "export ROS_IP=<${MAINIP}" >>~/.bashrc
source ~/.bashrc

rostopic list | grep main_camera
set +e