#!/bin/bash

# параметры подключения

echo "Введите IP БВС"
read RASPIP
echo "Вы ввели IP = $RASPIP, подключаемся"

SERVER="ip@{$RASPIP}"
PASSWORD="raspberry"
PORT="22"

# отправка команд на сервер
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $PORT $SERVER "roscore"
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $PORT $SERVER "roslaunch clover clover.launch"

sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $PORT $SERVER "echo "export ROS_MASTER_URI=http://${RASPIP}:11311" >>~/.bashrc"
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $PORT $SERVER "echo "export ROS_IP=${RASPIP}" >>~/.bashrc"
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $PORT $SERVER "source ~/.bashrc"


# на стороне сревера
echo "export ROS_MASTER_URI=http://${RASPIP}:11311" >>~/.bashrc
echo "export ROS_IP=<${RASPIP}" >>~/.bashrc
source ~/.bashrc

rostopic list | grep main_camera