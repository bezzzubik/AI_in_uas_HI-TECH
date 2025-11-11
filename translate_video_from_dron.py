from flask import Flask, Response, render_template_string
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

app = Flask(__name__)
bridge = CvBridge()
frame = None

def image_callback(msg):
    global frame
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

# Инициализация ROS-ноды
rospy.init_node('video_server', anonymous=True)
rospy.Subscriber('/main_camera/image_raw', Image, image_callback)

def generate():
    global frame
    while True:
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        else:
            # Если кадр не готов, отправляем пустой кадр
            print("AAAAAAAAAAAAAAAAAAAAAA")
            blank_image = cv2.imencode('.jpg', cv2.imread('blank.jpg'))[1]
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + blank_image.tobytes() + b'\r\n')

# Главная страница
@app.route('/')
def index():
    html = """
    <html>
      <head>
        <title>Видео с дрона</title>
      </head>
      <body>
        <h1>Добро пожаловать!</h1>
        <p><a href="/video_feed">Перейти к трансляции</a></p>
      </body>
    </html>
    """
    return render_template_string(html)

# Страница с видеопотоком
@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)