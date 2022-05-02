from flask import Flask,render_template,Response
import cv2
import tensorflow as tf
import numpy as np
# from Member import *
from Member import *
from User import *
from datetime import timedelta
###############
from facemaskdetection_2 import stream


app = Flask(__name__)
camera = cv2.VideoCapture(0)


def gen_frame():
    """Video streaming generator function."""
    while True:
        frame = stream()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') # concate frame one by one and show result

@app.route("/")
def Index():
    # return "This is home"
    return render_template('login.html',headername="Login เข้าใช้งานระบบ")

@app.route("/test")
def index1():
    return render_template('test.html',)

@app.route("/video")
def video():
    return Response(gen_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

app.secret_key = "sattawat"
app.permanent_session_lifetime = timedelta(days=1)
app.register_blueprint(member)
app.register_blueprint(user)


if __name__ == '__main__':
    app.run(debug = True)


