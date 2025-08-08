from flask import Flask, request, jsonify
import cv2
import numpy as np
from emotion_recognition_cnn import predict_emotion
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    emotion = predict_emotion(img)
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run()
