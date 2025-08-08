from flask import Flask, request, jsonify
import cv2
import numpy as np
from emotion_recognition_cnn import predict_emotion

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    emotion = predict_emotion(img)
    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
