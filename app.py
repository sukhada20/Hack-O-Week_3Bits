from flask import Flask, request, jsonify
import cv2
import numpy as np
# Load your model here (e.g., with Keras, PyTorch)
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    f = request.files['file']
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    # Preprocess img and run through model
    prediction = your_model.predict(img)  # Replace with actual code
    return jsonify({'emotion': prediction})

if __name__ == '__main__':
    app.run(debug=True)
