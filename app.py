import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load pre-trained emotion detection model (update path as needed)
MODEL_PATH = 'model/emotion_model.h5'
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

@st.cache_resource
def load_emotion_model():
    model = load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(FACE_CASCADE_PATH)

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_face(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return faces

def predict_emotion(face_img, model):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (48, 48))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    preds = model.predict(face_img)[0]
    emotion_idx = np.argmax(preds)
    return EMOTIONS[emotion_idx], preds

def main():
    st.set_page_config(page_title="Real-Time Emotion Detection", page_icon=":smiley:", layout="centered")
    st.title("😊 Real-Time Face Expression Recognition")
    st.markdown("""
        <style>
            .emotion-bar {margin: 0.5em 0;}
            .emotion-label {font-weight: 600;}
            .emotion-value {float: right;}
            .face-box {border: 2px solid #fff; border-radius: 8px;}
        </style>
        """, unsafe_allow_html=True)

    st.sidebar.title("App Settings")
    st.sidebar.info("Allow camera access to test real-time emotion detection.")

    run = st.sidebar.toggle("Run Camera", value=False)
    model = load_emotion_model()
    face_cascade = load_face_detector()

    FRAME_WINDOW = st.image([])

    cap = None
    if run:
        cap = cv2.VideoCapture(0)

    while run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to access camera.")
            break

        faces = detect_face(frame, face_cascade)
        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            emotion, preds = predict_emotion(face_img, model)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

            # Show emotion probabilities (UI)
            st.markdown("#### Emotion Probabilities")
            for idx, (emo, prob) in enumerate(zip(EMOTIONS, preds)):
                st.progress(float(prob), text=f"{emo}: {prob:.2f}")

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    if cap:
        cap.release()

    st.markdown("---")
    st.info("Developed with :heart: using Streamlit & Deep Learning")

if __name__ == "__main__":
    main()
