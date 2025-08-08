import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

MODEL_PATH = 'model.h5'
FACE_CASCADE_PATH = 'HaarcascadeclassifierCascadeClassifier.xml'

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

    # Create persistent placeholders for progress bars
    prob_container = st.container()
    prob_bars = [prob_container.progress(0.0, text=f"{emo}: 0.00") for emo in EMOTIONS]

    cap = None
    if run:
        cap = cv2.VideoCapture(0)

    while run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Unable to access camera.")
            break

        faces = detect_face(frame, face_cascade)
        face_results = []

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            emotion, preds = predict_emotion(face_img, model)
            face_results.append((emotion, preds, (x, y, w, h)))
            # Draw face box and label on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Dynamically update emotion probabilities for the first detected face
        if face_results:
            emotion, preds, box = face_results[0]
            for i, (emo, prob) in enumerate(zip(EMOTIONS, preds)):
                prob_bars[i].progress(float(prob), text=f"{emo}: {prob:.2f}")
        else:
            # If no faces, reset bars to 0
            for i, emo in enumerate(EMOTIONS):
                prob_bars[i].progress(0.0, text=f"{emo}: 0.00")

    if cap:
        cap.release()

    st.markdown("---")
    st.info("Developed with :heart: using Streamlit & Deep Learning")

if __name__ == "__main__":
    main()
