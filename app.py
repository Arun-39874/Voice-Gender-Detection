import streamlit as st
from sklearn.model_selection import train_test_split
import joblib
import librosa
import numpy as np

model = joblib.load("voice_model.pkl")

st.title("Voice Gender Detection")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

def extract_features(audio_data, _):
    try:
        audio_data = audio_data.astype(np.float32)
        features = np.mean(librosa.feature.mfcc(y=audio_data, sr=_).T, axis=0)
        return features
    except Exception as e:
        print("Error extracting features:", str(e))
        return None

def predict_gender(model, audio_data):
    features = extract_features(audio_data, 44100)

    if features is not None:
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        return prediction[0]
    else:
        return None

if uploaded_file is not None:
    audio_data, _ = librosa.load(uploaded_file, sr=None)

    predicted_gender = predict_gender(model, audio_data)
    if predicted_gender==1:
        predicted_gender = "Male"
    else:
        predicted_gender = "Female"
    if predicted_gender is not None:
        st.success(f"Predicted Gender: {predicted_gender}")
    else:
        st.error("Error processing the audio file.")