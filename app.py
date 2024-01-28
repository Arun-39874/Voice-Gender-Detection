import streamlit as st
import numpy as np
import soundfile as sf
from sklearn.model_selection import train_test_split
import joblib
import librosa

# Load the trained model
model = joblib.load("voice_model.pkl")

# Streamlit app
st.title("Voice Gender Detection")

# Option to choose between live recording and file upload
input_option = st.radio("Select Input Option:", ["Live", "File"])

# Function to extract audio features
def extract_features(audio_data, _):
    try:
        audio_data = audio_data.astype(np.float32)
        features = np.mean(librosa.feature.mfcc(y=audio_data, sr=_).T, axis=0)
        return features
    except Exception as e:
        print("Error extracting features:", str(e))
        return None

# Function to predict gender
def predict_gender(model, audio_data):
    features = extract_features(audio_data, 44100)

    if features is not None:
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        return prediction[0]
    else:
        return None

# Live Audio Recording Section
if input_option == "Live":
    start_recording = st.button("Start Speaking")

    if start_recording:
        st.info("Recording 5 seconds of audio. Speak now...")

        # Set up audio recording parameters
        duration = 5  # seconds
        fs = 44100  # sample rate
        audio_data = st.audio_recorder(duration=duration, sample_rate=fs, channels=1)

        st.audio(audio_data, format="audio/wav")  # Display the recorded audio (optional)

        # Perform prediction
        predicted_gender = predict_gender(model, audio_data)
        if predicted_gender == 1:
            predicted_gender = "Male"
        else:
            predicted_gender = "Female"

        # Display the result
        if predicted_gender is not None:
            st.success(f"Predicted Gender: {predicted_gender}")
        else:
            st.error("Error processing live audio.")

# File Upload Section
elif input_option == "File":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

    if uploaded_file is not None:
        # Use soundfile to load the audio content directly
        audio_data, fs = sf.read(uploaded_file)

        # Perform prediction
        predicted_gender = predict_gender(model, audio_data)
        if predicted_gender == 1:
            predicted_gender = "Male"
        else:
            predicted_gender = "Female"

        # Display the result
        if predicted_gender is not None:
            st.success(f"Predicted Gender: {predicted_gender}")
        else:
            st.error("Error processing the audio file.")
