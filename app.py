import streamlit as st
from sklearn.model_selection import train_test_split
import joblib
import librosa
import numpy as np

# Load the trained model
model = joblib.load("voice_model.pkl")

# Streamlit app
st.title("Voice Gender Detection")

# File upload
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

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

# Perform prediction when the button is pressed
if uploaded_file is not None:
    # Use librosa to load the audio content directly
    audio_data, _ = librosa.load(uploaded_file, sr=None)

    # Perform prediction
    predicted_gender = predict_gender(model, audio_data)
    if predicted_gender==1:
        predicted_gender = "Male"
    else:
        predicted_gender = "Female"
    # Display the result
    if predicted_gender is not None:
        st.success(f"Predicted Gender: {predicted_gender}")
    else:
        st.error("Error processing the audio file.")