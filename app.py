import streamlit as st
from sklearn.model_selection import train_test_split
import joblib
import librosa
import numpy as np
from pydub import AudioSegment

# Load the trained model
model = joblib.load("voice_model.pkl")

# Streamlit app
st.title("Voice Gender Detection")

# File upload
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

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
    # Check the file extension
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "mp3":
        # Convert MP3 to WAV using pydub
        audio = AudioSegment.from_mp3(uploaded_file)
        audio_data = np.array(audio.get_array_of_samples())
        sr = audio.frame_rate
    elif file_extension == "wav":
        # Use librosa to load the audio content directly
        audio_data, sr = librosa.load(uploaded_file, sr=None)
    else:
        st.error("Unsupported file format. Please upload a WAV or MP3 file.")
        st.stop()

    # Perform prediction
    predicted_gender = predict_gender(model, audio_data)
    predicted_gender = "Male" if predicted_gender == 1 else "Female"

    # Display the result
    st.success(f"Predicted Gender: {predicted_gender}")
