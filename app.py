from flask import Flask, render_template, request, jsonify
import joblib
import librosa
import numpy as np
import sounddevice as sd
import os
import io
import wave
import shutil
import soundfile as sf

app = Flask(__name__)

# Load the trained model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "voice_model.pkl")
model = joblib.load(model_path)

# Directory to store live recordings
live_recordings_dir = os.path.join(script_dir, "live_recordings")
os.makedirs(live_recordings_dir, exist_ok=True)

def extract_features(audio_data, _):
    try:
        audio_data = audio_data.astype(np.float32)
        features = np.mean(librosa.feature.mfcc(y=audio_data, sr=_).T, axis=0)
    except Exception as e:
        print("Error extracting features:", str(e))
        return None
    return features

def predict_gender(audio_data):
    features = extract_features(audio_data, 44100)
    
    if features is not None:
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        gender = "Male" if prediction[0] == 1 else "Female"
        return gender
    else:
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'audio_data' in request.files:
        # Process audio file
        file = request.files['audio_data']
        audio_data, _ = librosa.load(file, sr=None)
    else:
        # Record live audio for 5 seconds
        print("Recording 5 seconds of audio. Speak now...")
        audio_data = sd.rec(int(5 * 44100), samplerate=44100, channels=1, dtype=np.int16)
        sd.wait()
        audio_data = audio_data.flatten()
         # Save the recorded audio as a .wav file
        save_path = "recorded_voice.wav"
        wf = wave.open(save_path, "wb")
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(audio_data.tobytes())
        wf.close()

    predicted_gender = predict_gender(audio_data)
    return jsonify({'gender': predicted_gender})

# Add this route to delete previous live recordings
@app.route('/delete_previous_recordings', methods=['GET'])
def delete_previous_recordings():
    try:
        # Delete the live recordings directory and recreate it
        shutil.rmtree(live_recordings_dir)
        os.makedirs(live_recordings_dir, exist_ok=True)
        return jsonify({'message': 'Previous live recordings deleted successfully'})
    except Exception as e:
        return jsonify({'error': f'Error deleting previous live recordings: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
