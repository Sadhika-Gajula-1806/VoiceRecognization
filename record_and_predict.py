import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np
import pickle
import time
from python_speech_features import mfcc

# Record voice
duration = 3  # seconds
fs = 44100  # Sample rate
print("🎙️ Recording your voice for 3 seconds...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()

# Save recording
wav.write("your_voice.wav", fs, recording)
print("✅ Voice recorded and saved as your_voice.wav")

# Load model and scaler
print("📦 Loading model and scaler...")
with open("gender_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Extract features
print("🔍 Extracting features...")
(rate, signal) = wav.read("your_voice.wav")

# Take only 1D array if stereo
if signal.ndim > 1:
    signal = signal[:, 0]

# Compute MFCCs
mfcc_features = mfcc(signal, samplerate=rate, numcep=20, nfft=1103)
mfcc_mean = np.mean(mfcc_features, axis=0).reshape(1, -1)  # shape: (1, 20)

# Scale features
scaled_features = scaler.transform(mfcc_mean)

# Predict
prediction = model.predict(scaled_features)[0]
label = "Male" if prediction == 1 else "Female"
print(f"🧠 Predicted Gender: {label}")
