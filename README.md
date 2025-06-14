# VoiceRecognization
# 🎙️ Voice Gender Recognition using Machine Learning

This project uses **Machine Learning** to detect the **gender of a speaker's voice** by analyzing audio features. It includes live voice recording and real-time prediction using a trained model.

---

## 📁 Project Structure

```
voice-gender-recognition/
├── gender_voice_dataset.csv        # Dataset for training
├── gender_voice_recognition.py     # Trains the model
├── record_and_predict.py           # Records voice & predicts gender
├── gender_model.pkl                # Trained model
├── scaler.pkl                      # Feature scaler used during training
└── README.md                       # Project documentation
```

---

## 📊 Model Performance

- **Model Used:** Support Vector Machine (SVM)
- **Feature Extraction:** MFCCs (Mel-Frequency Cepstral Coefficients)
- **Confusion Matrix:**

```
[[293   4]
 [ 11 326]]
```

✅ High accuracy on test data  
✅ Model saved as `gender_model.pkl`

---

## 📦 Requirements

Install all necessary dependencies:

```bash
pip install sounddevice scipy python_speech_features numpy scikit-learn pandas
```

Make sure your system has a working microphone for real-time prediction.

---

## ▶️ How to Run

### Step 1: Train the Model

```bash
python gender_voice_recognition.py
```

This will:
- Load the CSV dataset
- Extract MFCC features
- Train an SVM classifier
- Save the trained model as `gender_model.pkl`
- Save the scaler as `scaler.pkl`

### Step 2: Record and Predict

```bash
python record_and_predict.py
```

This will:
- Record 3 seconds of your voice
- Extract MFCC features
- Use the saved model to predict if the voice is **Male** or **Female**

---

## 📌 Dataset Used

Dataset: [Voice Gender Dataset](https://www.kaggle.com/datasets/primaryobjects/voicegender)

It contains extracted voice features such as:
- mean frequency
- spectral centroid
- MFCCs
- jitter
- shimmer
- and more...

---

## 📈 Possible Improvements

- Add a GUI using Tkinter or PyQt
- Deploy on web with Flask or Streamlit
- Support for multilingual voice input
- Real-time prediction on streaming audio

---

## 💻 Tech Stack

- Python
- Scikit-learn
- NumPy, Pandas
- SoundDevice
- SciPy
- Python Speech Features



## ⭐ Star the repo if you find it useful!
