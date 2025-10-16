# Bibliotheken
import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf

# Laden von Modell & Encoder
MODEL_PATH = "models/cnn_lstm_emotion_model_5dim.keras"
ENCODER_PATH = "models/label_encoder.pkl"

@st.cache_resource
def load_model_and_encoder():
    """
    Lädt das Keras-Modell und den Label-Encoder einmal und speichert diese im Cache
    """
    model = tf.keras.models.load_model(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    return model, le


def extract_features_from_signal(signal, sr, max_len=216):
    """
    Extrahiert akustische Merkmale
    """
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    mel = librosa.feature.melspectrogram(y=signal, sr=sr)
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_chroma=40)

    def pad(x):
        if x.shape[1] < max_len:
            return np.pad(x, ((0,0),(0,max_len-x.shape[1])), mode="constant")
        else:
            return x[:, :max_len]

    mfcc = pad(mfcc)
    delta = pad(delta)
    delta2 = pad(delta2)
    mel = pad(mel)
    chroma = pad(chroma)

    return np.stack([mfcc, delta, delta2, mel[:40], chroma[:40]], axis=-1)

def extract_features_from_file(file_path, sr=22050):
    """
    Lädt eine Audiodatei und extrahiert Merkmale über extract_features_from_signal
    """
    signal, sr = librosa.load(file_path, sr=sr)
    return extract_features_from_signal(signal, sr)

def predict_emotion(file_path):
    """
    Führt die komplette Pipeline durch: Laden, Extraktion, Vorhersage, Rückübersetzung.
    """
    model, le = load_model_and_encoder()
    features = extract_features_from_file(file_path)
    features = np.expand_dims(features, axis=0)

    probs = model.predict(features)[0]
    emotion = le.inverse_transform([np.argmax(probs)])[0]
    return emotion, probs, le.classes_
