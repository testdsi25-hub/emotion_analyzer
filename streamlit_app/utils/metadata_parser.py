# Bibliotheken
import streamlit as st
import pandas as pd
import numpy as np
import librosa
import os
import glob
from pathlib import Path

# Mapping-Dictionaries für konsistente Darstellung
EMOTION_COLOR_MAP = {
    'neutral': '#95a5a6',
    'calm': '#3498db',
    'happy': '#f1c40f',
    'sad': '#2c3e50',
    'angry': '#e74c3c',
    'fearful': '#9b59b6',
    'disgust': '#27ae60',
    'surprised': '#e67e22'
}

GENDER_DISPLAY_MAP = {
    'male': 'männlich',
    'female': 'weiblich'
}

INTENSITY_DISPLAY_MAP = {
    'normal': 'normal',
    'strong': 'stark'
}

def parse_ravdess_filename(filename) -> dict:
    """
    Parst RAVDESS Dateinamen um Metadaten zu extrahieren.
    Format der Audiodateinamen: Modality-VocalChannel-Emotion-Emotional Intensity-Statement-Repetition-Actor.wav
    """
    try:
        # Entfernt Dateiendung und teile nach '-'
        parts = Path(filename).stem.split('-') # entfernt die Dateiendung und unterteilt den Dateinnamen in die einzelnen Teile
        
        if len(parts) != 7: # -> wäre ein ungültiges Dateiformat
            return None
            
        emotion_mapping = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'angry',
            '06': 'fearful',
            '07': 'disgust',
            '08': 'surprised'
        }
        
        intensity_mapping = {
            '01': 'normal',
            '02': 'strong'
        }
        
        statement_mapping = {
            '01': 'Kids are talking by the door',
            '02': 'Dogs are sitting by the door'
        }
        
        # Geschlecht basierend auf Actor ID (ungerade = männlich, gerade = weiblich)
        actor_id = int(parts[6])
        gender = 'male' if actor_id % 2 == 1 else 'female'
        
        return {
            'filename': filename,
            'modality': parts[0],
            'vocal_channel': parts[1],
            'emotion': emotion_mapping.get(parts[2], 'Unbekannt'),
            'intensity': intensity_mapping.get(parts[3], 'Unbekannt'),
            'statement': statement_mapping.get(parts[4], 'Unbekannt'),
            'repetition': parts[5],
            'actor_id': actor_id,
            'gender': gender
        }
    except Exception as e:
        st.error(f"Fehler beim Parsen von {filename}: {str(e)}")
        return None

@st.cache_data
def load_ravdess_metadata(data_path) -> pd.DataFrame:
    """Lädt und parst alle RAVDESS Metadaten aus dem angegebenen Pfad."""
    
    # Verwendet glob.glob, um alle .wav-Dateien rekursiv zu finden
    wav_files = glob.glob(os.path.join(data_path, "**", "*.wav"), recursive=True)
    
    # Erstellt eine Liste von Metadaten-Wörterbüchern
    metadata_list = []
    for file_path in wav_files:
        filename = os.path.basename(file_path)
        metadata = parse_ravdess_filename(filename)
        if metadata:
            metadata['file_path'] = file_path
            metadata_list.append(metadata)
            
    # Konvertiert die Liste in einen DataFrame und gib ihn zurück
    return pd.DataFrame(metadata_list)

@st.cache_data
def calculate_length_average(df) -> float:
    """Berechnet die durchschnittliche Dauer der Audiodateien in Sekunden."""
    length = []
    for path in df['file_path']:
        try:
            y, sr = librosa.load(path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            length.append(duration)
        except Exception as e:
            st.warning(f"Fehler beim Laden von {path}: {e}")
    
    if length:
        return np.mean(length)
    return 0.0

def format_metadata_for_display(metadata_row) -> dict[str]:
    """
    Formatiert Metadaten für die Anzeige in der UI.
    Konvertiert interne Werte in benutzerfreundliche deutsche Bezeichnungen.
    """
    return {
        'emotion': metadata_row['emotion'].title(),
        'speaker_id': f"Actor_{metadata_row['actor_id']:02d}",
        'gender': GENDER_DISPLAY_MAP.get(metadata_row['gender'], metadata_row['gender']),
        'intensity': INTENSITY_DISPLAY_MAP.get(metadata_row['intensity'], metadata_row['intensity']),
        'sentence_type': metadata_row['statement'],
        'filename': metadata_row['filename'],
        'filepath': metadata_row['file_path']
    }

def load_audio_data(file_path, sr=48000) -> dict:
    """
    Lädt Audiodaten und berechnet verschiedene Darstellungen.
    
    Params:
        file_path: Pfad zur Audiodatei
        sr: Sample Rate
        
    Returns:
        Dictionary mit Audiodaten und berechneten Features
    """
    try:
        # Audio laden
        y, sr = librosa.load(file_path, sr=sr)
        
        # Zeit-Array für x-Achse
        time = librosa.frames_to_time(range(len(y)), sr=sr)
        
        # Mel-Spektrogramm berechnen
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # MFCC berechnen
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Spektrogramm für detailliertere Analyse
        stft = librosa.stft(y)
        stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        
        # Pitch tracking
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
        
        return {
            'y': y,
            'sr': sr,
            'time': time,
            'mel_spec_db': mel_spec_db,
            'mfccs': mfccs,
            'stft_db': stft_db,
            'pitches': pitches,
            'magnitudes': magnitudes
        }
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Audiodatei: {str(e)}")
        return None


def display_audio_metadata(metadata) -> None:
    """Zeigt Audio-Metadaten in einer übersichtlichen Form an."""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎭 Emotion", metadata['emotion'].title())
        st.metric("👥 Geschlecht", GENDER_DISPLAY_MAP.get(metadata['gender'], metadata['gender']))
    
    with col2:
        st.metric("🎚️ Intensität", INTENSITY_DISPLAY_MAP.get(metadata['intensity'], metadata['intensity']))
        st.metric("🎤 Sprecher", f"Actor {metadata['actor_id']:02d}")
    
    with col3:
        st.metric("💬 Aussage", metadata['statement'])
        st.metric("📁 Datei", metadata['filename'])

    return None


def display_extracted_features(features) -> None:
    """Zeigt extrahierte Audio-Features an (fokussiert auf Einzeldatei-Analyse)."""
    
    st.markdown("🧮 Features dieser Datei extrahiert.")
    
    # Grundlegende Features
    st.write("**📊 Akustische Eigenschaften dieser Aufnahme:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'duration' in features:
            st.metric("📏 Dauer", f"{features['duration']:.2f} s")
        if 'rms_mean' in features:
            st.metric("🔊 Lautstärke", f"{features['rms_mean']:.2f} dB")
    
    with col2:
        if 'pitch_mean' in features:
            st.metric("🎼 Tonhöhe", f"{features['pitch_mean']:.1f} Hz")
        if 'spectral_centroid_mean' in features:
            st.metric("🔥 Spektrale Energie", f"{features['spectral_centroid_mean']:.1f} Hz")
    
    with col3:
        if 'zcr_mean' in features:
            st.metric("🌊 Stimmrauheit", f"{features['zcr_mean']:.4f}")
    
    # Kurze MFCC Übersicht (weniger Details als in Feature Analysis)
    with st.expander("🎧 MFCC Koeffizienten dieser Datei"):
        mfcc_cols = st.columns(4)
        mfcc_features = [f'mfcc_{i+1}_mean' for i in range(6)]  # Nur erste 6 anzeigen
        
        for idx, mfcc_feature in enumerate(mfcc_features):
            if mfcc_feature in features:
                with mfcc_cols[idx % 4]:
                    st.metric(f"MFCC {idx+1}", f"{features[mfcc_feature]:.3f}")

        st.info("💡 Für detaillierte MFCC-Analysen und Vergleiche zwischen Emotionen besuchen Sie die **Feature-Analysis** Seite.")
    
    return None