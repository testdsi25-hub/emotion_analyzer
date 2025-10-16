# Bibliotheken
import librosa
import numpy as np
import pandas as pd
import streamlit as st

def extract_audio_features(file_path: str, sr: int = None) -> dict:
    """
    Extrahiert verschiedene akustische Merkmale aus einer Audio-Datei.
    
    Params:
        file_path: Pfad zur Audio-Datei
        sr: Sample Rate für das Laden der Audio-Datei
        
    Returns:
        Dictionary mit extrahierten Features
    """
    try:
        # Audio laden
        y, sr = librosa.load(file_path, sr=sr)
        
        # Dauer
        duration = librosa.get_duration(y=y, sr=sr)
        
        # RMS Energie (Root Mean Square) - Lautstärke
        rms = librosa.feature.rms(y=y)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)  # Umrechnung in dBFS
        rms_mean = np.mean(rms_db)
        rms_std = np.std(rms_db)

        
        # Tonhöhe (Pitch) mit piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
        # Extrahiere nur die dominanten Pitches
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:  # Nur gültige Pitch-Werte
                pitch_values.append(pitch)
        
        pitch_mean = np.mean(pitch_values) if pitch_values else 0
        pitch_std = np.std(pitch_values) if pitch_values else 0
        
        # Spektrale Eigenschaften - Energie
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_centroid_mean = np.mean(spectral_centroids)
        spectral_centroid_std = np.std(spectral_centroids)
        
        # Spektrale Rolloff (Energie-Verteilung)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        
        # Zero Crossing Rate (Maß für "Rauheit" der Stimme)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        
        # MFCCs (Mel-Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_features = {}
        for i in range(13):
            mfcc_features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            mfcc_features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
        
        # Alle Features zusammenfassen
        features = {
            # Grundlegende Features
            'duration': duration,
            'rms_mean': rms_mean,
            'rms_std': rms_std,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'spectral_centroid_mean': spectral_centroid_mean,
            'spectral_centroid_std': spectral_centroid_std,
            'spectral_rolloff_mean': spectral_rolloff_mean,
            'zcr_mean': zcr_mean,
            
            # Erweiterte Features
            **mfcc_features # ** entpackt ein Dictionary in oben benannte Parameter
        }
        
        return features
        
    except Exception as e:
        st.warning(f"Fehler beim Extrahieren der Features aus {file_path}: {str(e)}")
        return {}

@st.cache_data
def extract_features_for_dataset(df: pd.DataFrame, max_files: int = None) -> pd.DataFrame:
    """
    Extrahiert Features für den gesamten Datensatz.
    
    Params:
        df: DataFrame mit RAVDESS Metadaten
        max_files: Maximale Anzahl zu verarbeitender Dateien (für Tests)
        
    Returns:
        DataFrame mit Features für alle Dateien
    """
    if max_files:
        df = df.head(max_files)
    
    features_list = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in df.iterrows():
        progress = (idx + 1) / len(df)
        progress_bar.progress(progress)
        status_text.text(f"Verarbeite Datei {idx + 1}/{len(df)}: {row['filename']}")
        
        # Extrahiere Features
        features = extract_audio_features(row['file_path'])
        
        if features:  # Nur wenn Features erfolgreich extrahiert wurden
            # Füge Metadaten hinzu
            features.update({
                'filename': row['filename'],
                'emotion': row['emotion'],
                'intensity': row['intensity'],
                'gender': row['gender'],
                'actor_id': row['actor_id'],
                'statement': row['statement']
            })
            features_list.append(features)
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(features_list)

def get_feature_descriptions() -> dict[str, dict[str, str]]:
    """
    Gibt Beschreibungen der verschiedenen Audio-Features zurück.
    """
    return {
        'basic_features': {
            'duration': {
                'name': '📏 Dauer (duration)',
                'description': 'Länge der Audioaufnahme in Sekunden',
                'unit': 'Sekunden',
                'interpretation': 'Unterschiede im Sprechtempo zwischen Emotionen.'
            },
            'rms_mean': {
                'name': '🔊 Lautstärke (RMS)',
                'description': 'Durchschnittliche Lautstärke/Energie der Aufnahme',
                'unit': 'dB',
                'interpretation': 'Durchschnittliche Lautstärke der Aufnahme (in dB), wobei 0 dB der lauteste Wert im Signal ist.'
            },
            'pitch_mean': {
                'name': '🎼 Tonhöhe (Pitch)',
                'description': 'Durchschnittliche Grundfrequenz der Stimme',
                'unit': 'Hz',
                'interpretation': 'Höhere Tonlage oft bei Angst, tiefere bei Trauer.'
            },
            'spectral_centroid_mean': {
                'name': '🔥 Spektrale Energie (spectral_centroid)',
                'description': 'Schwerpunkt des Frequenzspektrums',
                'unit': 'Hz',
                'interpretation': 'Maß für die Helligkeit/Schärfe der Stimme.'
            },
            'zcr_mean': {
                'name': '🌊 Stimmrauheit (ZCR)',
                'description': 'Zero Crossing Rate ist ein Maß für Rauheit des Klangs (viele Nulldurchgänge = hohe Frequenzänderung, wenige Nulldurchgänge = glattes, langsames Signal).',
                'unit': 'Anteil (0-1)',
                'interpretation': "Höhere Werte bei rauen, aggressiven Sprechen oder bei Frikativen/Reibelaute, wie 's' oder 'f', und niedrige Werte bei Vokalen, wie 'a' oder 'o', oder weichen Konsonanten, wie 'm, n, l'."
            },
        },
        'mfcc_features': {
            'mfcc_1_mean': {
                'name': '🎧 MFCC 1',
                'description': 'Erster Mel-Frequency Cepstral Coefficient beschreibt die grobe spektrale Struktur der Stimme.',
                'unit': 'Dimensionloser Wert',
                'interpretation': 'Gibt Aufschluss über die allgemeine Form der Frequenzverteilung (z. B. Helligkeit oder Dunkelheit des Klangs), oft sensitiv für Unterschiede im Vokalklang oder der Emotion.'
            },
            'mfcc_2_mean': {
                'name': '🎧 MFCC 2',
                'description': 'Zweiter MFCC-Koeffizient betont mittlere Frequenzbereiche.',
                'unit': 'Dimensionloser Wert',
                'interpretation': 'Ergänzt MFCC 1 durch feiner aufgelöste Information über mittlere Sprachfrequenzen.'
            },
            'mfcc_3_mean': {
                'name': '🎧 MFCC 3',
                'description': 'Dritter MFCC-Koeffizient beschreibt die Formantübergänge.',
                'unit': 'Dimensionloser Wert',
                'interpretation': 'Wichtiger Koeffizient zur Erkennung von Artikulationsmustern (z. B. Vokalwechsel).'
            },
            'mfcc_4_mean': {
                'name': '🎧 MFCC 4',
                'description': 'Vierter MFCC-Koeffizient zeigt die mittlere Spektralstruktur.',
                'unit': 'Dimensionloser Wert',
                'interpretation': 'Beschreibt Übergänge in der mittleren Frequenzstruktur, hilfreich bei Sprachcharakteristik.'
            },
            'mfcc_5_mean': {
                'name': '🎧 MFCC 5',
                'description': 'Fünfter MFCC-Koeffizient stellt feinere spektrale Merkmale dar.',
                'unit': 'Dimensionloser Wert',
                'interpretation': 'Stärker sensitiv gegenüber Variationen in Betonung und Artikulation.'
            },
            'mfcc_6_mean': {
                'name': '🎧 MFCC 6',
                'description': 'Sechster MFCC-Koeffizient beschreibt die Sprachmodulation.',
                'unit': 'Dimensionloser Wert',
                'interpretation': 'Reagiert auf subtile Unterschiede in Sprachmelodie und Modulation.'
            },
            'mfcc_7_mean': {
                'name': '🎧 MFCC 7',
                'description': 'Siebter MFCC-Koeffizient analysiert hohe Frequenzen und Artikulation.',
                'unit': 'Dimensionloser Wert',
                'interpretation': 'Bedeutsam für die Wahrnehmung von Zischlauten und scharfen Artikulationen.'
            },
            'mfcc_8_mean': {
                'name': '🎧 MFCC 8',
                'description': 'Achter MFCC-Koeffizient macht Artikulationsdetails sichtbar.',
                'unit': 'Dimensionloser Wert',
                'interpretation': 'Trägt zur Unterscheidung von feinen Klangmerkmalen bei.'
            },
            'mfcc_9_mean': {
                'name': '🎧 MFCC 9',
                'description': 'Neunter MFCC-Koeffizient zeigt feine Unterschiede in Timbre.',
                'unit': 'Dimensionloser Wert',
                'interpretation': 'Betont subtile klangliche Unterschiede zwischen Emotionen.'
            },
            'mfcc_10_mean': {
                'name': '🎧 MFCC 10',
                'description': 'Zehnter MFCC-Koeffizient konzentriert sich auf hohe spektrale Details.',
                'unit': 'Dimensionloser Wert',
                'interpretation': 'Sensitiv für Höhen und Frikative wie „s“, „f“ – wichtig für Sprachklarheit.'
            },
            'mfcc_11_mean': {
                'name': '🎧 MFCC 11',
                'description': 'Elfter MFCC-Koeffizient konzentriert sich auf sehr hohe Frequenzdetails.',
                'unit': 'Dimensionloser Wert',
                'interpretation': 'Unterstützt die Erkennung feiner Stimmmerkmale und Emotionen.'
            },
            'mfcc_12_mean': {
                'name': '🎧 MFCC 12',
                'description': 'Zwölfter MFCC-Koeffizient zeigt Nuancen in der Klangfarbe.',
                'unit': 'Dimensionloser Wert',
                'interpretation': 'Betont feine Variationen im Timbre und Sprechweise.'
            },
            'mfcc_13_mean': {
                'name': '🎧 MFCC 13',
                'description': 'Dreizehnter MFCC-Koeffizient stellt hochfrequente Formantenreste dar.',
                'unit': 'Dimensionloser Wert',
                'interpretation': 'Ergänzt das Gesamtbild mit feinsten spektralen Restinformationen.'
            }
}

    }

def calculate_feature_statistics(df: pd.DataFrame, feature_col: str, group_by: str = 'emotion') -> pd.DataFrame:
    """
    Berechnet Statistiken für ein Feature gruppiert nach einer Kategorie.
    
    Params:
        df: DataFrame mit Features
        feature_col: Name der Feature-Spalte
        group_by: Spalte nach der gruppiert werden soll
        
    Returns:
        DataFrame mit Statistiken (mean, std, min, max, median)
    """
    df[feature_col] = pd.to_numeric(df[feature_col], errors='coerce') 

    stats = df.groupby(group_by)[feature_col].agg([
        'mean', 'std', 'min', 'max', 'median', 'count'
    ]).round(3)
    
    return stats.reset_index()

def get_basic_features_list() -> list[str]:
    """Gibt eine Liste der wichtigsten Features für die Analyse zurück."""

    return [
        'duration', 'rms_mean', 'pitch_mean', 
        'spectral_centroid_mean', 'zcr_mean'
    ]

def get_mfcc_features_list() -> list[str]:
    """Gibt eine Liste aller MFCC Features zurück."""

    return [f'mfcc_{i+1}_mean' for i in range(13)]