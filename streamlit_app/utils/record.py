# Bibliotheken
import wave
import numpy as np
import io
from scipy import signal
import streamlit as st

#### Hilfsfunktionen ####

def normalize_audio(audio_bytes, sampwidth) -> float:
    """Wandelt Rohdaten aus WAV in normalisiertes float32 [-1.0, 1.0]"""

    if sampwidth == 1:
        audio = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128) / 128.0
    elif sampwidth == 2:
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio /= 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(audio_bytes, dtype=np.int32).astype(np.float32)
        audio /= 2**31
    else:
        raise ValueError(f"Nicht unterstützte Sample-Breite: {sampwidth * 8} Bit")
    return audio


def resample_audio(audio_data, orig_rate, target_rate) -> np.ndarray:
    """Resampled Audio mit polyphasen-basierter Methode"""
    return signal.resample_poly(audio_data, up=target_rate, down=orig_rate)


def convert_channels(audio_data, from_channels, to_channels) -> float:
    """Konvertiert Kanalanzahl"""

    if from_channels == to_channels:
        return audio_data

    if from_channels > 1:
        audio_data = audio_data.reshape(-1, from_channels)

    if to_channels == 1:
        # Mono: Mittelwert über alle Kanäle
        return np.mean(audio_data, axis=1)
    else:
        raise ValueError("Nur Mono-Zielkanalanzahl (1) wird unterstützt.")


def float32_to_int16(audio) -> np.ndarray:
    """Konvertiert normalisiertes float32 Audio [-1, 1] zurück zu int16"""
    return np.clip(audio * 32767.0, -32768, 32767).astype(np.int16) # clip wählt einen Bereich aus


#### Hauptfunktionen ####

def read_audio_properties(audio_bytes) -> tuple[int, int, int, int, float, tuple]:
    """Liest grundlegende WAV-Parameter"""

    try:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            duration = nframes / framerate if framerate else 0
            params = wf.getparams()
        return nchannels, sampwidth, framerate, nframes, duration, params
    except Exception as e:
        st.error(f"Fehler beim Lesen der WAV-Datei: {e}")
        return None, None, None, None, None, None


def convert_to_target_format(audio_bytes, target_channels=2, target_rate=48000) -> bytes:
    """Konvertiert beliebige WAV-Datei in Zielformat (z.B. für ML-Modelle)"""

    try:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())

        # Normalisieren
        audio_data = normalize_audio(frames, sampwidth)

        # Mehrkanalig reshapen (falls nötig)
        if nchannels > 1:
            audio_data = audio_data.reshape(-1, nchannels)

        # Resampling
        if framerate != target_rate:
            st.info(f"🔄 Resample: {framerate} Hz ➜ {target_rate} Hz")
            if nchannels == 1:
                audio_data = resample_audio(audio_data, framerate, target_rate)
            else:
                audio_data = np.column_stack([
                    resample_audio(audio_data[:, ch], framerate, target_rate)
                    for ch in range(nchannels)
                ])

        # Kanal-Konvertierung
        if nchannels != target_channels:
            st.info(f"🎧 Konvertiere Kanäle: {nchannels} ➜ {target_channels}")
            audio_data = convert_channels(audio_data, nchannels, target_channels)

        # Zurück in 16-bit PCM WAV
        int16_data = float32_to_int16(audio_data)

        # Schreiben
        output_io = io.BytesIO()
        with wave.open(output_io, 'wb') as wf:
            wf.setnchannels(target_channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(target_rate)
            wf.writeframes(int16_data.tobytes())
        output_io.seek(0)

        return output_io.read()

    except Exception as e:
        st.error(f"Fehler bei der Konvertierung: {e}")
        st.warning("⚠️ Rückfall auf Original-Audio")
        return audio_bytes
