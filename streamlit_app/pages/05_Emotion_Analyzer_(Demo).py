# Bibliothek
import streamlit as st
import tempfile
import numpy as np
import matplotlib.pyplot as plt

# Module importieren
from utils import record as rec
from utils import emotion_predictor as emp
from utils import metadata_parser as mp
from utils import visualisation as vis

# Konfiguration der Seite
st.set_page_config(
    page_title="Emotion Analyzer",
    page_icon="❤️", # Favicon
    layout="wide", # nutzt die gesamte zur Verfügung stehende Bildschirmbreite
    initial_sidebar_state="expanded" # Seitenleiste ist standardmäßig ausgeklappt.
)

def main():
    st.title("❤️ Emotion Analyzer (Demo)")
    st.subheader("🎙️ Audioaufnahme für RAVDESS-trainiertes Modell")
    st.write("Laden Sie eine WAV-Datei hoch oder nehmen Sie Audio direkt im Browser auf.")
    
    # Tab-Interface für verschiedene Input-Methoden
    tab1, tab2 = st.tabs(["📁 Datei hochladen", "🎤 Browser-Aufnahme"])
    
    audio_bytes = None
    
    with tab1:
        st.subheader("WAV-Datei hochladen")
        uploaded_file = st.file_uploader(
            "Wählen Sie eine WAV-Datei aus", 
            type=['wav'], 
            help="Unterstützte Formate: WAV"
        )
        
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            st.success("✅ Datei erfolgreich hochgeladen!")
    
    with tab2:
        st.subheader("Browser-Aufnahme")
        st.info("💡 **Anleitung:** Verwenden Sie das native Streamlit Audio-Input Widget unten.")
        
        # Streamlit's native audio input
        recorded_audio = st.audio_input("Sprechen Sie jetzt:")
        
        if recorded_audio is not None:
            audio_bytes = recorded_audio.read()
            st.success("✅ Aufnahme erfolgreich!")
    
    # Audio-Verarbeitung, wenn Audio verfügbar ist
    if audio_bytes:
        st.divider()
        
        # Zeige ursprüngliche Audio-Eigenschaften
        st.subheader("📊 Original Audio-Eigenschaften:")
        props = rec.read_audio_properties(audio_bytes)
        
        if props[0] is not None:
            nchannels, sampwidth, framerate, nframes, duration, params = props
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Kanäle", nchannels)
                st.metric("Sample Width", f"{sampwidth} Bytes")
            
            with col2:
                st.metric("Sample Rate", f"{framerate:,} Hz")
                st.metric("Frames", f"{nframes:,}")
            
            with col3:
                st.metric("Dauer", f"{duration:.2f} s")
                if nchannels == 1:
                    st.info("📢 Mono")
                else:
                    st.info("🎵 Stereo")
            
            with st.expander("🔍 Detaillierte Parameter"):
                st.code(f"Komplette Parameter: {params}", language="python")
        
        # Spiele Original-Audio ab
        st.subheader("🔊 Original-Aufnahme:")
        st.audio(audio_bytes, format='audio/wav')
        
        # RAVDESS-Konvertierung
        st.subheader("🔄 RAVDESS-kompatible Konvertierung:")
        
        if props[0] is not None:
            nchannels, sampwidth, framerate, nframes, duration, params = props
            
            needs_conversion = nchannels != 1 or framerate != 48000
            
            if needs_conversion:
                st.warning(f"⚠️ Konvertierung erforderlich: {nchannels} Kanal(e) → 1 Kanal, {framerate} Hz → 48000 Hz")
                
                with st.spinner("Konvertiere zu RAVDESS-Format (48 kHz Stereo)..."):
                    converted_audio = rec.convert_to_target_format(audio_bytes, 1, 48000)
                    
                    # Zeige konvertierte Eigenschaften
                    conv_props = rec.read_audio_properties(converted_audio)
                    if conv_props[0] is not None:
                        nchannels_conv, sampwidth_conv, framerate_conv, nframes_conv, duration_conv, params_conv = conv_props
                        
                        st.success("✅ Erfolgreich konvertiert!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Kanäle", nchannels_conv, delta=nchannels_conv-nchannels)
                            st.metric("Sample Width", f"{sampwidth_conv} Bytes")
                        
                        with col2:
                            st.metric("Sample Rate", f"{framerate_conv:,} Hz", delta=framerate_conv-framerate)
                            st.metric("Frames", f"{nframes_conv:,}")
                        
                        with col3:
                            st.metric("Dauer", f"{duration_conv:.2f} s")
                            st.success("🎯 RAVDESS-ready!")
                        
                        # Spiele konvertiertes Audio ab
                        st.subheader("🔊 Konvertierte Aufnahme (48 kHz Stereo):")
                        st.audio(converted_audio, format='audio/wav')
                        
                        # Session State für weitere Verwendung
                        st.session_state["original_audio"] = audio_bytes
                        st.session_state["ravdess_audio"] = converted_audio
                        
            else:
                st.success("✅ Audio ist bereits RAVDESS-kompatibel (48 kHz Stereo)!")
                converted_audio = audio_bytes
                st.session_state["original_audio"] = audio_bytes
                st.session_state["ravdess_audio"] = audio_bytes
            
            st.divider()
            st.subheader("🧠 Emotionserkennung")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                tmpfile.write(st.session_state["ravdess_audio"])
                temp_audio_path = tmpfile.name

            with st.spinner("🔍 Analysiere Emotion..."):
                emotion, probs, labels = emp.predict_emotion(temp_audio_path)

            st.success(f"🎧 Erkannte Emotion: **{emotion.upper()}**")

            # Visualisierung
            fig = vis.plot_emotion_prediction(labels, probs)
            st.plotly_chart(fig, use_container_width=True)

           
            # Tabelle anzeigen
            st.subheader("🧾 Wahrscheinlichkeiten")
            st.dataframe({
                "Emotion": labels,
                "Wahrscheinlichkeit (%)": np.round(probs * 100, 2)
            })
            

            # Download-Buttons
            st.divider()
            st.subheader("💾 Download")
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📁 Original WAV herunterladen", 
                    data=audio_bytes, 
                    file_name="original_aufnahme.wav", 
                    mime="audio/wav",
                    help="Ursprüngliche Aufnahme/hochgeladene Datei"
                )
            
            with col2:
                st.download_button(
                    "🎯 RAVDESS WAV herunterladen", 
                    data=converted_audio if needs_conversion else audio_bytes, 
                    file_name="ravdess_kompatibel.wav", 
                    mime="audio/wav",
                    help="48 kHz Stereo Format für RAVDESS-Modell"
                )
            
    else:
        st.info("👆 Laden Sie eine WAV-Datei hoch oder nehmen Sie Audio auf, um zu beginnen.")

if __name__ == "__main__":
    main()