# Bibliothek
import streamlit as st

# Import der eigenen Module
from utils import metadata_parser as mp
from utils import audio_features as af
from utils import visualisation as vis

# Konfiguration der Seite
st.set_page_config(
    page_title="Visualizer",
    page_icon="🎨",
    layout="wide"
)

# Hauptfunktion der Seite
def main():
    st.title("🎨 Visualizer einzelner Audio-Dateien")    
    st.info("🔎 **Fokus dieser Seite:** Technische Visualisierung einzelner Audio-Signale (Wellenform, Spektrogramme, Pitch-Tracking)")
    
    # Sidebar für Konfiguration (gleich wie Audio Features)
    with st.sidebar:
        st.header("⚙️ Konfiguration")
        data_path = "data/ravdess"
        st.markdown(f"**Datenpfad:** `{data_path}`")
        st.divider()
    
    # Metadaten laden
    with st.spinner("🔄 Lade RAVDESS-Metadaten..."):
        try:
            metadata_df = mp.load_ravdess_metadata(data_path)
            
            if metadata_df.empty:
                st.error("❌ Keine gültigen RAVDESS-Dateien gefunden.")
                st.stop()
                
            st.success(f"✅ {len(metadata_df)} Audiodateien gefunden!")
            
        except Exception as e:
            st.error(f"❌ Fehler beim Laden der Metadaten: {str(e)}")
            st.stop()
    
    # Hinweis zu Features Analysis
    st.markdown("""
    💡 **Tipp:** Für statistische Analysen und Vergleiche zwischen Emotionen besuchen Sie die **Feature Analysis**, siehe **Seitenleiste**.
    """)
    
    # Dateiauswahl
    st.divider()
    st.subheader("🎯 Datei auswählen")
    
    # Filter-Optionen
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_emotion = st.selectbox(
            "🎭 Emotion filtern:",
            options=['Alle'] + sorted(metadata_df['emotion'].unique().tolist())
        )
    
    with col2:
        selected_gender = st.selectbox(
            "👥 Geschlecht filtern:",
            options=['Alle'] + sorted(metadata_df['gender'].unique().tolist())
        )
    
    with col3:
        selected_intensity = st.selectbox(
            "🎚️ Intensität filtern:",
            options=['Alle'] + sorted(metadata_df['intensity'].unique().tolist())
        )
    
    # DataFrame filtern
    filtered_df = metadata_df.copy()
    
    if selected_emotion != 'Alle':
        filtered_df = filtered_df[filtered_df['emotion'] == selected_emotion]
    if selected_gender != 'Alle':
        filtered_df = filtered_df[filtered_df['gender'] == selected_gender]
    if selected_intensity != 'Alle':
        filtered_df = filtered_df[filtered_df['intensity'] == selected_intensity]
    
    if filtered_df.empty:
        st.warning("⚠️ Keine Dateien entsprechen den gewählten Filtern.")
        st.stop()
    
    # Dateiauswahl-Dropdown
    file_options = []
    for _, row in filtered_df.iterrows():
        display_name = f"{row['emotion'].title()} - {mp.GENDER_DISPLAY_MAP.get(row['gender'])} - Actor {row['actor_id']:02d} - {row['filename']}"
        file_options.append((display_name, row['file_path'], dict(row)))
    
    selected_display, selected_path, selected_metadata = st.selectbox(
        "🎵 Audiodatei wählen:",
        options=file_options,
        format_func=lambda x: x[0]
    )
    
    # Audio-Player
    st.markdown("##### 🎧 Audio-Wiedergabe")
    try:
        audio_file = open(selected_path, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')
    except Exception as e:
        st.error(f"❌ Fehler beim Laden der Audiodatei: {str(e)}")
    
    # Metadaten anzeigen
    st.markdown("---")
    st.markdown("### ℹ️ Information zur gewählten Audio-Datei")
    mp.display_audio_metadata(selected_metadata)
    
    # Audio-Features extrahieren
    st.markdown("---")
    st.subheader("🔧 Feature-Extraktion")
    
    with st.spinner("🔄 Extrahiere Audio-Features..."):
        features = af.extract_audio_features(selected_path)
        
        if features:
            mp.display_extracted_features(features)
        else:
            st.error("❌ Fehler bei der Feature-Extraktion.")
            st.stop()
    
    # Audio-Visualisierungen
    st.subheader("📊 Audio-Visualisierungen")
    
    # Lade Audiodaten für Visualisierung
    with st.spinner("🔄 Berechne Visualisierungen..."):
        audio_data = mp.load_audio_data(selected_path)
        
        if audio_data is None:
            st.error("❌ Fehler beim Laden der Audiodaten für Visualisierung.")
            st.stop()
    
    # Tab-Layout für verschiedene Visualisierungen
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Wellenform", "🌈 Mel-Spektrogramm", "🔥 MFCC-Heatmap", "🎼 Pitch-Tracking"])
    
    with tab1:
        st.markdown("""
        **📉 Wellenform:** Zeigt die Amplitude (Lautstärke) der Audiodatei über die Zeit. 
        Hohe Peaks bedeuten laute Abschnitte, niedrige Werte sind leise Passagen.
        """)
        waveform_fig = vis.create_waveform_plot(audio_data)
        st.plotly_chart(waveform_fig, use_container_width=True)

         # Interpretationshilfe
        st.subheader("💡 Signal-Interpretationshilfe")

        with st.expander("🤔 Wie interpretiere ich dieses Audio-Signal?"):
            st.markdown("""
            ### 📉 Wellenform (Amplitude über Zeit)
            - **Hohe Peaks**: Laute Silben oder Betonungen
            - **Niedrige Bereiche**: Pausen oder leise Passagen
            - **Gleichmäßige Amplitude**: Ruhiges, kontrolliertes Sprechen
            - **Unregelmäßige Amplitude**: Emotionale oder aufgeregte Sprache
                        """)

    with tab2:
        st.markdown("""
        **🌈 Mel-Spektrogramm:** Visualisiert die Frequenzen über die Zeit in einer für das menschliche Gehör optimierten Skala. 
        Hellere Bereiche zeigen stärkere Frequenzkomponenten. Vertikale Linien sind oft Konsonanten, horizontale Bänder Vokale.
        """)
        mel_spec_fig = vis.create_mel_spectrogram_plot(audio_data)
        st.plotly_chart(mel_spec_fig, use_container_width=True)

        # Interpretationshilfe
        st.subheader("💡 Signal-Interpretationshilfe")

        with st.expander("🤔 Wie interpretiere ich dieses Audio-Signal?"):
            st.markdown("""
            ### 🌈 Mel-Spektrogramm (Frequenz-Zeit-Analyse)
            - **Horizontale Bänder**: Vokale (a, e, i, o, u) mit charakteristischen Formanten
            - **Vertikale Linien**: Konsonanten mit kurzer, breitbandiger Energie
            - **Helle untere Bereiche**: Grundfrequenz und Obertöne
            - **Dunkle Bereiche**: Geringe Energie in diesen Frequenzbereichen
                        """)
    
    with tab3:
        st.markdown("""
        **🔥 MFCC-Heatmap:** Zeigt die 13 wichtigsten spektralen Merkmale über die Zeit. 
        MFCC-Koeffizienten erfassen die charakteristische "Klangfarbe" der Stimme und sind essentiell für Spracherkennung.
        """)
        mfcc_fig = vis.create_mfcc_heatmap_plot(audio_data)
        st.plotly_chart(mfcc_fig, use_container_width=True)

        # Interpretationshilfe
        st.subheader("💡 Signal-Interpretationshilfe")
        
        with st.expander("🤔 Wie interpretiere ich dieses Audio-Signal?"):
            st.markdown("""
            ### 🔥 MFCC-Heatmap (Spektrale Koeffizienten)
            - **MFCC 1**: Gesamtenergie des Signals
            - **MFCC 2-4**: Charakteristische Sprachmerkmale
            - **MFCC 5-13**: Feinere spektrale Details
            - **Rot/Warm**: Positive Werte, **Blau/Kalt**: Negative Werte
                        """)
    
    with tab4:
        st.markdown("""
        **🎼 Pitch-Tracking:** Verfolgt die Grundfrequenz (Tonhöhe) der Stimme über die Zeit. 
        Höhere Werte bedeuten höhere Stimme, Sprünge zeigen Intonationsänderungen. Emotionen beeinflussen oft das Pitch-Muster.
        """)
        pitch_fig = vis.create_pitch_plot(audio_data)
        st.plotly_chart(pitch_fig, use_container_width=True)
        # Interpretationshilfe
        st.subheader("💡 Signal-Interpretationshilfe")

        with st.expander("🤔 Wie interpretiere ich dieses Audio-Signal?"):
            st.markdown("""
            ### 🎼 Pitch-Tracking (Grundfrequenz)
            - **Konstante Linien**: Monotone Sprache
            - **Aufwärts-Trends**: Fragende Intonation
            - **Abwärts-Trends**: Aussagesätze
            - **Sprünge**: Emotionale Betonung oder Wortgrenzen
                        """)

if __name__ == "__main__":
    main()