# Bibliotheken
import streamlit as st

# Module importieren
from utils import metadata_parser as mp
from utils import visualisation as vis

# Konfiguration der Seite
st.set_page_config(
    page_title="Dashboard",
    page_icon="📊", # Favicon
    layout="wide", # nutzt die gesamte zur Verfügung stehende Bildschirmbreite
    initial_sidebar_state="expanded" # Seitenleiste ist standardmäßig ausgeklappt.
)

def main():
    # Header
    st.title("📊 Überblick über das RAVDESS Dataset")
    st.divider()
    st.markdown("### Willkommen zum RAVDESS Emotional Speech Dataset Explorer!")
    st.markdown("Hier erhalten Sie einen umfassenden Überblick über den Datensatz.")
    
    # Sidebar für Konfiguration
    with st.sidebar:
        st.header("⚙️ Konfiguration")
        
        # Datenpfad Eingabe
        default_path = "data/ravdess"
        data_path = st.text_input(
            "Pfad zu den RAVDESS Daten:",
            value=default_path,
            help="Geben Sie den Pfad zum Verzeichnis mit den RAVDESS WAV-Dateien an"
        )
        
        # Aktualisieren Button
        if st.button("🔄 Daten neu laden"):
            st.cache_data.clear()
            st.rerun()
    
    # Lädt Daten
    with st.spinner("📂 Lade RAVDESS Daten..."):
        df = mp.load_ravdess_metadata(data_path)
    
    # Prüft ob Daten geladen wurden
    if df.empty:
        st.error(f"""
        ❌ **Keine RAVDESS Dateien gefunden!**
        
        Bitte überprüfen Sie:
        - Ist der Pfad `{data_path}` korrekt?
        - Befinden sich WAV-Dateien im angegebenen Verzeichnis?
        - Haben die Dateien das korrekte RAVDESS Namensformat?
        
        Erwartetes Format: `Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor.wav`
        """)
        st.stop()
    
    # Berechnet durchschnittliche Audiodauer
    length_avg = mp.calculate_length_average(df)

    # Hauptmetriken
    st.markdown("### 📈 Übersicht")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="📦 Sprachdateien",
            value=len(df),
            help="Gesamtanzahl der gefundenen RAVDESS Audiodateien"
        )
    
    with col2:
        st.metric(
            label="🧑‍🤝‍🧑 Sprecherinnen & Sprecher",
            value=df['actor_id'].nunique(),
            help="Anzahl der verschiedenen Sprechenden im Datensatz"
        )
    
    with col3:
        st.metric(
            label="😊 Emotionen",
            value=df['emotion'].nunique(),
            help="Anzahl der verschiedenen Emotionskategorien"
        )
    
    with col4:
        st.metric(
            label="🎭 Aussagen",
            value=df['statement'].nunique(),
            help="Anzahl der verschiedenen gesprochenen Aussagen"
        )
    
    with col5:
        st.metric(
            label="⏱️ Durchschnittliche Dauer",
            value=f"{length_avg:.2f} s",
            help="Durchschnittliche Länge der Audiodateien in Sekunden"
        )

    st.divider()
    
    # Visualisierungen
    st.markdown("### 📊 Detailanalyse")
    
    # Emotionsverteilung
    st.plotly_chart(
        vis.create_emotion_distribution_chart(df), 
        use_container_width=True
    )
    
    # Geschlecht und Intensität
    st.plotly_chart(
        vis.create_gender_intensity_charts(df), 
        use_container_width=True
    )
    
    # Datentabelle
    with st.expander("📋 Rohdaten anzeigen"):
        st.markdown("### Sample der geladenen Daten:")
        st.dataframe(
            df.sample(20),
            use_container_width=True,
            hide_index=True
        )
        
        # Download Button für vollständige Daten
        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Vollständige Metadaten als CSV herunterladen",
            data=csv,
            file_name="ravdess_metadata.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()