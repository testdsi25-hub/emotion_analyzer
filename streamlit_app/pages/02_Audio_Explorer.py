# Bibliothek
import streamlit as st

# Module importieren
from utils import metadata_parser as mp

# Konfiguration der Seite
st.set_page_config(
    page_title="Audio-Explorer", 
    page_icon="🎧", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

def main():
    st.title("🎧 Audio-Explorer")
    st.divider()
    
    # Sidebar für Konfiguration 
    with st.sidebar:
        st.header("⚙️ Konfiguration")
        data_path = "data/ravdess"
        st.markdown(f"**Datenpfad:** `{data_path}`")
        st.divider()

    # Audio-Dateien laden mit der gemeinsamen Funktion
    with st.spinner("Lade RAVDESS Audio-Dateien..."):
        df = mp.load_ravdess_metadata(data_path)
    
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
    
    st.success(f"✅ {len(df)} RAVDESS Audio-Dateien gefunden")
    
     # Zusätzliche Informationen
    with st.expander("ℹ️ Informationen"):
        st.markdown("""
        **Über den RAVDESS Audio-Explorer:**
        - Durchsuchen Sie den RAVDESS Speech Datensatz
        - 8 Emotionen: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
        - 24 professionelle Schauspielende (Ungerade Zahlen sind männlich, gerade Zahlen sind weiblich.)
        - 2 Intensitätsstufen: Normal und Strong
        - 2 Sätze: "Kids are talking by the door" und "Dogs are sitting by the door"
        
        **RAVDESS Dateiformat:** Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
        
        **Beispiel:** 03-01-06-01-02-01-12.wav
        - 03: Audio only
        - 01: Speech
        - 06: Fearful
        - 01: Normal intensity
        - 02: "Dogs are sitting by the door"
        - 01: 1st repetition
        - 12: Actor 12 (female)
        """)

    # Sidebar für Filter
    st.sidebar.header("🔎 Filter")
    
    # Emotionsfilter
    emotions = ['Alle'] + sorted(df['emotion'].unique().tolist())
    selected_emotion = st.sidebar.selectbox("Emotion:", emotions)
    
    # Sprecher-Filter
    speakers = ['Alle'] + sorted([f"Actor_{actor:02d}" for actor in df['actor_id'].unique()])
    selected_speaker = st.sidebar.selectbox("Sprecher:", speakers)
    
    # Geschlechtsfilter (deutsche Labels)
    genders = ['Alle', 'männlich', 'weiblich']
    selected_gender = st.sidebar.selectbox("Geschlecht:", genders)
    
    # Intensitätsfilter
    intensities = ['Alle'] + sorted(df['intensity'].unique().tolist())
    selected_intensity = st.sidebar.selectbox("Intensität:", intensities)

    # Filter für Satz (statement)
    statement_options = df['statement'].unique().tolist()
    selected_statement = st.sidebar.selectbox("Wähle einen Satz:", ["Alle"] + statement_options)

    # Filter für Wiederholung (repetition)
    repetition_options = sorted(df['repetition'].unique())
    selected_repetition = st.sidebar.selectbox("Wähle die Wiederholung:", ["Alle"] + repetition_options)
    
    # Filter anwenden
    filtered_df = df.copy()
    
    if selected_emotion != 'Alle':
        filtered_df = filtered_df[filtered_df['emotion'] == selected_emotion]
    
    if selected_speaker != 'Alle':
        actor_num = int(selected_speaker.replace('Actor_', ''))
        filtered_df = filtered_df[filtered_df['actor_id'] == actor_num]
    
    if selected_gender != 'Alle':
        gender_key = 'male' if selected_gender == 'männlich' else 'female'
        filtered_df = filtered_df[filtered_df['gender'] == gender_key]
    
    if selected_intensity != 'Alle':
        filtered_df = filtered_df[filtered_df['intensity'] == selected_intensity]
    
    if selected_statement != "Alle":
        filtered_df = filtered_df[filtered_df['statement'] == selected_statement]

    if selected_repetition != "Alle":
        filtered_df = filtered_df[filtered_df['repetition'] == selected_repetition]
    
    # Ergebnisse anzeigen
    st.subheader(f"📋 Gefilterte Ergebnisse ({len(filtered_df)} Dateien)")
    
    if filtered_df.empty:
        st.warning("🔍 Keine Dateien entsprechen den gewählten Filtern.")
        return
    
    # Layout: 2 Spalten
    col1, col2 = st.columns([4, 1])
    
    with col2:
        st.subheader("📊 Statistiken")
        
        # Emotionsverteilung
        emotion_counts = filtered_df['emotion'].value_counts()
        st.write("**Emotionen:**")
        for emotion, count in emotion_counts.items():
            st.write(f"• {emotion.title()}: {count}")
        
        st.divider()
        
        # Geschlechtsverteilung (mit deutschen Labels)
        gender_counts = filtered_df['gender'].map({'male': 'männlich', 'female': 'weiblich'}).value_counts()
        st.write("**Geschlecht:**")
        for gender, count in gender_counts.items():
            st.write(f"• {gender}: {count}")
    
    with col1:
        st.subheader("🎧 Audio-Dateien")
        
        # Pagination
        items_per_page = 5
        total_pages = (len(filtered_df) - 1) // items_per_page + 1
        
        if total_pages > 1:
            page = st.selectbox("Seite:", range(1, total_pages + 1))
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            page_df = filtered_df.iloc[start_idx:end_idx]
        else:
            page_df = filtered_df
        
        # Audio-Dateien anzeigen
        for idx, row in page_df.iterrows():
            # Formatiere Metadaten für Display
            display_data = mp.format_metadata_for_display(row)
            
            with st.expander(f"🎵 {display_data['filename']}", expanded=True):
                # Audio Player
                st.audio(display_data['filepath'])
                
                # Metadaten in 3 Spalten
                meta_col1, meta_col2, meta_col3 = st.columns(3)
                
                with meta_col1:
                    st.write("**📝 Emotion:**", display_data['emotion'])
                    st.write("**🎭 Sprecher:**", display_data['speaker_id'])
                
                with meta_col2:
                    st.write("**⚥ Geschlecht:**", display_data['gender'])
                    st.write("**🔊 Intensität:**", display_data['intensity'].title())
                
                with meta_col3:
                    st.write("**💬 Satz:**", display_data['sentence_type'])
                    st.write("**📁 Datei:**", display_data['filename'])

if __name__ == "__main__":
    main()