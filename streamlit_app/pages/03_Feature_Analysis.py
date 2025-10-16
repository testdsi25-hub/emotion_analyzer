# Bibliothek
import streamlit as st
import pandas as pd

# Module importieren
from utils import metadata_parser as mp
from utils import audio_features as af
from utils import visualisation as vis

# Konfiguration der Seite
st.set_page_config(
    page_title="Feature-Analyse", 
    page_icon="🔎", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

def main():
    st.title("🔎 Feature-Analyse: Akustische Merkmale")
    st.divider()
    st.markdown("### Einblicke in die akustische Struktur der Emotionen")
    
    # Sidebar für Konfiguration (gleich wie Audio Features)
    with st.sidebar:
        st.header("⚙️ Konfiguration")
        data_path = "data/ravdess"
        st.markdown(f"**Datenpfad:** `{data_path}`")
        st.divider()
        
        # Feature Extraktion Einstellungen
        st.subheader("🔧 Feature Extraktion")
        max_files = st.number_input(
            "Max. Anzahl Dateien (0 = alle):",
            min_value=0,
            max_value=1500,
            value=0,
            help="Begrenzt die Anzahl der zu verarbeitenden Dateien für Tests"
        )
        
        # Cache leeren
        if st.button("🔄 Features neu berechnen", help="Klick leert den Cache, um die Features neu zu berechnen."):
            st.cache_data.clear()
            st.rerun()
    
    # Metadaten laden
    with st.spinner("📂 Lade RAVDESS Metadaten..."):
        try:
            df_metadata = mp.load_ravdess_metadata(data_path)
        except Exception as e:
            st.error(f"Fehler beim Laden der Metadaten: {e}")
            st.stop()
    
    if df_metadata.empty:
        st.error(f"""
        ❌ **Keine RAVDESS Dateien gefunden!**
        
        Bitte überprüfen Sie den Pfad: `{data_path}`
        """)
        st.stop()
    
    st.success(f"✅ {len(df_metadata)} Audio-Dateien gefunden")
    
    # Feature Extraktion
    st.divider()
    st.markdown("### 🔧 Feature-Extraktion")
    
    max_files_to_process = max_files if max_files > 0 else len(df_metadata)
    
    if st.button("🚀 Features extrahieren", type="primary"): # primary: auffälligeres Design als die normalen Buttons
        with st.spinner(f"🧮 Extrahiere Features aus {max_files_to_process} Dateien ..."):
            df_features = af.extract_features_for_dataset(
                df_metadata, 
                max_files=max_files if max_files > 0 else None
            )
            
            # Speichert Features in Session State
            st.session_state['df_features'] = df_features
            st.success(f"✅ Features für {len(df_features)} Dateien extrahiert!")
    
    # Prüft, ob Features verfügbar sind
    if 'df_features' not in st.session_state:
        st.info("👆🏽 Klicken Sie auf 'Features extrahieren', um zu beginnen.")
        st.stop()
    
    df_features = st.session_state['df_features']
    
    if df_features.empty:
        st.error("❌ Keine Features extrahiert. Bitte versuchen Sie es erneut.")
        st.stop()
    
    # Feature Übersicht
    st.divider()
    st.markdown("### 📈 Feature-Übersicht")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎵 Verarbeitete Dateien", len(df_features))
    with col2:
        st.metric("📊 Extrahierte Features", len([col for col in df_features.columns if col not in ['filename', 'emotion', 'intensity', 'gender', 'actor_id', 'statement']]))
    with col3:
        st.metric("😊 Emotionen", df_features['emotion'].nunique())
    with col4:
        st.metric("⏱️ Durchschnittliche Dauer", f"{df_features['duration'].mean():.2f}s")
    
    # Hauptanalyse
    st.divider()
    st.markdown("### 📊 Detailanalyse")
    
    # Tab-Navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📏 Grundlegende Features", 
        "📈 Statistiken", 
        "🔗 Korrelationen",
        "🎯 Emotion-Vergleich",
        "🎧 MFCC Analyse"
    ])
    
    basic_features = af.get_basic_features_list()
    
    # Tab 1: Grundlegende Features
    with tab1:
        st.subheader("📏 Analyse der Grundlegenden Audio-Features")
        
        # Feature Auswahl
        selected_feature = st.selectbox(
            "Wählen Sie ein Feature zur Analyse:",
            basic_features,
            format_func=lambda x: af.get_feature_descriptions()['basic_features'].get(x, {}).get('name', x)
        )
        
        if selected_feature in df_features.columns:

            # Feature-Metadaten holen
            feature_info = af.get_feature_descriptions()['basic_features'].get(selected_feature, {})
            feature_info_des = feature_info.get('description', 'Keine Beschreibung vorhanden.')
            feature_info_ipr = feature_info.get('interpretation', 'Keine Interpretation vorhanden.')
            st.markdown(
                    f"""
                    <div style='line-height: 1.6; font-size: 0.95rem; margin-bottom: 1.5rem;'>
                    💡 <strong>Interpretationshilfe</strong><br>
                    Beschreibung: &nbsp;&nbsp;&nbsp;&nbsp; {feature_info_des}<br>
                    Interpretation: &nbsp;&nbsp;&nbsp;&nbsp; {feature_info_ipr}
                    </div>
                    """,
                    unsafe_allow_html=True
                ) # Layout mit html angepasst

            col1, col2 = st.columns(2)
            
            with col1:
                # Boxplot
                fig_box = vis.create_feature_boxplot(df_features, selected_feature, "nach Emotion")
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                # Histogram
                fig_hist = vis.create_feature_histogram(df_features, selected_feature)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Statistik-Tabelle für das gewählte Feature
            st.subheader(f"📊 Statistiken: {af.get_feature_descriptions()['basic_features'].get(selected_feature, {}).get('name', selected_feature)}")
            stats_df = af.calculate_feature_statistics(df_features, selected_feature)
            st.dataframe(stats_df, use_container_width=True)
    
    # Tab 2: Statistiken
    with tab2:
        st.subheader("📈 Statistische Übersicht aller Features")
        
        # Wähle Gruppierung
        group_by = st.selectbox("Gruppieren nach:", ['emotion', 'gender', 'intensity'])
        
        # Erstelle Statistiken für alle basic features
        stats_summary = []
        for feature in basic_features:
            if feature in df_features.columns:
                stats = af.calculate_feature_statistics(df_features, feature, group_by)
                stats['feature'] = feature
                stats_summary.append(stats)
        
        if stats_summary:
            combined_stats = pd.concat(stats_summary, ignore_index=True)
            
            # Pivot für bessere Darstellung
            pivot_stats = combined_stats.pivot_table(
                index='feature',
                columns=group_by,
                values='mean',
                fill_value=0
            )
            
            st.dataframe(pivot_stats.round(3), use_container_width=True)
            
            # Download Option
            csv = combined_stats.to_csv(index=False)
            st.download_button(
                label="📥 Statistiken als CSV herunterladen",
                data=csv,
                file_name="ravdess_feature_statistics.csv",
                mime="text/csv"
            )
    
    # Tab 3: Korrelationen
    with tab3:
        st.subheader("🔗 Feature-Korrelationen")
        st.write("Zeigt Zusammenhänge zwischen verschiedenen grundlegenden Audio-Features.")
        
        available_basic_features = [f for f in basic_features if f in df_features.columns]
        
        if len(available_basic_features) > 1:
            fig_corr = vis.create_correlation_heatmap(df_features, available_basic_features)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Korrelationstabelle
            corr_matrix = df_features[available_basic_features].corr().round(3)
            st.subheader("📊 Korrelationsmatrix")
            st.dataframe(corr_matrix, use_container_width=True)
    
    # Tab 4: Emotion-Vergleich
    with tab4:
        st.subheader("🎯 Emotion-Vergleich im Feature-Raum")
        
        available_features = [f for f in basic_features if f in df_features.columns]
        
        if len(available_features) >= 3:
            # Radar Chart
            selected_features_radar = st.multiselect(
                "Features für Radar-Chart auswählen:",
                available_features,
                default=available_features[:6],
                help="Wählen Sie 3-8 Features für optimale Darstellung"
            )
            
            if len(selected_features_radar) >= 3:
                fig_radar = vis.create_emotion_comparison_radar(df_features, selected_features_radar)
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("Bitte wählen Sie mindestens 3 Features aus.")
        
        # Emotion-Durchschnitte Tabelle
        st.subheader("📊 Durchschnittswerte pro Emotion")
        emotion_averages = df_features.groupby('emotion')[available_features].mean().round(3)
        st.dataframe(emotion_averages, use_container_width=True)
    
    # Raw Data Export
    with st.expander("📋 Rohdaten anzeigen"):
        st.subheader("Extrahierte Features (Sample)")
        st.dataframe(df_features.head(20), use_container_width=True)
        
        # Download Button
        csv = df_features.to_csv(index=False)
        st.download_button(
            label="📥 Alle Features als CSV herunterladen",
            data=csv,
            file_name="ravdess_extracted_features.csv",
            mime="text/csv"
        )

    # Tab 5: MFCC Analyse
    with tab5:
        st.subheader("🎧 MFCC (Mel-Frequency Cepstral Coefficients) Analyse")
        st.write("MFCCs beschreiben die spektralen Eigenschaften der Stimme und sind wichtig für Spracherkennung.")
        
        # MFCC Heatmap
        fig_mfcc = vis.create_mfcc_heatmap(df_features)
        st.plotly_chart(fig_mfcc, use_container_width=True)
        
        # MFCC Feature-Infos holen
        mfcc_descriptions = af.get_feature_descriptions().get('mfcc_features', {})
        
        with st.expander("💡Interpretationshilfe zu den MFCC-Features"):
            for feature_key, feature_info in mfcc_descriptions.items():
                name = feature_info.get('name', feature_key)
                description = feature_info.get('description', 'Keine Beschreibung vorhanden.')
                interpretation = feature_info.get('interpretation', 'Keine Interpretation vorhanden.')

                st.markdown(f"**{name}**")
                st.markdown(
                    f"""
                    <div style='line-height: 1.6; font-size: 0.95rem; margin-bottom: 1.5rem;'>
                    <strong>Beschreibung:</strong> {description}<br>
                    <strong>Interpretation:</strong> {interpretation}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # MFCC Statistiken
        mfcc_features = af.get_mfcc_features_list()
        available_mfcc = [f for f in mfcc_features if f in df_features.columns]
        
        if available_mfcc:
            st.subheader("📊 MFCC Durchschnittswerte pro Emotion")
            mfcc_stats = df_features.groupby('emotion')[available_mfcc].mean().round(3)
            st.dataframe(mfcc_stats, use_container_width=True)
    

if __name__ == "__main__":
    main()