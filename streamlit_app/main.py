import streamlit as st

# Konfiguration der Seite
st.set_page_config(
    page_title="Home",
    page_icon="🏡", # Favicon
    layout="wide", # nutzt die gesamte zur Verfügung stehende Bildschirmbreite
    initial_sidebar_state="expanded" # Seitenleiste ist standardmäßig ausgeklappt.
)

st.title("🎵 Emotionen entdecken - mit Stimme, Grafiken und ML")
st.markdown(
    """
    Diese Anwendung analysiert und visualisiert Sprachaufnahmen aus dem [RAVDESS-Datensatz](https://zenodo.org/record/1188976).
    """)

st.subheader("👉🏽 Wähle eine Seite im Menü auf der linken Seite, um loszulegen!")
st.markdown(
    """
    📊 **Dashboard:** Verschaffen Sie sich einen schnellen Überblick, wie der Datensatz aufgebaut ist.

    🎧 **Audio Explorer:** Entdecken Sie die Daten und hören Sie sie an. 

    🔎 **Feature Analysis:** Gewinnen Sie Einblicke in die akustische Struktur der Emotionen.

    🎨 **Visualizer:** Lassen Sie sich akustische Merkmale visuell erklären. 

    ❤️ **Emotion Analyzer:** Testen Sie das Modell und lassen Sie Emotionen aus Sprachdaten analysieren. 

    📝 **Über das Projekt:** Blicken Sie hinter die Kulissen dieses Projekts.
    """
)

