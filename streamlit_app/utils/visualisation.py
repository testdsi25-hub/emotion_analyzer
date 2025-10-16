# Bibliotheken
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import librosa
import pandas as pd
from plotly.graph_objects import Figure 

# Zugriff auf andere Module
from utils import audio_features as af
from utils import metadata_parser as mp

#### DASHBOARD ####

def create_emotion_distribution_chart(df) -> Figure:
    """Erstellt ein Balkendiagramm für die Emotionsverteilung."""

    # Datenbasis
    emotion_counts = df['emotion'].value_counts().reset_index()
    emotion_counts.columns = ['Emotion', 'Anzahl']
    
    # Farben für verschiedene Emotionen
    color_map = mp.EMOTION_COLOR_MAP
    
    # Balkendiagramm
    fig = px.bar(
        emotion_counts, 
        x='Emotion', 
        y='Anzahl',
        title='Verteilung der Emotionen',
        color='Emotion',
        color_discrete_map=color_map
    )
    
    # Layout
    fig.update_layout(
        showlegend=False,
        xaxis_title="Emotion",
        yaxis_title="Anzahl der Dateien",
        plot_bgcolor='rgba(0,0,0,0)', # transparente backgroundcolor im plot
        paper_bgcolor='rgba(0,0,0,0)' # transparente backgroundcolor im paper
    )
    
    return fig

def create_gender_intensity_charts(df) -> Figure:
    """Erstellt Charts für Geschlechts- und Intensitätsverteilung."""
    
    # Subplot mit zwei Diagrammen
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Verteilung nach Geschlecht', 'Verteilung nach Intensität'),
        specs=[[{"type": "pie"}, {"type": "pie"}]]
    )
    
    # Geschlechtsverteilung
    gender_counts = df['gender'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=gender_counts.index,
            values=gender_counts.values,
            name="Geschlecht",
            marker_colors=['#3498db', '#e74c3c']
        ),
        row=1, col=1
    )
    
    # Intensitätsverteilung
    intensity_counts = df['intensity'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=intensity_counts.index,
            values=intensity_counts.values,
            name="Intensität",
            marker_colors=['#2ecc71', '#f39c12']
        ),
        row=1, col=2
    )
    
    # Layout
    fig.update_layout(
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


#### FEATURE ANALYSIS ####

def create_feature_boxplot(df, feature, title_suffix="") -> Figure:
    """Erstellt Boxplot für ein Feature gruppiert nach Emotion."""
    
    color_map = mp.EMOTION_COLOR_MAP
    
    fig = px.box(
        df, 
        x='emotion', 
        y=feature,
        color='emotion',
        color_discrete_map=color_map,
        title=f"{af.get_feature_descriptions()['basic_features'].get(feature, {}).get('name', feature)} {title_suffix}",
        labels={'emotion': 'Emotion', feature: af.get_feature_descriptions()['basic_features'].get(feature, {}).get('unit', 'Wert')}
    )
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Emotion",
        yaxis_title=af.get_feature_descriptions()['basic_features'].get(feature, {}).get('unit', 'Wert')
    )
    
    return fig

def create_feature_histogram(df, feature) -> Figure:
    """Erstellt Histogram für ein Feature."""
    
    color_map = mp.EMOTION_COLOR_MAP
    
    fig = px.histogram(
        df, 
        x=feature, 
        color='emotion',
        color_discrete_map=color_map,
        title=f"Verteilung: {af.get_feature_descriptions()['basic_features'].get(feature, {}).get('name', feature)}",
        labels={feature: af.get_feature_descriptions()['basic_features'].get(feature, {}).get('unit', 'Wert')},
        marginal="box"  # Fügt Boxplot am Rand hinzu
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    
    return fig

def create_correlation_heatmap(df, features) -> Figure:
    """Erstellt Korrelations-Heatmap für Features."""
    
    # Berechne Korrelationsmatrix
    corr_matrix = df[features].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Korrelation zwischen Audio-Features",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_emotion_comparison_radar(df, features) -> Figure:
    """Erstellt Radar Chart zum Vergleich der Emotionen."""
    
    # Normalisiere Features für bessere Vergleichbarkeit
    emotion_stats = df.groupby('emotion')[features].mean()
    
    # Normalisierung (Min-Max Scaling)
    emotion_stats_normalized = (emotion_stats - emotion_stats.min()) / (emotion_stats.max() - emotion_stats.min())
    
    color_map = mp.EMOTION_COLOR_MAP
    
    fig = go.Figure()
    
    for emotion in emotion_stats_normalized.index:
        fig.add_trace(go.Scatterpolar(
            r=emotion_stats_normalized.loc[emotion].values,
            theta=features,
            fill='toself',
            name=emotion.title(),
            line_color=color_map.get(emotion, '#000000'),
            fillcolor=color_map.get(emotion, '#000000'),
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Emotionen im Feature-Vergleich (normalisiert)"
    )
    
    return fig

def create_mfcc_heatmap(df) -> Figure:
    """Erstellt Heatmap für MFCC Features."""
    
    mfcc_features = af.get_mfcc_features_list()
    
    # Durchschnittliche MFCC Werte pro Emotion
    mfcc_by_emotion = df.groupby('emotion')[mfcc_features].mean()
    
    fig = px.imshow(
        mfcc_by_emotion,
        title="MFCC Features nach Emotion",
        color_continuous_scale="magma",
        aspect="auto",
        labels={'x': 'MFCC Koeffizient', 'y': 'Emotion', 'color': 'Durchschnittswert'}
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

#### VISUALIZER ####

def create_waveform_plot(audio_data) -> Figure:
    """Erstellt eine interaktive Wellenform-Visualisierung."""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=audio_data['time'],
        y=audio_data['y'],
        mode='lines',
        name='Wellenform',
        line=dict(color='#3498db', width=1),
        hovertemplate='Zeit: %{x:.3f}s<br>Amplitude: %{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='📉 Wellenform der Audioaufnahme',
        xaxis_title='Zeit (s)',
        yaxis_title='Amplitude',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400,
        hovermode='x'
    )
    
    return fig

def create_mel_spectrogram_plot(audio_data) -> Figure:
    """Erstellt eine Mel-Spektrogramm Visualisierung."""
    
    mel_spec_db = audio_data['mel_spec_db']
    sr = audio_data['sr']
    
    # Zeit- und Frequenz-Arrays für die Achsen
    time_frames = librosa.frames_to_time(range(mel_spec_db.shape[1]), sr=sr)
    mel_frequencies = librosa.mel_frequencies(n_mels=128, fmin=0, fmax=sr/2)
    
    fig = go.Figure(data=go.Heatmap(
        z=mel_spec_db,
        x=time_frames,
        y=mel_frequencies,
        colorscale='magma',
        colorbar=dict(title="Intensität (dB)"),
        hovertemplate='Zeit: %{x:.2f}s<br>Frequenz: %{y:.0f}Hz<br>Intensität: %{z:.1f}dB<extra></extra>'
    ))
    
    fig.update_layout(
        title='🌈 Mel-Spektrogramm',
        xaxis_title='Zeit (s)',
        yaxis_title='Frequenz (Hz)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    
    return fig

def create_mfcc_heatmap_plot(audio_data) -> Figure:
    """Erstellt eine MFCC-Heatmap Visualisierung."""
    
    mfccs = audio_data['mfccs']
    sr = audio_data['sr']
    
    # Zeit-Array für x-Achse
    time_frames = librosa.frames_to_time(range(mfccs.shape[1]), sr=sr)
    mfcc_labels = [f'MFCC {i+1}' for i in range(mfccs.shape[0])]
    
    fig = go.Figure(data=go.Heatmap(
        z=mfccs,
        x=time_frames,
        y=mfcc_labels,
        colorscale='RdBu',
        colorbar=dict(title="MFCC Wert"),
        hovertemplate='Zeit: %{x:.2f}s<br>Koeffizient: %{y}<br>Wert: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='🔥 MFCC-Heatmap (Mel-Frequency Cepstral Coefficients)',
        xaxis_title='Zeit (s)',
        yaxis_title='MFCC Koeffizient',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    return fig

def create_pitch_plot(audio_data) -> Figure:
    """Erstellt eine Pitch-Tracking Visualisierung."""
    
    pitches = audio_data['pitches']
    magnitudes = audio_data['magnitudes']
    sr = audio_data['sr']
    
    # Extrahiere dominante Pitches
    pitch_values = []
    time_values = []
    
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:  # Nur gültige Pitch-Werte
            pitch_values.append(pitch)
            time_values.append(librosa.frames_to_time(t, sr=sr))
    
    if pitch_values:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_values,
            y=pitch_values,
            mode='markers+lines',
            name='Grundfrequenz',
            line=dict(color='#e74c3c', width=2),
            marker=dict(size=4),
            hovertemplate='Zeit: %{x:.3f}s<br>Frequenz: %{y:.1f}Hz<extra></extra>'
        ))
        
        fig.update_layout(
            title='🎼 Pitch-Tracking (Grundfrequenz)',
            xaxis_title='Zeit (s)',
            yaxis_title='Frequenz (Hz)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        
        return fig
    else:
        # Fallback wenn kein Pitch erkannt wurde
        fig = go.Figure()
        fig.add_annotation(
            text="Keine dominante Grundfrequenz erkannt",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        
        fig.update_layout(
            title='🎼 Pitch-Tracking (Grundfrequenz)',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    

#### EMOTION ANALYZER ####

def plot_emotion_prediction(labels, probs, title="📊 Emotion Prediction") -> Figure:
    """
    Erstellt ein farblich kodiertes Balkendiagramm der Emotionserkennungsergebnisse.
    """

    # DataFrame erstellen
    df = pd.DataFrame({
        "Emotion": labels,
        "Wahrscheinlichkeit": probs
    })

    # Balkendiagramm
    fig = px.bar(
        df,
        x="Emotion",
        y="Wahrscheinlichkeit",
        color="Emotion",
        color_discrete_map=mp.EMOTION_COLOR_MAP,
        title=title
    )

    # Layout-Feinschliff
    fig.update_layout(
        showlegend=False,
        xaxis_title="Emotion",
        yaxis_title="Wahrscheinlichkeit",
        yaxis_tickformat=".0%",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig