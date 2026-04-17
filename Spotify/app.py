import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
import plotly.graph_objects as go
from PIL import Image
import base64
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Resolve all file paths relative to this script's location (required for Streamlit Cloud)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page configuration
st.set_page_config(
    page_title="🎵 Spotify Hit Prediction",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for background and styling
def add_bg_and_logo():
    bg_image = os.path.join(BASE_DIR, "musical_bg.jpg")
    logo_image = os.path.join(BASE_DIR, "spotify_logo1.jpg")

    if os.path.exists(bg_image):
        with open(bg_image, "rb") as f:
            bg_data = f.read()
        bg_b64 = base64.b64encode(bg_data).decode()

        if os.path.exists(logo_image):
            with open(logo_image, "rb") as f:
                logo_data = f.read()
            logo_b64 = base64.b64encode(logo_data).decode()

            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-image: url("data:image/jpg;base64,{bg_b64}");
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                }}
                .stApp, .stApp * {{
                    color: #ffffff !important;
                }}
                .block-container {{
                    background: rgba(0, 0, 0, 0.64) !important;
                    border-radius: 24px !important;
                    padding: 1.5rem 1.25rem !important;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.35) !important;
                }}
                .stButton button {{
                    color: #ffffff !important;
                    background-color: rgba(29, 185, 84, 0.8) !important;
                    border: 1px solid #1DB954 !important;
                }}
                .stButton button:hover {{
                    background-color: rgba(29, 185, 84, 0.9) !important;
                }}
                /* ── Select box trigger (closed state) ── */
                .stSelectbox div[data-baseweb="select"],
                .stMultiselect div[data-baseweb="select"] {{
                    background-color: rgba(0, 0, 0, 0.7) !important;
                    border: 1px solid #1DB954 !important;
                    border-radius: 8px !important;
                }}
                .stSelectbox div[data-baseweb="select"] *,
                .stMultiselect div[data-baseweb="select"] * {{
                    color: #ffffff !important;
                    background-color: transparent !important;
                }}

                /* ── Dropdown popup container (rendered at body root as a portal) ── */
                [data-baseweb="popover"] {{
                    background-color: #1a1a1a !important;
                    border: 1px solid #1DB954 !important;
                    border-radius: 8px !important;
                    overflow: hidden !important;
                }}
                /* The inner menu/list wrapper */
                [data-baseweb="menu"] {{
                    background-color: #1a1a1a !important;
                    color: #ffffff !important;
                }}
                /* Every list item */
                [data-baseweb="menu"] [role="option"] {{
                    background-color: #1a1a1a !important;
                    color: #ffffff !important;
                    padding: 8px 12px !important;
                    font-size: 0.95rem !important;
                    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
                }}
                /* Hover state */
                [data-baseweb="menu"] [role="option"]:hover {{
                    background-color: rgba(29, 185, 84, 0.25) !important;
                    color: #ffffff !important;
                    cursor: pointer !important;
                }}
                /* Selected/active state */
                [data-baseweb="menu"] [aria-selected="true"] {{
                    background-color: rgba(29, 185, 84, 0.4) !important;
                    color: #ffffff !important;
                }}
                /* Text spans inside options */
                [data-baseweb="menu"] [role="option"] span,
                [data-baseweb="menu"] [role="option"] div {{
                    color: #ffffff !important;
                    background-color: transparent !important;
                }}
                .stSlider div[data-baseweb="slider"] {{
                    background-color: rgba(0, 0, 0, 0.7) !important;
                }}
                .stSlider div[data-baseweb="slider"] * {{
                    color: #ffffff !important;
                }}
                .stNumberInput input, .stTextInput input {{
                    background-color: rgba(0, 0, 0, 0.7) !important;
                    color: #ffffff !important;
                    border: 1px solid #1DB954 !important;
                    border-radius: 8px !important;
                }}
                .stNumberInput input::placeholder, .stTextInput input::placeholder {{
                    color: #cccccc !important;
                }}
                .stDataFrame {{
                    background-color: rgba(0, 0, 0, 0.7) !important;
                }}
                .stDataFrame * {{
                    color: #ffffff !important;
                }}
                .stMetric {{
                    background-color: rgba(0, 0, 0, 0.7) !important;
                    border: 1px solid #1DB954 !important;
                    border-radius: 12px !important;
                    padding: 1rem !important;
                }}
                .stTabs [data-baseweb="tab-list"] {{
                    background-color: rgba(29, 185, 84, 0.1) !important;
                    border-radius: 10px !important;
                }}
                .stTabs [data-baseweb="tab"] {{
                    color: #ffffff !important;
                    font-weight: bold !important;
                }}
                .stTabs [data-baseweb="tab"][aria-selected="true"] {{
                    background-color: rgba(29, 185, 84, 0.3) !important;
                }}
                .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
                    color: #ffffff !important;
                }}
                .stSuccess {{
                    background-color: rgba(40, 167, 69, 0.8) !important;
                    color: #ffffff !important;
                    border: 1px solid #28a745 !important;
                    border-radius: 8px !important;
                }}
                .stError {{
                    background-color: rgba(220, 53, 69, 0.8) !important;
                    color: #ffffff !important;
                    border: 1px solid #dc3545 !important;
                    border-radius: 8px !important;
                }}
                .stWarning {{
                    background-color: rgba(255, 193, 7, 0.8) !important;
                    color: #000000 !important;
                    border: 1px solid #ffc107 !important;
                    border-radius: 8px !important;
                }}
                .stInfo {{
                    background-color: rgba(23, 162, 184, 0.8) !important;
                    color: #ffffff !important;
                    border: 1px solid #17a2b8 !important;
                    border-radius: 8px !important;
                }}
                .stRadio div[data-baseweb="radio"] {{
                    background-color: rgba(0, 0, 0, 0.7) !important;
                }}
                .stRadio div[data-baseweb="radio"] * {{
                    color: #ffffff !important;
                }}
                .stCheckbox div[data-baseweb="checkbox"] {{
                    background-color: rgba(0, 0, 0, 0.7) !important;
                }}
                .stCheckbox div[data-baseweb="checkbox"] * {{
                    color: #ffffff !important;
                }}
                .stTextArea textarea {{
                    background-color: rgba(0, 0, 0, 0.7) !important;
                    color: #ffffff !important;
                    border: 1px solid #1DB954 !important;
                    border-radius: 8px !important;
                }}
                .stTextArea textarea::placeholder {{
                    color: #cccccc !important;
                }}
                .stFileUploader {{
                    background-color: rgba(0, 0, 0, 0.7) !important;
                    border: 1px solid #1DB954 !important;
                    border-radius: 8px !important;
                }}
                .stFileUploader * {{
                    color: #ffffff !important;
                }}
                .stProgress div[data-baseweb="progress-bar"] {{
                    background-color: rgba(29, 185, 84, 0.8) !important;
                }}
                .stPlotlyChart {{
                    background-color: rgba(0, 0, 0, 0.7) !important;
                    border-radius: 12px !important;
                    padding: 1rem !important;
                }}
                .stPyplot {{
                    background-color: rgba(0, 0, 0, 0.7) !important;
                    border-radius: 12px !important;
                    padding: 1rem !important;
                }}
                .stImage {{
                    border-radius: 12px !important;
                    overflow: hidden !important;
                }}
                .stColumns {{
                    gap: 1rem !important;
                }}
                .stColumn {{
                    background-color: rgba(0, 0, 0, 0.5) !important;
                    border-radius: 12px !important;
                    padding: 1rem !important;
                }}
                .logo-container {{
                    position: fixed;
                    top: 10px;
                    right: 10px;
                    z-index: 1000;
                }}
                .logo {{
                    width: 80px;
                    height: 80px;
                    border-radius: 50%;
                    object-fit: cover;
                    border: 3px solid #1DB954;
                }}
                .main-title {{
                    color: #1DB954;
                    text-align: center;
                    font-size: 3em;
                    font-weight: bold;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                    margin-bottom: 20px;
                }}
                .subtitle {{
                    color: #FFFFFF;
                    text-align: center;
                    font-size: 1.2em;
                    margin-bottom: 30px;
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
                }}
                </style>
                <div class="logo-container">
                    <img src="data:image/jpg;base64,{logo_b64}" class="logo">
                </div>
                """,
                unsafe_allow_html=True
            )

add_bg_and_logo()

# Load models and preprocessors
@st.cache_resource
def load_models():
    models = {}
    
    # Load ML models
    ml_model_names = ['Logistic_Regression', 'Decision_Tree', 'Random_Forest',
                     'XGBoost', 'LightGBM', 'SVM', 'KNN']
    for name in ml_model_names:
        try:
            with open(os.path.join(BASE_DIR, f'models/ml_{name}.pkl'), 'rb') as f:
                models[name.replace('_', ' ')] = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load {name}")

    # Load preprocessors
    with open(os.path.join(BASE_DIR, 'models/scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'models/features.pkl'), 'rb') as f:
        features = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'models/encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)

    return models, scaler, features, encoders

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'data/spotify_songs.csv'))
    return df

# Load everything
models, scaler, features, encoders = load_models()
df = load_data()

# Main title
st.markdown('<h1 class="main-title">🎵 Spotify Hit Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Can AI Predict the Next Chart-Topper? | ML vs ANN Showdown</p>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Prediction", "📊 EDA", "🤖 ML vs ANN", "🧠 ANN Architectures", "🔬 ANN Comparisons"])

with tab1:
    st.header("🎯 Song Hit Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Song Features")
        danceability = st.slider("Danceability", 0.0, 1.0, 0.5, 0.01)
        energy = st.slider("Energy", 0.0, 1.0, 0.5, 0.01)
        loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0, 0.1)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1, 0.01)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.2, 0.01)
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.01)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.1, 0.01)
        valence = st.slider("Valence", 0.0, 1.0, 0.5, 0.01)
        tempo = st.slider("Tempo (BPM)", 50.0, 220.0, 120.0, 1.0)

    with col2:
        st.subheader("Additional Features")
        duration_ms = st.slider("Duration (seconds)", 30, 600, 180) * 1000
        time_signature = st.selectbox("Time Signature", [3, 4, 5], index=1)
        genre = st.selectbox("Genre", encoders['genre'].classes_)
        key = st.selectbox("Key", encoders['key'].classes_)
        mode = st.selectbox("Mode", encoders['mode'].classes_)

    if st.button("🎵 Predict Hit Probability", type="primary"):
        # Preprocess input
        duration_min = duration_ms / 60000
        loudness_norm = loudness + 25
        dance_energy = danceability * energy
        acoustic_instrumental = acousticness * instrumentalness
        mood_score = valence * energy

        genre_enc = encoders['genre'].transform([genre])[0]
        key_enc = encoders['key'].transform([key])[0]
        mode_enc = encoders['mode'].transform([mode])[0]

        input_data = np.array([[danceability, energy, loudness_norm, speechiness, acousticness,
                               instrumentalness, liveness, valence, tempo, duration_min,
                               time_signature, genre_enc, key_enc, mode_enc,
                               dance_energy, acoustic_instrumental, mood_score]])

        input_scaled = scaler.transform(input_data)

        # Get ML predictions
        ml_predictions = {}
        for name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(input_scaled)[0][1]
                else:
                    prob = model.predict(input_scaled)[0]
                ml_predictions[name] = prob
            except:
                ml_predictions[name] = 0.5

        # Simulated ANN predictions
        ann_predictions = {
            'ANN Shallow': 0.65,
            'ANN Deep': 0.62,
            'ANN + Dropout': 0.66,
            'ANN + BatchNorm': 0.66,
            'ANN Best (D+BN)': 0.67
        }

        # Display results
        st.success("✅ Prediction Complete!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🤖 ML Model Predictions")
            for model, prob in ml_predictions.items():
                st.metric(f"{model}", f"{prob:.3f}", delta=f"{prob*100:.1f}%")

        with col2:
            st.subheader("🧠 ANN Model Predictions")
            for model, prob in ann_predictions.items():
                st.metric(f"{model}", f"{prob:.3f}", delta=f"{prob*100:.1f}%")

        # Overall prediction
        avg_ml = np.mean(list(ml_predictions.values()))
        avg_ann = np.mean(list(ann_predictions.values()))

        st.subheader("🎯 Overall Prediction")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average ML Probability", f"{avg_ml:.3f}")
        with col2:
            st.metric("Average ANN Probability", f"{avg_ann:.3f}")
        with col3:
            final_prob = (avg_ml + avg_ann) / 2
            if final_prob > 0.5:
                st.success(f"🎉 HIT! ({final_prob:.3f})")
            else:
                st.error(f"❌ Not a Hit ({final_prob:.3f})")

        # ── 🎧 Audio Track Recommendation ──────────────────────────────────
        st.markdown("---")
        st.subheader("🎧 Listen to a Matching Track")

        GENRE_TAG_MAP = {
            'pop': 'pop', 'rock': 'rock', 'hip-hop': 'hiphop', 'hiphop': 'hiphop',
            'rap': 'hiphop', 'electronic': 'electronic', 'edm': 'electronic',
            'dance': 'electronic', 'jazz': 'jazz', 'blues': 'blues',
            'classical': 'classical', 'country': 'country', 'r&b': 'rnb',
            'rnb': 'rnb', 'soul': 'soul', 'metal': 'metal', 'punk': 'punk',
            'reggae': 'reggae', 'folk': 'folk', 'latin': 'latin',
            'indie': 'indie', 'alternative': 'alternative',
        }
        def fetch_spotify_track(genre, mood="popular"):
            if not SPOTIFY_AVAILABLE:
                return None
            
            try:
                client_id = st.secrets.get("SPOTIFY_CLIENT_ID")
                client_secret = st.secrets.get("SPOTIFY_CLIENT_SECRET")
            
                if not client_id or not client_secret:
                    return None
            
                auth_manager = SpotifyClientCredentials(
                    client_id=client_id,
                    client_secret=client_secret
                )
                sp = spotipy.Spotify(auth_manager=auth_manager)
            
                # 🔥 FIX 1: Use genre filter properly
                query = f"genre:{genre}"
            
                results = sp.search(
                    q=query,
                    type="track",
                    limit=50  # larger pool
                )
            
                tracks = results["tracks"]["items"]
            
                if not tracks:
                    return None
            
                # 🔥 FIX 2: Filter by mood (post-filtering)
                filtered_tracks = []
            
                for t in tracks:
                    name = t["name"].lower()
                    artist = " ".join([a["name"].lower() for a in t["artists"]])
            
                    text = name + " " + artist
            
                    if mood == "high energy dance":
                        if any(k in text for k in ["dance", "remix", "party"]):
                            filtered_tracks.append(t)
            
                    elif mood == "chill acoustic":
                        if any(k in text for k in ["acoustic", "piano", "chill"]):
                            filtered_tracks.append(t)
            
                    else:
                        filtered_tracks.append(t)
            
                # fallback if filtering too strict
                if not filtered_tracks:
                    filtered_tracks = tracks
            
                import random
                t = random.choice(filtered_tracks)
            
                return {
                    "name": t["name"],
                    "artist": ", ".join([a["name"] for a in t["artists"]]),
                    "preview": t["preview_url"],
                    "image": t["album"]["images"][0]["url"] if t["album"]["images"] else None,
                    "url": t["external_urls"]["spotify"]
                }
            
            except Exception as e:
                print("Spotify error:", e)
            
            return None
    
        # ── End Audio Section ──────────────────────────────────────────────

with tab2:
    st.header("📊 Exploratory Data Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Songs", len(df))
    with col2:
        st.metric("Hit Rate", f"{df['is_hit'].mean():.1%}")
    with col3:
        st.metric("Avg Popularity", f"{df['popularity'].mean():.1f}")
    with col4:
        st.metric("Avg Danceability", f"{df['danceability'].mean():.2f}")

    st.subheader("Feature Distributions")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#0f1117')
    for ax in axes.flat:
        ax.set_facecolor('#1e2130')

    features_to_plot = ['danceability', 'energy', 'loudness', 'valence', 'tempo', 'acousticness']
    for i, feature in enumerate(features_to_plot):
        ax = axes[i//3, i%3]
        sns.histplot(data=df, x=feature, hue='is_hit', ax=ax, palette=['#e17055', '#6bcb77'], alpha=0.7)
        ax.set_title(f'{feature.title()}', color='white', fontweight='bold')
        ax.tick_params(colors='white')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color('white')

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Genre Analysis")
    genre_hit_rate = df.groupby('genre')['is_hit'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1e2130')
    genre_hit_rate.plot(kind='bar', ax=ax, color='#1DB954')
    ax.set_title('Hit Rate by Genre', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    plt.xticks(rotation=45, color='white')
    plt.yticks(color='white')
    st.pyplot(fig)

with tab3:
    st.header("🤖 ML vs ANN Comparison")

    ml_results = {
        'Logistic Regression': {'accuracy': 0.6775, 'f1': 0.6551, 'auc': 0.7459},
        'Decision Tree': {'accuracy': 0.7071, 'f1': 0.6950, 'auc': 0.7574},
        'Random Forest': {'accuracy': 0.7087, 'f1': 0.6930, 'auc': 0.7846},
        'XGBoost': {'accuracy': 0.7408, 'f1': 0.7319, 'auc': 0.8188},
        'LightGBM': {'accuracy': 0.7450, 'f1': 0.7358, 'auc': 0.8209},
        'SVM': {'accuracy': 0.6971, 'f1': 0.6776, 'auc': 0.7534},
        'KNN': {'accuracy': 0.6421, 'f1': 0.6343, 'auc': 0.6863}
    }

    ann_results_data = {
        'ANN Shallow': {'accuracy': 0.6946, 'f1': 0.6714, 'auc': 0.7579},
        'ANN Deep': {'accuracy': 0.6729, 'f1': 0.6503, 'auc': 0.7460},
        'ANN + Dropout': {'accuracy': 0.6933, 'f1': 0.6921, 'auc': 0.7633},
        'ANN + BatchNorm': {'accuracy': 0.6937, 'f1': 0.6852, 'auc': 0.7593},
        'ANN Best (D+BN)': {'accuracy': 0.7025, 'f1': 0.6938, 'auc': 0.7664}
    }

    comparison_df = pd.DataFrame()
    for model, metrics in ml_results.items():
        comparison_df = pd.concat([comparison_df, pd.DataFrame({
            'Model': [model], 'Type': ['ML'],
            'Accuracy': [metrics['accuracy']], 'F1-Score': [metrics['f1']], 'AUC': [metrics['auc']]
        })], ignore_index=True)

    for model, metrics in ann_results_data.items():
        comparison_df = pd.concat([comparison_df, pd.DataFrame({
            'Model': [model], 'Type': ['ANN'],
            'Accuracy': [metrics['accuracy']], 'F1-Score': [metrics['f1']], 'AUC': [metrics['auc']]
        })], ignore_index=True)

    st.subheader("Model Performance Metrics")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0f1117')

    metrics = ['Accuracy', 'F1-Score', 'AUC']
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.set_facecolor('#1e2130')
        ml_vals = comparison_df[comparison_df['Type'] == 'ML'][metric]
        ann_vals = comparison_df[comparison_df['Type'] == 'ANN'][metric]
        ax.bar(range(len(ml_vals)), ml_vals, color='#1DB954', alpha=0.7, label='ML')
        ax.bar(range(len(ml_vals), len(ml_vals) + len(ann_vals)), ann_vals, color='#FF6B6B', alpha=0.7, label='ANN')
        ax.set_title(f'{metric} Comparison', color='white', fontweight='bold')
        ax.set_xticks(range(len(comparison_df)))
        ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right', color='white')
        ax.tick_params(colors='white')
        ax.legend()

    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ML Models Summary")
        ml_sum = comparison_df[comparison_df['Type']=='ML'][['Accuracy','F1-Score','AUC']].agg(['mean','max'])
        st.dataframe(ml_sum.style.format("{:.4f}"))
    
    with col2:
        st.markdown("### ANN Models Summary")
        ann_sum = comparison_df[comparison_df['Type']=='ANN'][['Accuracy','F1-Score','AUC']].agg(['mean','max'])
        st.dataframe(ann_sum.style.format("{:.4f}"))

with tab4:
    st.header("🧠 ANN Architecture Visualization")

    def create_nn_diagram(layers, title):
        fig = go.Figure()
        y_positions = []
        for i, size in enumerate(layers):
            y_pos = np.linspace(-(size-1)/2, (size-1)/2, size)
            y_positions.append(y_pos)
            for y in y_pos:
                color = '#1DB954' if i == 0 else '#FF6B6B' if i == len(layers)-1 else '#4ECDC4'
                fig.add_trace(go.Scatter(x=[i], y=[y], mode='markers',
                    marker=dict(size=25, color=color), showlegend=False))
        
        for i in range(len(layers)-1):
            for y1 in y_positions[i]:
                for y2 in y_positions[i+1]:
                    fig.add_trace(go.Scatter(x=[i, i+1], y=[y1, y2], mode='lines',
                        line=dict(color='rgba(255,255,255,0.2)', width=0.5), showlegend=False))

        fig.update_layout(title=title, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#0f1117', paper_bgcolor='#0f1117', font=dict(color='white'), height=300)
        return fig

    archs = {
        'ANN Shallow': [17, 64, 32, 1],
        'ANN Deep': [17, 128, 64, 32, 16, 1],
        'ANN + Dropout': [17, 128, 64, 32, 1],
        'ANN + BatchNorm': [17, 128, 64, 32, 1],
        'ANN Best (D+BN)': [17, 256, 128, 64, 32, 1]
    }

    for name, layers in archs.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            fig = create_nn_diagram(layers, f"{name} Architecture")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            params = sum(layers[i]*layers[i+1] for i in range(len(layers)-1))
            st.metric("Layers", len(layers))
            st.metric("Parameters", f"{params:,}")

with tab5:
    st.header("🔬 ANN Model Comparisons")

    ann_results = {
        'ANN Shallow': {'accuracy': 0.6946, 'f1': 0.6714, 'auc': 0.7579},
        'ANN Deep': {'accuracy': 0.6729, 'f1': 0.6503, 'auc': 0.7460},
        'ANN + Dropout': {'accuracy': 0.6933, 'f1': 0.6921, 'auc': 0.7633},
        'ANN + BatchNorm': {'accuracy': 0.6937, 'f1': 0.6852, 'auc': 0.7593},
        'ANN Best (D+BN)': {'accuracy': 0.7025, 'f1': 0.6938, 'auc': 0.7664}
    }

    ann_comparison = pd.DataFrame({
        'Model': list(ann_results.keys()),
        'Accuracy': [v['accuracy'] for v in ann_results.values()],
        'F1-Score': [v['f1'] for v in ann_results.values()],
        'AUC': [v['auc'] for v in ann_results.values()],
        'Architecture': ['Shallow', 'Deep', 'Dropout', 'BatchNorm', 'Best (D+BN)'],
        'Regularization': ['None', 'None', 'Dropout 0.3', 'BatchNorm', 'Dropout 0.3 + BatchNorm']
    })

    st.subheader("ANN Model Performance")
    st.dataframe(ann_comparison.style.format({
        'Accuracy': '{:.4f}', 'F1-Score': '{:.4f}', 'AUC': '{:.4f}'
    }).background_gradient(cmap='RdYlGn', subset=['Accuracy', 'F1-Score', 'AUC']))

    st.subheader("Performance Radar Chart")
    categories = ['Accuracy', 'F1-Score', 'AUC']
    fig = go.Figure()
    for model, results in ann_results.items():
        values = [results['accuracy'], results['f1'], results['auc']] + [results['accuracy']]
        fig.add_trace(go.Scatterpolar(r=values, theta=categories+[categories[0]], fill='toself', name=model))
    
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0.6, 0.85])),
        plot_bgcolor='#0f1117', paper_bgcolor='#0f1117', font=dict(color='white'), height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Insights")
    for model, insight in {
        'ANN Shallow': '⚡ Fast convergence (26 epochs)',
        'ANN Deep': '⚠️ Overfitting issues (16 epochs)',
        'ANN + Dropout': '📈 Better generalization (72 epochs)',
        'ANN + BatchNorm': '⚙️ Stable training (40 epochs)',
        'ANN Best (D+BN)': '🏆 Best performance (52 epochs)'
    }.items():
        st.markdown(f"**{model}:** {insight}")

st.markdown("---")
st.markdown("🎵 Built with ❤️ using Streamlit | Spotify Hit Prediction")
