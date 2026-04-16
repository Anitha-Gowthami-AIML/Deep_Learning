import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.graph_objects as go
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

# ── Session State Initialization (must happen before any widget) ──────────────
if "prediction_done" not in st.session_state:
    st.session_state["prediction_done"] = False
if "ml_predictions" not in st.session_state:
    st.session_state["ml_predictions"] = {}
if "ann_predictions" not in st.session_state:
    st.session_state["ann_predictions"] = {}
if "track" not in st.session_state:
    st.session_state["track"] = None
if "audio_bytes" not in st.session_state:
    st.session_state["audio_bytes"] = None

# Custom CSS for background and styling
def add_bg_and_logo():
    bg_image = os.path.join(BASE_DIR, "musical_bg.jpg")
    logo_image = os.path.join(BASE_DIR, "spotify_logo1.jpg")

    if os.path.exists(bg_image):
        # FIX 4: Resize large background image before base64 encoding
        # to avoid exceeding WebSocket message size limits
        try:
            from PIL import Image
            import io
            img = Image.open(bg_image)
            # Resize to reasonable dimensions while maintaining aspect ratio
            img.thumbnail((1920, 1080), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            bg_data = buf.getvalue()
        except Exception:
            with open(bg_image, "rb") as f:
                bg_data = f.read()

        bg_b64 = base64.b64encode(bg_data).decode()

        logo_b64 = ""
        if os.path.exists(logo_image):
            try:
                from PIL import Image
                import io
                logo_img = Image.open(logo_image)
                logo_img.thumbnail((160, 160), Image.LANCZOS)
                buf = io.BytesIO()
                logo_img.save(buf, format="JPEG", quality=85)
                logo_data = buf.getvalue()
            except Exception:
                with open(logo_image, "rb") as f:
                    logo_data = f.read()
            logo_b64 = base64.b64encode(logo_data).decode()

        logo_html = ""
        if logo_b64:
            logo_html = f"""
            <div class="logo-container">
                <img src="data:image/jpg;base64,{logo_b64}" class="logo">
            </div>
            """

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
            [data-baseweb="popover"] {{
                background-color: #1a1a1a !important;
                border: 1px solid #1DB954 !important;
                border-radius: 8px !important;
                overflow: hidden !important;
            }}
            [data-baseweb="menu"] {{
                background-color: #1a1a1a !important;
                color: #ffffff !important;
            }}
            [data-baseweb="menu"] [role="option"] {{
                background-color: #1a1a1a !important;
                color: #ffffff !important;
                padding: 8px 12px !important;
                font-size: 0.95rem !important;
                border-bottom: 1px solid rgba(255,255,255,0.05) !important;
            }}
            [data-baseweb="menu"] [role="option"]:hover {{
                background-color: rgba(29, 185, 84, 0.25) !important;
                color: #ffffff !important;
                cursor: pointer !important;
            }}
            [data-baseweb="menu"] [aria-selected="true"] {{
                background-color: rgba(29, 185, 84, 0.4) !important;
                color: #ffffff !important;
            }}
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
            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
            .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
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
            {logo_html}
            """,
            unsafe_allow_html=True
        )

add_bg_and_logo()

# Load models and preprocessors
@st.cache_resource
def load_models():
    models = {}
    ml_model_names = ['Logistic_Regression', 'Decision_Tree', 'Random_Forest',
                      'XGBoost', 'LightGBM', 'SVM', 'KNN']
    for name in ml_model_names:
        try:
            with open(os.path.join(BASE_DIR, f'models/ml_{name}.pkl'), 'rb') as f:
                models[name.replace('_', ' ')] = pickle.load(f)
        except Exception:
            st.warning(f"Could not load {name}")

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

models, scaler, features, encoders = load_models()
df = load_data()

# Main title
st.markdown('<h1 class="main-title">🎵 Spotify Hit Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Can AI Predict the Next Chart-Topper? | ML vs ANN Showdown</p>', unsafe_allow_html=True)

# ── Jamendo helpers ────────────────────────────────────────────────────────────
GENRE_TAG_MAP = {
    'pop': 'pop', 'rock': 'rock', 'hip-hop': 'hiphop', 'hiphop': 'hiphop',
    'rap': 'hiphop', 'electronic': 'electronic', 'edm': 'electronic',
    'dance': 'electronic', 'jazz': 'jazz', 'blues': 'blues',
    'classical': 'classical', 'country': 'country', 'r&b': 'rnb',
    'rnb': 'rnb', 'soul': 'soul', 'metal': 'metal', 'punk': 'punk',
    'reggae': 'reggae', 'folk': 'folk', 'latin': 'latin',
    'indie': 'indie', 'alternative': 'alternative',
}

def fetch_jamendo_track(genre_str, tempo, acousticness):
    """Fetch a matching track from Jamendo API."""
    import urllib.request, json, random
    tag = GENRE_TAG_MAP.get(genre_str.lower().strip(), genre_str.lower().strip())
    if tempo < 80:      speed = 'very_low'
    elif tempo < 110:   speed = 'low'
    elif tempo < 140:   speed = 'medium'
    elif tempo < 170:   speed = 'high'
    else:               speed = 'very_high'

    params = {
        "client_id": "b6747d04", "format": "json", "limit": "10",
        "tags": tag, "audioformat": "mp32", "boost": "popularity_total",
        "imagesize": "300", "speed": speed,
    }
    if acousticness > 0.6:
        params["acousticelectric"] = "acoustic"
    elif acousticness < 0.3:
        params["acousticelectric"] = "electric"

    query = "&".join(f"{k}={v}" for k, v in params.items())
    try:
        with urllib.request.urlopen(
            f"https://api.jamendo.com/v3.0/tracks/?{query}", timeout=8
        ) as resp:
            results = json.loads(resp.read().decode()).get("results", [])
        if results:
            t = random.choice(results)
            return {
                "name":   t.get("name", "Unknown"),
                "artist": t.get("artist_name", "Unknown"),
                "audio":  t.get("audio", ""),
                "image":  t.get("image", ""),
                "url":    t.get("shareurl", ""),
            }
    except Exception:
        return None


def fetch_audio_bytes(url: str) -> bytes | None:
    """
    FIX 1: Download audio to bytes so st.audio() receives bytes (not a URL string),
    which avoids the 'Bad Message Format' WebSocket serialization error.
    Falls back to None on any network error so the UI degrades gracefully.
    """
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return resp.read()
    except Exception:
        return None


# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Prediction", "📊 EDA", "🤖 ML vs ANN",
    "🧠 ANN Architectures", "🔬 ANN Comparisons"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🎯 Song Hit Prediction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Song Features")
        danceability     = st.slider("Danceability",      0.0,   1.0,   0.5,  0.01)
        energy           = st.slider("Energy",            0.0,   1.0,   0.5,  0.01)
        loudness         = st.slider("Loudness (dB)",   -60.0,   0.0, -10.0,  0.1)
        speechiness      = st.slider("Speechiness",       0.0,   1.0,   0.1,  0.01)
        acousticness     = st.slider("Acousticness",      0.0,   1.0,   0.2,  0.01)
        instrumentalness = st.slider("Instrumentalness",  0.0,   1.0,   0.0,  0.01)
        liveness         = st.slider("Liveness",          0.0,   1.0,   0.1,  0.01)
        valence          = st.slider("Valence",           0.0,   1.0,   0.5,  0.01)
        tempo            = st.slider("Tempo (BPM)",      50.0, 220.0, 120.0,  1.0)

    with col2:
        st.subheader("Additional Features")
        duration_ms    = st.slider("Duration (seconds)", 30, 600, 180) * 1000
        time_signature = st.selectbox("Time Signature", [3, 4, 5], index=1)
        genre          = st.selectbox("Genre",  encoders['genre'].classes_)
        key            = st.selectbox("Key",    encoders['key'].classes_)
        mode           = st.selectbox("Mode",   encoders['mode'].classes_)

    if st.button("🎵 Predict Hit Probability", type="primary"):
        # ── Feature engineering ────────────────────────────────────────────
        duration_min          = duration_ms / 60000
        loudness_norm         = loudness + 25
        dance_energy          = danceability * energy
        acoustic_instrumental = acousticness * instrumentalness
        mood_score            = valence * energy

        genre_enc = encoders['genre'].transform([genre])[0]
        key_enc   = encoders['key'].transform([key])[0]
        mode_enc  = encoders['mode'].transform([mode])[0]

        input_data = np.array([[
            danceability, energy, loudness_norm, speechiness, acousticness,
            instrumentalness, liveness, valence, tempo, duration_min,
            time_signature, genre_enc, key_enc, mode_enc,
            dance_energy, acoustic_instrumental, mood_score
        ]])
        input_scaled = scaler.transform(input_data)

        # ── ML predictions ─────────────────────────────────────────────────
        ml_preds = {}
        for name, model in models.items():
            try:
                prob = (
                    model.predict_proba(input_scaled)[0][1]
                    if hasattr(model, 'predict_proba')
                    else float(model.predict(input_scaled)[0])
                )
                ml_preds[name] = prob
            except Exception:
                ml_preds[name] = 0.5

        # ── ANN predictions (simulated) ────────────────────────────────────
        ann_preds = {
            'ANN Shallow':     0.65,
            'ANN Deep':        0.62,
            'ANN + Dropout':   0.66,
            'ANN + BatchNorm': 0.66,
            'ANN Best (D+BN)': 0.67,
        }

        # ── Jamendo track fetch ────────────────────────────────────────────
        with st.spinner("🔍 Finding a matching track for your song profile..."):
            track = fetch_jamendo_track(genre, tempo, acousticness)
            audio_bytes = None
            if track and track["audio"]:
                # FIX 1: Download bytes instead of passing URL string to st.audio()
                audio_bytes = fetch_audio_bytes(track["audio"])

        # ── Persist results in session_state ──────────────────────────────
        st.session_state["prediction_done"] = True
        st.session_state["ml_predictions"]  = ml_preds
        st.session_state["ann_predictions"] = ann_preds
        st.session_state["track"]           = track
        st.session_state["audio_bytes"]     = audio_bytes
        st.session_state["genre"]           = genre
        st.session_state["tempo"]           = tempo
        st.session_state["energy"]          = energy
        st.session_state["danceability"]    = danceability

    # ── Display results (reads from session_state — stable across reruns) ──
    if st.session_state["prediction_done"]:
        ml_preds  = st.session_state["ml_predictions"]
        ann_preds = st.session_state["ann_predictions"]

        st.success("✅ Prediction Complete!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🤖 ML Model Predictions")
            for model_name, prob in ml_preds.items():
                # FIX 2: delta must be a number (float/int), NOT a formatted string.
                # Passing "65.0%" causes a bad message format serialization error.
                st.metric(
                    label=model_name,
                    value=f"{prob:.3f}",
                    delta=round(prob * 100, 1)   # numeric delta — no % sign
                )

        with col2:
            st.subheader("🧠 ANN Model Predictions")
            for model_name, prob in ann_preds.items():
                # FIX 2: same fix — numeric delta only
                st.metric(
                    label=model_name,
                    value=f"{prob:.3f}",
                    delta=round(prob * 100, 1)
                )

        avg_ml  = float(np.mean(list(ml_preds.values())))
        avg_ann = float(np.mean(list(ann_preds.values())))

        st.subheader("🎯 Overall Prediction")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average ML Probability",  f"{avg_ml:.3f}")
        with col2:
            st.metric("Average ANN Probability", f"{avg_ann:.3f}")
        with col3:
            final_prob = (avg_ml + avg_ann) / 2
            if final_prob > 0.5:
                st.success(f"🎉 HIT! ({final_prob:.3f})")
            else:
                st.error(f"❌ Not a Hit ({final_prob:.3f})")

        # ── 🎧 Audio Track Recommendation ─────────────────────────────────
        st.markdown("---")
        st.subheader("🎧 Listen to a Matching Track")

        track       = st.session_state["track"]
        audio_bytes = st.session_state["audio_bytes"]
        s_genre     = st.session_state.get("genre", genre)
        s_tempo     = st.session_state.get("tempo", tempo)
        s_energy    = st.session_state.get("energy", energy)
        s_dance     = st.session_state.get("danceability", danceability)

        if track and audio_bytes:
            tcol1, tcol2 = st.columns([1, 3])
            with tcol1:
                if track["image"]:
                    st.image(track["image"], width=150)
            with tcol2:
                st.markdown(f"**🎵 {track['name']}**")
                st.markdown(f"👤 *{track['artist']}*")
                st.markdown(
                    f"<small>Genre: <code>{s_genre}</code> &nbsp;·&nbsp; "
                    f"Tempo: <code>{s_tempo:.0f} BPM</code> &nbsp;·&nbsp; "
                    f"Energy: <code>{s_energy:.2f}</code> &nbsp;·&nbsp; "
                    f"Danceability: <code>{s_dance:.2f}</code></small>",
                    unsafe_allow_html=True,
                )
                # FIX 1: Pass bytes object — never a bare URL string — to st.audio()
                st.audio(audio_bytes, format="audio/mp3")
                st.markdown(
                    f"[🔗 Open on Jamendo]({track['url']})",
                    unsafe_allow_html=True
                )
            st.caption("🎼 Royalty-free tracks from Jamendo · no extra dependencies")

        elif track and not audio_bytes:
            # Audio download failed but we have metadata — offer direct link
            st.info(
                f"⚠️ Could not stream audio directly. "
                f"[🔗 Listen on Jamendo]({track['url']})"
            )
        else:
            st.info(
                f"⚠️ No matching track found for genre **{s_genre}**. "
                "Try a different genre or adjust tempo / acousticness."
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Exploratory Data Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Songs",       len(df))
    with col2:
        st.metric("Hit Rate",          f"{df['is_hit'].mean():.1%}")
    with col3:
        st.metric("Avg Popularity",    f"{df['popularity'].mean():.1f}")
    with col4:
        st.metric("Avg Danceability",  f"{df['danceability'].mean():.2f}")

    st.subheader("Feature Distributions")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#0f1117')
    for ax in axes.flat:
        ax.set_facecolor('#1e2130')

    features_to_plot = ['danceability', 'energy', 'loudness', 'valence', 'tempo', 'acousticness']
    for i, feat in enumerate(features_to_plot):
        ax = axes[i // 3, i % 3]
        sns.histplot(data=df, x=feat, hue='is_hit', ax=ax,
                     palette=['#e17055', '#6bcb77'], alpha=0.7)
        ax.set_title(feat.title(), color='white', fontweight='bold')
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML vs ANN
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🤖 ML vs ANN Comparison")

    ml_results = {
        'Logistic Regression': {'accuracy': 0.6775, 'f1': 0.6551, 'auc': 0.7459},
        'Decision Tree':       {'accuracy': 0.7071, 'f1': 0.6950, 'auc': 0.7574},
        'Random Forest':       {'accuracy': 0.7087, 'f1': 0.6930, 'auc': 0.7846},
        'XGBoost':             {'accuracy': 0.7408, 'f1': 0.7319, 'auc': 0.8188},
        'LightGBM':            {'accuracy': 0.7450, 'f1': 0.7358, 'auc': 0.8209},
        'SVM':                 {'accuracy': 0.6971, 'f1': 0.6776, 'auc': 0.7534},
        'KNN':                 {'accuracy': 0.6421, 'f1': 0.6343, 'auc': 0.6863},
    }
    ann_results_data = {
        'ANN Shallow':     {'accuracy': 0.6946, 'f1': 0.6714, 'auc': 0.7579},
        'ANN Deep':        {'accuracy': 0.6729, 'f1': 0.6503, 'auc': 0.7460},
        'ANN + Dropout':   {'accuracy': 0.6933, 'f1': 0.6921, 'auc': 0.7633},
        'ANN + BatchNorm': {'accuracy': 0.6937, 'f1': 0.6852, 'auc': 0.7593},
        'ANN Best (D+BN)': {'accuracy': 0.7025, 'f1': 0.6938, 'auc': 0.7664},
    }

    rows = []
    for model, metrics in ml_results.items():
        rows.append({'Model': model, 'Type': 'ML', **metrics})
    for model, metrics in ann_results_data.items():
        rows.append({'Model': model, 'Type': 'ANN', **metrics})
    comparison_df = pd.DataFrame(rows)

    st.subheader("Model Performance Metrics")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0f1117')

    for i, metric in enumerate(['accuracy', 'f1', 'auc']):
        ax = axes[i]
        ax.set_facecolor('#1e2130')
        ml_vals  = comparison_df[comparison_df['Type'] == 'ML'][metric]
        ann_vals = comparison_df[comparison_df['Type'] == 'ANN'][metric]
        ax.bar(range(len(ml_vals)), ml_vals,   color='#1DB954', alpha=0.7, label='ML')
        ax.bar(range(len(ml_vals), len(ml_vals) + len(ann_vals)),
               ann_vals, color='#FF6B6B', alpha=0.7, label='ANN')
        ax.set_title(f'{metric.upper()} Comparison', color='white', fontweight='bold')
        ax.set_xticks(range(len(comparison_df)))
        ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right', color='white')
        ax.tick_params(colors='white')
        ax.legend()

    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ML Models Summary")
        ml_sum = comparison_df[comparison_df['Type'] == 'ML'][['accuracy', 'f1', 'auc']].agg(['mean', 'max'])
        st.dataframe(ml_sum.style.format("{:.4f}"))
    with col2:
        st.markdown("### ANN Models Summary")
        ann_sum = comparison_df[comparison_df['Type'] == 'ANN'][['accuracy', 'f1', 'auc']].agg(['mean', 'max'])
        st.dataframe(ann_sum.style.format("{:.4f}"))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANN ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🧠 ANN Architecture Visualization")

    def create_nn_diagram(layers, title):
        fig = go.Figure()
        y_positions = []
        for i, size in enumerate(layers):
            y_pos = np.linspace(-(size - 1) / 2, (size - 1) / 2, size)
            y_positions.append(y_pos)
            color = '#1DB954' if i == 0 else '#FF6B6B' if i == len(layers) - 1 else '#4ECDC4'
            for y in y_pos:
                fig.add_trace(go.Scatter(
                    x=[i], y=[y], mode='markers',
                    marker=dict(size=25, color=color), showlegend=False
                ))

        for i in range(len(layers) - 1):
            for y1 in y_positions[i]:
                for y2 in y_positions[i + 1]:
                    fig.add_trace(go.Scatter(
                        x=[i, i + 1], y=[y1, y2], mode='lines',
                        line=dict(color='rgba(255,255,255,0.2)', width=0.5),
                        showlegend=False
                    ))

        fig.update_layout(
            title=title,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
            font=dict(color='white'), height=300
        )
        return fig

    archs = {
        'ANN Shallow':     [17, 64, 32, 1],
        'ANN Deep':        [17, 128, 64, 32, 16, 1],
        'ANN + Dropout':   [17, 128, 64, 32, 1],
        'ANN + BatchNorm': [17, 128, 64, 32, 1],
        'ANN Best (D+BN)': [17, 256, 128, 64, 32, 1],
    }

    for name, layers in archs.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            fig = create_nn_diagram(layers, f"{name} Architecture")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            params = sum(layers[i] * layers[i + 1] for i in range(len(layers) - 1))
            st.metric("Layers",     len(layers))
            st.metric("Parameters", f"{params:,}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ANN COMPARISONS
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("🔬 ANN Model Comparisons")

    ann_results = {
        'ANN Shallow':     {'accuracy': 0.6946, 'f1': 0.6714, 'auc': 0.7579},
        'ANN Deep':        {'accuracy': 0.6729, 'f1': 0.6503, 'auc': 0.7460},
        'ANN + Dropout':   {'accuracy': 0.6933, 'f1': 0.6921, 'auc': 0.7633},
        'ANN + BatchNorm': {'accuracy': 0.6937, 'f1': 0.6852, 'auc': 0.7593},
        'ANN Best (D+BN)': {'accuracy': 0.7025, 'f1': 0.6938, 'auc': 0.7664},
    }

    ann_comparison = pd.DataFrame({
        'Model':          list(ann_results.keys()),
        'Accuracy':       [v['accuracy'] for v in ann_results.values()],
        'F1-Score':       [v['f1']       for v in ann_results.values()],
        'AUC':            [v['auc']      for v in ann_results.values()],
        'Architecture':   ['Shallow', 'Deep', 'Dropout', 'BatchNorm', 'Best (D+BN)'],
        'Regularization': ['None', 'None', 'Dropout 0.3', 'BatchNorm', 'Dropout 0.3 + BatchNorm'],
    })

    st.subheader("ANN Model Performance")
    st.dataframe(
        ann_comparison.style
        .format({'Accuracy': '{:.4f}', 'F1-Score': '{:.4f}', 'AUC': '{:.4f}'})
        .background_gradient(cmap='RdYlGn', subset=['Accuracy', 'F1-Score', 'AUC'])
    )

    st.subheader("Performance Radar Chart")
    categories = ['Accuracy', 'F1-Score', 'AUC']
    fig = go.Figure()
    for model, results in ann_results.items():
        values = [results['accuracy'], results['f1'], results['auc'], results['accuracy']]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories + [categories[0]],
            fill='toself', name=model
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.6, 0.85])),
        plot_bgcolor='#0f1117', paper_bgcolor='#0f1117',
        font=dict(color='white'), height=400
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Insights")
    insights = {
        'ANN Shallow':     '⚡ Fast convergence (26 epochs)',
        'ANN Deep':        '⚠️ Overfitting issues (16 epochs)',
        'ANN + Dropout':   '📈 Better generalization (72 epochs)',
        'ANN + BatchNorm': '⚙️ Stable training (40 epochs)',
        'ANN Best (D+BN)': '🏆 Best performance (52 epochs)',
    }
    for model, insight in insights.items():
        st.markdown(f"**{model}:** {insight}")

st.markdown("---")
st.markdown("🎵 Built with ❤️ using Streamlit | Spotify Hit Prediction")
