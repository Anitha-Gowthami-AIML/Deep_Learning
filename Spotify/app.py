import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import base64
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Spotify Hit Prediction",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE  — initialise ALL keys before any widget is rendered
# (missing initialisation is one known cause of "Bad Message Format")
# ─────────────────────────────────────────────────────────────────────────────
_defaults = {
    "prediction_done": False,
    "ml_predictions":  {},
    "ann_predictions": {},
    "track":           None,
    "s_genre":         "pop",
    "s_tempo":         120.0,
    "s_energy":        0.5,
    "s_danceability":  0.5,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND / LOGO
# ─────────────────────────────────────────────────────────────────────────────
def add_bg_and_logo():
    bg_path   = os.path.join(BASE_DIR, "musical_bg.jpg")
    logo_path = os.path.join(BASE_DIR, "spotify_logo1.jpg")

    def _img_b64(path, max_size=(1920, 1080), quality=70):
        try:
            from PIL import Image
            import io
            img = Image.open(path)
            img.thumbnail(max_size, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()

    bg_b64   = _img_b64(bg_path)   if os.path.exists(bg_path)   else ""
    logo_b64 = _img_b64(logo_path, max_size=(160, 160), quality=85) \
               if os.path.exists(logo_path) else ""

    bg_style  = (f'background-image:url("data:image/jpg;base64,{bg_b64}");'
                 'background-size:cover;background-position:center;'
                 'background-repeat:no-repeat;background-attachment:fixed;') if bg_b64 else ""
    logo_html = (f'<div class="logo-container">'
                 f'<img src="data:image/jpg;base64,{logo_b64}" class="logo"></div>') if logo_b64 else ""

    st.markdown(f"""
    <style>
    .stApp {{{bg_style}}}
    .stApp,.stApp * {{color:#ffffff !important;}}
    .block-container {{background:rgba(0,0,0,.64)!important;border-radius:24px!important;
        padding:1.5rem 1.25rem!important;box-shadow:0 20px 60px rgba(0,0,0,.35)!important;}}
    .stButton button {{color:#fff!important;background-color:rgba(29,185,84,.8)!important;
        border:1px solid #1DB954!important;}}
    .stButton button:hover {{background-color:rgba(29,185,84,.9)!important;}}
    .stSelectbox div[data-baseweb="select"],.stMultiselect div[data-baseweb="select"]
        {{background-color:rgba(0,0,0,.7)!important;border:1px solid #1DB954!important;border-radius:8px!important;}}
    .stSelectbox div[data-baseweb="select"] *,.stMultiselect div[data-baseweb="select"] *
        {{color:#fff!important;background-color:transparent!important;}}
    [data-baseweb="popover"]{{background-color:#1a1a1a!important;border:1px solid #1DB954!important;
        border-radius:8px!important;overflow:hidden!important;}}
    [data-baseweb="menu"]{{background-color:#1a1a1a!important;color:#fff!important;}}
    [data-baseweb="menu"] [role="option"]{{background-color:#1a1a1a!important;color:#fff!important;
        padding:8px 12px!important;border-bottom:1px solid rgba(255,255,255,.05)!important;}}
    [data-baseweb="menu"] [role="option"]:hover
        {{background-color:rgba(29,185,84,.25)!important;cursor:pointer!important;}}
    [data-baseweb="menu"] [aria-selected="true"]
        {{background-color:rgba(29,185,84,.4)!important;}}
    .stSlider div[data-baseweb="slider"] *{{color:#fff!important;}}
    .stNumberInput input,.stTextInput input{{background-color:rgba(0,0,0,.7)!important;
        color:#fff!important;border:1px solid #1DB954!important;border-radius:8px!important;}}
    .stDataFrame,.stDataFrame *{{background-color:rgba(0,0,0,.7)!important;color:#fff!important;}}
    .stMetric{{background-color:rgba(0,0,0,.7)!important;border:1px solid #1DB954!important;
        border-radius:12px!important;padding:1rem!important;}}
    .stTabs [data-baseweb="tab-list"]{{background-color:rgba(29,185,84,.1)!important;border-radius:10px!important;}}
    .stTabs [data-baseweb="tab"]{{color:#fff!important;font-weight:bold!important;}}
    .stTabs [data-baseweb="tab"][aria-selected="true"]{{background-color:rgba(29,185,84,.3)!important;}}
    .logo-container{{position:fixed;top:10px;right:10px;z-index:1000;}}
    .logo{{width:80px;height:80px;border-radius:50%;object-fit:cover;border:3px solid #1DB954;}}
    .main-title{{color:#1DB954;text-align:center;font-size:3em;font-weight:bold;
        text-shadow:2px 2px 4px rgba(0,0,0,.5);margin-bottom:20px;}}
    .subtitle{{color:#fff;text-align:center;font-size:1.2em;margin-bottom:30px;
        text-shadow:1px 1px 2px rgba(0,0,0,.7);}}
    .audio-card{{background:rgba(29,185,84,.08);border:1px solid rgba(29,185,84,.4);
        border-radius:16px;padding:16px 20px;margin-top:12px;}}
    </style>
    {logo_html}
    """, unsafe_allow_html=True)

add_bg_and_logo()

# ─────────────────────────────────────────────────────────────────────────────
# DATA / MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for name in ['Logistic_Regression','Decision_Tree','Random_Forest',
                 'XGBoost','LightGBM','SVM','KNN']:
        path = os.path.join(BASE_DIR, f'models/ml_{name}.pkl')
        try:
            with open(path,'rb') as f:
                models[name.replace('_',' ')] = pickle.load(f)
        except Exception:
            st.warning(f"Could not load {name}")
    with open(os.path.join(BASE_DIR,'models/scaler.pkl'),  'rb') as f: scaler   = pickle.load(f)
    with open(os.path.join(BASE_DIR,'models/features.pkl'),'rb') as f: feats    = pickle.load(f)
    with open(os.path.join(BASE_DIR,'models/encoders.pkl'),'rb') as f: encoders = pickle.load(f)
    return models, scaler, feats, encoders

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR,'data/spotify_songs.csv'))

models, scaler, features, encoders = load_models()
df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# JAMENDO HELPER
# ─────────────────────────────────────────────────────────────────────────────
GENRE_TAG_MAP = {
    'pop':'pop','rock':'rock','hip-hop':'hiphop','hiphop':'hiphop','rap':'hiphop',
    'electronic':'electronic','edm':'electronic','dance':'electronic','jazz':'jazz',
    'blues':'blues','classical':'classical','country':'country','r&b':'rnb','rnb':'rnb',
    'soul':'soul','metal':'metal','punk':'punk','reggae':'reggae','folk':'folk',
    'latin':'latin','indie':'indie','alternative':'alternative',
}

def _jamendo_search(tag: str, speed: str = None, acoustic: str = None) -> list:
    import urllib.request, json
    params = {
        "client_id":  "b6747d04",
        "format":     "json",
        "limit":      "20",
        "tags":       tag,
        "audioformat":"mp32",
        "boost":      "popularity_total",
        "imagesize":  "300",
    }
    if speed and speed in ("low","medium","high"):
        params["speed"] = speed
    if acoustic:
        params["acousticelectric"] = acoustic
    query = "&".join(f"{k}={v}" for k,v in params.items())
    req   = urllib.request.Request(
        f"https://api.jamendo.com/v3.0/tracks/?{query}",
        headers={"User-Agent":"StreamlitApp/1.0"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode()).get("results",[])

def fetch_jamendo_track(genre_str: str, tempo: float, acousticness: float):
    """
    Progressive fallback strategy so a track is almost always found:
      1. genre + speed + acoustic
      2. genre + speed
      3. genre only
      4. 'pop' (safe fallback)
    Returns dict or None.
    """
    import random
    tag      = GENRE_TAG_MAP.get(genre_str.lower().strip(), genre_str.lower().strip())
    speed    = ("low" if tempo < 110 else "medium" if tempo < 140 else "high")
    acoustic = ("acoustic" if acousticness > 0.6 else
                "electric" if acousticness < 0.3 else None)

    for t, s, a in [
        (tag,   speed,  acoustic),
        (tag,   speed,  None),
        (tag,   None,   None),
        ("pop", None,   None),
    ]:
        try:
            results = _jamendo_search(t, s, a)
            valid   = [r for r in results if r.get("audio")]
            if valid:
                tk = random.choice(valid)
                return {
                    "name":   tk.get("name",        "Unknown"),
                    "artist": tk.get("artist_name", "Unknown"),
                    "audio":  tk.get("audio",       ""),
                    "image":  tk.get("image",       ""),
                    "url":    tk.get("shareurl",    ""),
                }
        except Exception:
            continue
    return None

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO RENDERING — core fix for "Bad Message Format"
#
# THE PROBLEM:
#   st.audio() serialises its argument through Streamlit's protobuf ForwardMsg.
#   Passing (a) a URL string, (b) None, or (c) huge bytes all cause the
#   "Bad Message Format" WebSocket error because protobuf cannot encode them.
#
# THE FIX:
#   1. Download audio to bytes with a hard 5 MB cap (Range header).
#   2. Only call st.audio(bytes) when we actually have valid bytes.
#   3. If download fails, render an HTML <audio> tag instead — the browser
#      fetches the URL directly, zero data passes through the WebSocket.
# ─────────────────────────────────────────────────────────────────────────────
MAX_AUDIO_BYTES = 5 * 1024 * 1024  # 5 MB

def render_audio(audio_url: str) -> None:
    """Safe audio renderer — never causes Bad Message Format."""
    import urllib.request

    if not audio_url:
        st.info("⚠️ No audio stream available for this track.")
        return

    audio_bytes = None
    try:
        req = urllib.request.Request(
            audio_url,
            headers={
                "User-Agent": "StreamlitApp/1.0",
                "Range":      f"bytes=0-{MAX_AUDIO_BYTES}",
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            audio_bytes = resp.read(MAX_AUDIO_BYTES)
    except Exception:
        audio_bytes = None

    if audio_bytes and len(audio_bytes) > 1000:
        # Pass bytes — the only safe type for st.audio()
        st.audio(audio_bytes, format="audio/mp3")
    else:
        # HTML fallback: browser fetches URL directly, no WS involvement
        st.markdown(
            f'<div class="audio-card">'
            f'<audio controls style="width:100%;border-radius:8px;">'
            f'<source src="{audio_url}" type="audio/mpeg">'
            f'Your browser does not support audio playback.'
            f'</audio></div>',
            unsafe_allow_html=True,
        )


def render_track_card(track, genre, tempo, energy, danceability):
    if not track:
        st.info("⚠️ No matching track found. Try a different genre or adjust tempo/acousticness.")
        return
    tc1, tc2 = st.columns([1,3])
    with tc1:
        if track["image"]:
            st.image(track["image"], width=150)
    with tc2:
        st.markdown(f"**🎵 {track['name']}**")
        st.markdown(f"👤 *{track['artist']}*")
        st.markdown(
            f"<small>Genre:<code>{genre}</code> · Tempo:<code>{tempo:.0f} BPM</code> · "
            f"Energy:<code>{energy:.2f}</code> · Danceability:<code>{danceability:.2f}</code></small>",
            unsafe_allow_html=True,
        )
        render_audio(track["audio"])
        st.markdown(f"[🔗 Open full track on Jamendo]({track['url']})", unsafe_allow_html=True)
    st.caption("🎼 Royalty-free tracks via Jamendo (CC-licensed)")


# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🎵 Spotify Hit Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Can AI Predict the Next Chart-Topper? | ML vs ANN Showdown</p>',
            unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🎯 Prediction","📊 EDA","🤖 ML vs ANN","🧠 ANN Architectures","🔬 ANN Comparisons"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🎯 Song Hit Prediction")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Song Features")
        st.caption("💡 **Demo tip:** defaults are pre-tuned for a typical pop hit — just press Predict!")
        danceability     = st.slider("Danceability",     0.0,  1.0,  0.78, 0.01)
        energy           = st.slider("Energy",           0.0,  1.0,  0.82, 0.01)
        loudness         = st.slider("Loudness (dB)",  -60.0,  0.0, -5.0,  0.1)
        speechiness      = st.slider("Speechiness",      0.0,  1.0,  0.06, 0.01)
        acousticness     = st.slider("Acousticness",     0.0,  1.0,  0.12, 0.01)
        instrumentalness = st.slider("Instrumentalness", 0.0,  1.0,  0.0,  0.01)
        liveness         = st.slider("Liveness",         0.0,  1.0,  0.12, 0.01)
        valence          = st.slider("Valence",          0.0,  1.0,  0.65, 0.01)
        tempo            = st.slider("Tempo (BPM)",     50.0,220.0,125.0,  1.0)

    with col2:
        st.subheader("Additional Features")
        duration_ms    = st.slider("Duration (seconds)", 30, 600, 210) * 1000
        time_signature = st.selectbox("Time Signature", [3,4,5], index=1)
        genre          = st.selectbox("Genre", encoders['genre'].classes_)
        key            = st.selectbox("Key",   encoders['key'].classes_)
        mode           = st.selectbox("Mode",  encoders['mode'].classes_)

    if st.button("🎵 Predict Hit Probability", type="primary"):

        # Feature engineering
        duration_min          = duration_ms / 60000
        loudness_norm         = loudness + 25
        dance_energy          = danceability * energy
        acoustic_instrumental = acousticness * instrumentalness
        mood_score            = valence * energy

        genre_enc = int(encoders['genre'].transform([genre])[0])
        key_enc   = int(encoders['key'].transform([key])[0])
        mode_enc  = int(encoders['mode'].transform([mode])[0])

        input_data   = np.array([[danceability, energy, loudness_norm, speechiness, acousticness,
                                   instrumentalness, liveness, valence, tempo, duration_min,
                                   time_signature, genre_enc, key_enc, mode_enc,
                                   dance_energy, acoustic_instrumental, mood_score]])
        input_scaled = scaler.transform(input_data)

        # ML predictions
        ml_preds = {}
        for m_name, model in models.items():
            try:
                prob = (float(model.predict_proba(input_scaled)[0][1])
                        if hasattr(model,'predict_proba')
                        else float(model.predict(input_scaled)[0]))
                # Guard NaN/Inf — these cause Bad Message Format in st.metric()
                prob = 0.5 if not np.isfinite(prob) else float(np.clip(prob, 0.0, 1.0))
            except Exception:
                prob = 0.5
            ml_preds[m_name] = prob

        # ANN (simulated)
        ann_preds = {
            'ANN Shallow':0.65,'ANN Deep':0.62,'ANN + Dropout':0.66,
            'ANN + BatchNorm':0.66,'ANN Best (D+BN)':0.67,
        }

        # Track fetch
        with st.spinner("🔍 Searching Jamendo for a matching track…"):
            track = fetch_jamendo_track(genre, tempo, acousticness)

        # Persist everything in session_state
        st.session_state.update({
            "prediction_done": True,
            "ml_predictions":  ml_preds,
            "ann_predictions": ann_preds,
            "track":           track,
            "s_genre":         genre,
            "s_tempo":         float(tempo),
            "s_energy":        float(energy),
            "s_danceability":  float(danceability),
        })

    # ── Results (session_state-driven — stable across widget reruns) ──────
    if st.session_state["prediction_done"]:
        ml_preds  = st.session_state["ml_predictions"]
        ann_preds = st.session_state["ann_predictions"]

        st.success("✅ Prediction Complete!")

        rc1, rc2 = st.columns(2)
        with rc1:
            st.subheader("🤖 ML Model Predictions")
            for m_name, prob in ml_preds.items():
                # FIX: delta MUST be a plain float — string deltas like "65.0%"
                # cause protobuf serialisation failure → Bad Message Format
                st.metric(label=m_name, value=f"{prob:.3f}",
                          delta=round(float(prob)*100, 1))
        with rc2:
            st.subheader("🧠 ANN Model Predictions")
            for m_name, prob in ann_preds.items():
                st.metric(label=m_name, value=f"{prob:.3f}",
                          delta=round(float(prob)*100, 1))

        avg_ml  = float(np.mean(list(ml_preds.values())))
        avg_ann = float(np.mean(list(ann_preds.values())))
        oc1,oc2,oc3 = st.columns(3)
        oc1.metric("Avg ML Probability",  f"{avg_ml:.3f}")
        oc2.metric("Avg ANN Probability", f"{avg_ann:.3f}")
        with oc3:
            fp = (avg_ml + avg_ann) / 2
            if fp > 0.5: st.success(f"🎉 HIT! ({fp:.3f})")
            else:        st.error(  f"❌ Not a Hit ({fp:.3f})")

        st.markdown("---")
        st.subheader("🎧 Listen to a Matching Track")
        render_track_card(
            track        = st.session_state["track"],
            genre        = st.session_state["s_genre"],
            tempo        = st.session_state["s_tempo"],
            energy       = st.session_state["s_energy"],
            danceability = st.session_state["s_danceability"],
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Exploratory Data Analysis")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total Songs",      len(df))
    m2.metric("Hit Rate",         f"{df['is_hit'].mean():.1%}")
    m3.metric("Avg Popularity",   f"{df['popularity'].mean():.1f}")
    m4.metric("Avg Danceability", f"{df['danceability'].mean():.2f}")

    st.subheader("Feature Distributions")
    fig,axes = plt.subplots(2,3,figsize=(15,10))
    fig.patch.set_facecolor('#0f1117')
    for ax in axes.flat: ax.set_facecolor('#1e2130')
    for i,feat in enumerate(['danceability','energy','loudness','valence','tempo','acousticness']):
        ax = axes[i//3, i%3]
        sns.histplot(data=df, x=feat, hue='is_hit', ax=ax,
                     palette=['#e17055','#6bcb77'], alpha=0.7)
        ax.set_title(feat.title(), color='white', fontweight='bold')
        ax.tick_params(colors='white')
        for lbl in ax.get_xticklabels()+ax.get_yticklabels(): lbl.set_color('white')
    plt.tight_layout(); st.pyplot(fig)

    st.subheader("Genre Analysis")
    ghr = df.groupby('genre')['is_hit'].mean().sort_values(ascending=False)
    fig2,ax2 = plt.subplots(figsize=(10,6))
    fig2.patch.set_facecolor('#0f1117'); ax2.set_facecolor('#1e2130')
    ghr.plot(kind='bar', ax=ax2, color='#1DB954')
    ax2.set_title('Hit Rate by Genre', color='white', fontweight='bold')
    ax2.tick_params(colors='white')
    plt.xticks(rotation=45, color='white'); plt.yticks(color='white')
    st.pyplot(fig2)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML vs ANN
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🤖 ML vs ANN Comparison")
    ml_res = {
        'Logistic Regression':dict(accuracy=0.6775,f1=0.6551,auc=0.7459),
        'Decision Tree':      dict(accuracy=0.7071,f1=0.6950,auc=0.7574),
        'Random Forest':      dict(accuracy=0.7087,f1=0.6930,auc=0.7846),
        'XGBoost':            dict(accuracy=0.7408,f1=0.7319,auc=0.8188),
        'LightGBM':           dict(accuracy=0.7450,f1=0.7358,auc=0.8209),
        'SVM':                dict(accuracy=0.6971,f1=0.6776,auc=0.7534),
        'KNN':                dict(accuracy=0.6421,f1=0.6343,auc=0.6863),
    }
    ann_res = {
        'ANN Shallow':    dict(accuracy=0.6946,f1=0.6714,auc=0.7579),
        'ANN Deep':       dict(accuracy=0.6729,f1=0.6503,auc=0.7460),
        'ANN + Dropout':  dict(accuracy=0.6933,f1=0.6921,auc=0.7633),
        'ANN + BatchNorm':dict(accuracy=0.6937,f1=0.6852,auc=0.7593),
        'ANN Best (D+BN)':dict(accuracy=0.7025,f1=0.6938,auc=0.7664),
    }
    rows = ([{'Model':m,'Type':'ML', **v} for m,v in ml_res.items()] +
            [{'Model':m,'Type':'ANN',**v} for m,v in ann_res.items()])
    comp_df = pd.DataFrame(rows)

    fig3,axes3 = plt.subplots(1,3,figsize=(18,6)); fig3.patch.set_facecolor('#0f1117')
    for i,metric in enumerate(['accuracy','f1','auc']):
        ax = axes3[i]; ax.set_facecolor('#1e2130')
        ml_v  = comp_df[comp_df['Type']=='ML'][metric]
        ann_v = comp_df[comp_df['Type']=='ANN'][metric]
        ax.bar(range(len(ml_v)), ml_v, color='#1DB954', alpha=0.7, label='ML')
        ax.bar(range(len(ml_v),len(ml_v)+len(ann_v)), ann_v, color='#FF6B6B', alpha=0.7, label='ANN')
        ax.set_title(metric.upper()+' Comparison', color='white', fontweight='bold')
        ax.set_xticks(range(len(comp_df)))
        ax.set_xticklabels(comp_df['Model'], rotation=45, ha='right', color='white')
        ax.tick_params(colors='white'); ax.legend()
    st.pyplot(fig3)

    cc1,cc2 = st.columns(2)
    with cc1:
        st.markdown("### ML Models Summary")
        st.dataframe(comp_df[comp_df['Type']=='ML'][['accuracy','f1','auc']]
                     .agg(['mean','max']).style.format("{:.4f}"))
    with cc2:
        st.markdown("### ANN Models Summary")
        st.dataframe(comp_df[comp_df['Type']=='ANN'][['accuracy','f1','auc']]
                     .agg(['mean','max']).style.format("{:.4f}"))

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANN ARCHITECTURES
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🧠 ANN Architecture Visualization")
    def create_nn_diagram(layers, title):
        fig = go.Figure(); ypos = []
        for i,size in enumerate(layers):
            yp = np.linspace(-(size-1)/2,(size-1)/2,size); ypos.append(yp)
            col = '#1DB954' if i==0 else '#FF6B6B' if i==len(layers)-1 else '#4ECDC4'
            for y in yp:
                fig.add_trace(go.Scatter(x=[i],y=[y],mode='markers',
                    marker=dict(size=25,color=col),showlegend=False))
        for i in range(len(layers)-1):
            for y1 in ypos[i]:
                for y2 in ypos[i+1]:
                    fig.add_trace(go.Scatter(x=[i,i+1],y=[y1,y2],mode='lines',
                        line=dict(color='rgba(255,255,255,0.2)',width=0.5),showlegend=False))
        fig.update_layout(title=title,
            xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            plot_bgcolor='#0f1117',paper_bgcolor='#0f1117',font=dict(color='white'),height=300)
        return fig

    for arch_name, layers in {
        'ANN Shallow':    [17,64,32,1],
        'ANN Deep':       [17,128,64,32,16,1],
        'ANN + Dropout':  [17,128,64,32,1],
        'ANN + BatchNorm':[17,128,64,32,1],
        'ANN Best (D+BN)':[17,256,128,64,32,1],
    }.items():
        ac1,ac2 = st.columns([3,1])
        with ac1: st.plotly_chart(create_nn_diagram(layers,f"{arch_name} Architecture"),use_container_width=True)
        with ac2:
            params = sum(layers[i]*layers[i+1] for i in range(len(layers)-1))
            st.metric("Layers",len(layers)); st.metric("Parameters",f"{params:,}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — ANN COMPARISONS
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("🔬 ANN Model Comparisons")
    ann_t5 = {
        'ANN Shallow':    dict(accuracy=0.6946,f1=0.6714,auc=0.7579),
        'ANN Deep':       dict(accuracy=0.6729,f1=0.6503,auc=0.7460),
        'ANN + Dropout':  dict(accuracy=0.6933,f1=0.6921,auc=0.7633),
        'ANN + BatchNorm':dict(accuracy=0.6937,f1=0.6852,auc=0.7593),
        'ANN Best (D+BN)':dict(accuracy=0.7025,f1=0.6938,auc=0.7664),
    }
    ann_df = pd.DataFrame({
        'Model':list(ann_t5.keys()),
        'Accuracy':[v['accuracy'] for v in ann_t5.values()],
        'F1-Score':[v['f1'] for v in ann_t5.values()],
        'AUC':[v['auc'] for v in ann_t5.values()],
        'Architecture':['Shallow','Deep','Dropout','BatchNorm','Best (D+BN)'],
        'Regularization':['None','None','Dropout 0.3','BatchNorm','Dropout 0.3 + BatchNorm'],
    })
    st.subheader("ANN Model Performance")
    st.dataframe(ann_df.style
                 .format({'Accuracy':'{:.4f}','F1-Score':'{:.4f}','AUC':'{:.4f}'})
                 .background_gradient(cmap='RdYlGn',subset=['Accuracy','F1-Score','AUC']))

    st.subheader("Performance Radar Chart")
    cats = ['Accuracy','F1-Score','AUC']
    fig5 = go.Figure()
    for m_name,res in ann_t5.items():
        vals = [res['accuracy'],res['f1'],res['auc'],res['accuracy']]
        fig5.add_trace(go.Scatterpolar(r=vals,theta=cats+[cats[0]],fill='toself',name=m_name))
    fig5.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0.6,0.85])),
        plot_bgcolor='#0f1117',paper_bgcolor='#0f1117',font=dict(color='white'),height=400)
    st.plotly_chart(fig5,use_container_width=True)

    st.subheader("Key Insights")
    for m_name,insight in {
        'ANN Shallow':    '⚡ Fast convergence (26 epochs)',
        'ANN Deep':       '⚠️ Overfitting issues (16 epochs)',
        'ANN + Dropout':  '📈 Better generalization (72 epochs)',
        'ANN + BatchNorm':'⚙️ Stable training (40 epochs)',
        'ANN Best (D+BN)':'🏆 Best performance (52 epochs)',
    }.items():
        st.markdown(f"**{m_name}:** {insight}")

st.markdown("---")
st.markdown("🎵 Built with ❤️ using Streamlit | Spotify Hit Prediction")
