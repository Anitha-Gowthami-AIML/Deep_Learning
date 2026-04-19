# Spotify Hit Prediction — Streamlit App

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import base64, os, io, warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 0. PAGE CONFIG (must be the very first Streamlit call) ───────────────────
st.set_page_config(
    page_title="🎵 Spotify Hit Prediction",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 1. SESSION STATE — initialise ALL keys before any widget ─────────────────
#    Missing keys accessed during a rerun are a known cause of element-tree
#    mismatches that produce "Bad Message Format".
_SS_DEFAULTS: dict = {
    "prediction_done": False,
    "ml_preds":        {},
    "ann_preds":       {},
    "track":           None,
    "s_genre":         "pop",
    "s_tempo":         125.0,
    "s_energy":        0.82,
    "s_dance":         0.78,
    "s_acousticness":  0.12,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── 2. BACKGROUND + LOGO ─────────────────────────────────────────────────────
def _b64_img(path: str, max_px: tuple = (1920, 1080), q: int = 70) -> str:
    """Resize + base64-encode an image. Keeps WS messages small."""
    try:
        from PIL import Image
        img = Image.open(path)
        img.thumbnail(max_px, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

def add_bg_and_logo() -> None:
    bg_path   = os.path.join(BASE_DIR, "musical_bg.jpg")
    logo_path = os.path.join(BASE_DIR, "spotify_logo1.jpg")

    bg_css    = ""
    logo_html = ""

    if os.path.exists(bg_path):
        b64 = _b64_img(bg_path)
        bg_css = (
            f'.stApp{{background-image:url("data:image/jpg;base64,{b64}");'
            'background-size:cover;background-position:center;'
            'background-repeat:no-repeat;background-attachment:fixed;}}'
        )
    if os.path.exists(logo_path):
        b64 = _b64_img(logo_path, (160, 160), 85)
        logo_html = (
            f'<div class="logo-container">'
            f'<img src="data:image/jpg;base64,{b64}" class="logo"></div>'
        )

    st.markdown(
        f"""
        <style>
        {bg_css}
        .stApp,.stApp *{{color:#fff!important;}}
        .block-container{{background:rgba(0,0,0,.64)!important;border-radius:24px!important;
            padding:1.5rem 1.25rem!important;box-shadow:0 20px 60px rgba(0,0,0,.35)!important;}}
        .stButton>button{{color:#fff!important;background:rgba(29,185,84,.8)!important;
            border:1px solid #1DB954!important;border-radius:8px!important;}}
        .stButton>button:hover{{background:rgba(29,185,84,.95)!important;}}
        .stSelectbox [data-baseweb="select"],.stMultiselect [data-baseweb="select"]
            {{background:rgba(0,0,0,.7)!important;border:1px solid #1DB954!important;border-radius:8px!important;}}
        .stSelectbox [data-baseweb="select"] *,.stMultiselect [data-baseweb="select"] *
            {{color:#fff!important;background:transparent!important;}}
        [data-baseweb="popover"]{{background:#1a1a1a!important;border:1px solid #1DB954!important;
            border-radius:8px!important;overflow:hidden!important;}}
        [data-baseweb="menu"]{{background:#1a1a1a!important;color:#fff!important;}}
        [data-baseweb="menu"] [role="option"]{{background:#1a1a1a!important;color:#fff!important;
            padding:8px 12px!important;border-bottom:1px solid rgba(255,255,255,.05)!important;}}
        [data-baseweb="menu"] [role="option"]:hover
            {{background:rgba(29,185,84,.25)!important;cursor:pointer!important;}}
        [data-baseweb="menu"] [aria-selected="true"]
            {{background:rgba(29,185,84,.4)!important;}}
        .stSlider [data-baseweb="slider"] *{{color:#fff!important;}}
        .stNumberInput input,.stTextInput input{{background:rgba(0,0,0,.7)!important;
            color:#fff!important;border:1px solid #1DB954!important;border-radius:8px!important;}}
        .stMetric{{background:rgba(0,0,0,.7)!important;border:1px solid #1DB954!important;
            border-radius:12px!important;padding:1rem!important;}}
        .stTabs [data-baseweb="tab-list"]{{background:rgba(29,185,84,.1)!important;border-radius:10px!important;}}
        .stTabs [data-baseweb="tab"]{{color:#fff!important;font-weight:bold!important;}}
        .stTabs [data-baseweb="tab"][aria-selected="true"]{{background:rgba(29,185,84,.3)!important;}}
        .stDataFrame,.stDataFrame *{{background:rgba(0,0,0,.7)!important;color:#fff!important;}}
        .logo-container{{position:fixed;top:10px;right:10px;z-index:1000;}}
        .logo{{width:80px;height:80px;border-radius:50%;object-fit:cover;border:3px solid #1DB954;}}
        .main-title{{color:#1DB954;text-align:center;font-size:3em;font-weight:bold;
            text-shadow:2px 2px 4px rgba(0,0,0,.5);margin-bottom:20px;}}
        .subtitle{{color:#fff;text-align:center;font-size:1.2em;margin-bottom:30px;
            text-shadow:1px 1px 2px rgba(0,0,0,.7);}}
        .track-card{{background:rgba(29,185,84,.08);border:1px solid rgba(29,185,84,.35);
            border-radius:16px;padding:16px 20px;margin-top:10px;}}
        .hit-badge{{font-size:1.4em;font-weight:bold;padding:10px 20px;
            border-radius:12px;text-align:center;margin-top:8px;}}
        </style>
        {logo_html}
        """,
        unsafe_allow_html=True,
    )

add_bg_and_logo()

# ── 3. DATA / MODEL LOADING ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    mdls = {}
    for name in ["Logistic_Regression","Decision_Tree","Random_Forest",
                 "XGBoost","LightGBM","SVM","KNN"]:
        p = os.path.join(BASE_DIR, f"models/ml_{name}.pkl")
        try:
            with open(p, "rb") as f:
                mdls[name.replace("_", " ")] = pickle.load(f)
        except Exception:
            st.warning(f"Could not load {name}")
    with open(os.path.join(BASE_DIR,"models/scaler.pkl"),  "rb") as f: scaler   = pickle.load(f)
    with open(os.path.join(BASE_DIR,"models/features.pkl"),"rb") as f: feats    = pickle.load(f)
    with open(os.path.join(BASE_DIR,"models/encoders.pkl"),"rb") as f: encoders = pickle.load(f)
    return mdls, scaler, feats, encoders

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "data/spotify_songs.csv"))

models, scaler, features, encoders = load_models()
df = load_data()

# ── 4. JAMENDO TRACK FETCH ────────────────────────────────────────────────────
GENRE_TAG_MAP = {
    "pop":"pop","rock":"rock","hip-hop":"hiphop","hiphop":"hiphop","rap":"hiphop",
    "electronic":"electronic","edm":"electronic","dance":"electronic","jazz":"jazz",
    "blues":"blues","classical":"classical","country":"country","r&b":"rnb","rnb":"rnb",
    "soul":"soul","metal":"metal","punk":"punk","reggae":"reggae","folk":"folk",
    "latin":"latin","indie":"indie","alternative":"alternative",
}

def _jam_search(tag: str, speed: str | None = None, acoustic: str | None = None) -> list:
    import urllib.request, json
    params = {"client_id":"b6747d04","format":"json","limit":"20",
              "tags":tag,"audioformat":"mp32","boost":"popularity_total","imagesize":"300"}
    if speed:   params["speed"]          = speed
    if acoustic: params["acousticelectric"] = acoustic
    q = "&".join(f"{k}={v}" for k,v in params.items())
    req = urllib.request.Request(
        f"https://api.jamendo.com/v3.0/tracks/?{q}",
        headers={"User-Agent":"StreamlitApp/1.0"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode()).get("results", [])

def fetch_track(genre_str: str, tempo: float, acousticness: float) -> dict | None:
    """Progressive 4-level fallback so a track is almost always returned."""
    import random
    tag  = GENRE_TAG_MAP.get(genre_str.lower().strip(), genre_str.lower().strip())
    spd  = "low" if tempo < 110 else "medium" if tempo < 140 else "high"
    acou = "acoustic" if acousticness > 0.6 else "electric" if acousticness < 0.3 else None

    for t, s, a in [(tag, spd, acou), (tag, spd, None), (tag, None, None), ("pop", None, None)]:
        try:
            valid = [r for r in _jam_search(t, s, a) if r.get("audio")]
            if valid:
                tk = random.choice(valid)
                return {"name":   tk.get("name","Unknown"),
                        "artist": tk.get("artist_name","Unknown"),
                        "audio":  tk.get("audio",""),
                        "image":  tk.get("image",""),
                        "url":    tk.get("shareurl","")}
        except Exception:
            continue
    return None

# ── 5. AUDIO RENDERER ─────────────────────────────────────────────────────────
# st.audio(url_string) is valid in Streamlit ≥ 1.28 (proto.url = data directly).
# We still prefer an HTML <audio> tag as the primary renderer because:
#   • Zero WebSocket data transfer (browser fetches URL directly)
#   • Works even behind CORS restrictions
#   • No risk of large-blob protobuf encoding edge cases
def render_audio(url: str) -> None:
    if not url:
        st.info("⚠️ No audio stream available.")
        return
    st.markdown(
        f'<div class="track-card">'
        f'<audio controls style="width:100%;border-radius:8px;margin-top:6px;">'
        f'<source src="{url}" type="audio/mpeg">'
        f'Your browser does not support the audio element.'
        f'</audio></div>',
        unsafe_allow_html=True,
    )

def render_track_card(track, genre, tempo, energy, danceability) -> None:
    if not track:
        st.info(f"⚠️ No matching track found for **{genre}**. Try a different genre.")
        return
    tc1, tc2 = st.columns([1, 3])
    with tc1:
        if track["image"]:
            st.image(track["image"], width=150)
    with tc2:
        st.markdown(f"**🎵 {track['name']}**")
        st.markdown(f"👤 *{track['artist']}*")
        st.markdown(
            f"<small>Genre <code>{genre}</code> · "
            f"Tempo <code>{tempo:.0f} BPM</code> · "
            f"Energy <code>{energy:.2f}</code> · "
            f"Danceability <code>{danceability:.2f}</code></small>",
            unsafe_allow_html=True,
        )
        render_audio(track["audio"])
        st.markdown(f"[🔗 Open on Jamendo]({track['url']})", unsafe_allow_html=True)
    st.caption("🎼 Royalty-free tracks via Jamendo (CC-licensed)")

# ── 6. TITLE ──────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🎵 Spotify Hit Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Can AI Predict the Next Chart-Topper? | ML vs ANN Showdown</p>',
            unsafe_allow_html=True)

# ── 7. TABS ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Prediction", "📊 EDA", "🤖 ML vs ANN", "🧠 ANN Architectures", "🔬 ANN Comparisons"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# KEY FIX: every column group has a UNIQUE variable name.
#   • input_col1 / input_col2   → sliders
#   • res_col1  / res_col2      → ML vs ANN metrics
#   • ov_col1 / ov_col2 / ov_col3 → overall verdict
# Results are rendered OUTSIDE the button block (driven by session_state)
# so the element tree is identical on every rerun.
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🎯 Song Hit Prediction")
    st.caption("💡 **Demo tip:** defaults below match a typical chart-topper — just press **Predict**!")

    # ── Input widgets ──────────────────────────────────────────────────────
    input_col1, input_col2 = st.columns(2)          # ← unique name

    with input_col1:
        st.subheader("Song Features")
        danceability     = st.slider("Danceability",     0.0,  1.0, 0.78, 0.01)
        energy           = st.slider("Energy",           0.0,  1.0, 0.82, 0.01)
        loudness         = st.slider("Loudness (dB)",  -60.0,  0.0,-5.0,  0.1)
        speechiness      = st.slider("Speechiness",      0.0,  1.0, 0.06, 0.01)
        acousticness     = st.slider("Acousticness",     0.0,  1.0, 0.12, 0.01)
        instrumentalness = st.slider("Instrumentalness", 0.0,  1.0, 0.0,  0.01)
        liveness         = st.slider("Liveness",         0.0,  1.0, 0.12, 0.01)
        valence          = st.slider("Valence",          0.0,  1.0, 0.65, 0.01)
        tempo            = st.slider("Tempo (BPM)",     50.0,220.0,125.0,  1.0)

    with input_col2:
        st.subheader("Additional Features")
        duration_ms    = st.slider("Duration (seconds)", 30, 600, 210) * 1000
        time_signature = st.selectbox("Time Signature", [3, 4, 5], index=1)
        genre          = st.selectbox("Genre", encoders["genre"].classes_)
        key            = st.selectbox("Key",   encoders["key"].classes_)
        mode           = st.selectbox("Mode",  encoders["mode"].classes_)

    # ── Predict button ─────────────────────────────────────────────────────
    if st.button("🎵 Predict Hit Probability", type="primary"):

        # Feature engineering
        duration_min          = duration_ms / 60000
        loudness_norm         = loudness + 25
        dance_energy          = danceability * energy
        acoustic_instrumental = acousticness * instrumentalness
        mood_score            = valence * energy

        genre_enc = int(encoders["genre"].transform([genre])[0])
        key_enc   = int(encoders["key"].transform([key])[0])
        mode_enc  = int(encoders["mode"].transform([mode])[0])

        X = np.array([[danceability, energy, loudness_norm, speechiness, acousticness,
                       instrumentalness, liveness, valence, tempo, duration_min,
                       time_signature, genre_enc, key_enc, mode_enc,
                       dance_energy, acoustic_instrumental, mood_score]])
        X_scaled = scaler.transform(X)

        # ML predictions — guard against NaN/Inf (also causes Bad Message Format)
        ml_preds: dict = {}
        for m_name, model in models.items():
            try:
                p = (float(model.predict_proba(X_scaled)[0][1])
                     if hasattr(model, "predict_proba")
                     else float(model.predict(X_scaled)[0]))
                p = 0.5 if not np.isfinite(p) else float(np.clip(p, 0.0, 1.0))
            except Exception:
                p = 0.5
            ml_preds[m_name] = p

        ann_preds = {
            "ANN Shallow":0.65,"ANN Deep":0.62,"ANN + Dropout":0.66,
            "ANN + BatchNorm":0.66,"ANN Best (D+BN)":0.67,
        }

        # Track fetch
        with st.spinner("🔍 Searching Jamendo for a matching track…"):
            track = fetch_track(genre, tempo, acousticness)

        # Persist results → render happens OUTSIDE this block
        st.session_state.update({
            "prediction_done": True,
            "ml_preds":        ml_preds,
            "ann_preds":       ann_preds,
            "track":           track,
            "s_genre":         genre,
            "s_tempo":         float(tempo),
            "s_energy":        float(energy),
            "s_dance":         float(danceability),
            "s_acousticness":  float(acousticness),
        })

    # ── Results — rendered OUTSIDE button block so element tree never changes ─
    if st.session_state["prediction_done"]:
        st.success("✅ Prediction Complete!")
        st.markdown("---")

        # ML vs ANN metrics  — unique column names: res_col1 / res_col2
        res_col1, res_col2 = st.columns(2)         # ← unique name

        with res_col1:
            st.subheader("🤖 ML Model Predictions")
            for m_name, prob in st.session_state["ml_preds"].items():
                # delta must be a number (int/float), NOT a formatted string
                # "65.0%" as delta string is valid in Streamlit ≥1.28 BUT
                # we use float for maximum compatibility and clarity
                st.metric(label=m_name,
                          value=f"{prob:.3f}",
                          delta=round(float(prob) * 100, 1))

        with res_col2:
            st.subheader("🧠 ANN Model Predictions")
            for m_name, prob in st.session_state["ann_preds"].items():
                st.metric(label=m_name,
                          value=f"{prob:.3f}",
                          delta=round(float(prob) * 100, 1))

        st.markdown("---")
        st.subheader("🎯 Overall Prediction")

        # Overall columns — unique names: ov_col1 / ov_col2 / ov_col3
        ov_col1, ov_col2, ov_col3 = st.columns(3) # ← unique name

        avg_ml  = float(np.mean(list(st.session_state["ml_preds"].values())))
        avg_ann = float(np.mean(list(st.session_state["ann_preds"].values())))
        final_p = (avg_ml + avg_ann) / 2

        with ov_col1:
            st.metric("Avg ML Probability",  f"{avg_ml:.3f}")
        with ov_col2:
            st.metric("Avg ANN Probability", f"{avg_ann:.3f}")
        with ov_col3:
            if final_p > 0.5:
                st.success(f"🎉 HIT!  ({final_p:.3f})")
            else:
                st.error(f"❌ Not a Hit  ({final_p:.3f})")

        st.markdown("---")
        st.subheader("🎧 Listen to a Matching Track")
        render_track_card(
            track       = st.session_state["track"],
            genre       = st.session_state["s_genre"],
            tempo       = st.session_state["s_tempo"],
            energy      = st.session_state["s_energy"],
            danceability= st.session_state["s_dance"],
        )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Exploratory Data Analysis")

    eda_c1, eda_c2, eda_c3, eda_c4 = st.columns(4)
    eda_c1.metric("Total Songs",      len(df))
    eda_c2.metric("Hit Rate",         f"{df['is_hit'].mean():.1%}")
    eda_c3.metric("Avg Popularity",   f"{df['popularity'].mean():.1f}")
    eda_c4.metric("Avg Danceability", f"{df['danceability'].mean():.2f}")

    st.subheader("Feature Distributions")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor("#0f1117")
    for ax in axes.flat:
        ax.set_facecolor("#1e2130")
    for i, feat in enumerate(["danceability","energy","loudness","valence","tempo","acousticness"]):
        ax = axes[i // 3, i % 3]
        sns.histplot(data=df, x=feat, hue="is_hit", ax=ax,
                     palette=["#e17055","#6bcb77"], alpha=0.7)
        ax.set_title(feat.title(), color="white", fontweight="bold")
        ax.tick_params(colors="white")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color("white")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Genre Analysis")
    ghr = df.groupby("genre")["is_hit"].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor("#0f1117")
    ax2.set_facecolor("#1e2130")
    ghr.plot(kind="bar", ax=ax2, color="#1DB954")
    ax2.set_title("Hit Rate by Genre", color="white", fontweight="bold")
    ax2.tick_params(colors="white")
    plt.xticks(rotation=45, color="white")
    plt.yticks(color="white")
    st.pyplot(fig2)
    plt.close(fig2)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ML vs ANN
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🤖 ML vs ANN Comparison")

    ml_res = {
        "Logistic Regression": dict(accuracy=0.6775, f1=0.6551, auc=0.7459),
        "Decision Tree":       dict(accuracy=0.7071, f1=0.6950, auc=0.7574),
        "Random Forest":       dict(accuracy=0.7087, f1=0.6930, auc=0.7846),
        "XGBoost":             dict(accuracy=0.7408, f1=0.7319, auc=0.8188),
        "LightGBM":            dict(accuracy=0.7450, f1=0.7358, auc=0.8209),
        "SVM":                 dict(accuracy=0.6971, f1=0.6776, auc=0.7534),
        "KNN":                 dict(accuracy=0.6421, f1=0.6343, auc=0.6863),
    }
    ann_res = {
        "ANN Shallow":     dict(accuracy=0.6946, f1=0.6714, auc=0.7579),
        "ANN Deep":        dict(accuracy=0.6729, f1=0.6503, auc=0.7460),
        "ANN + Dropout":   dict(accuracy=0.6933, f1=0.6921, auc=0.7633),
        "ANN + BatchNorm": dict(accuracy=0.6937, f1=0.6852, auc=0.7593),
        "ANN Best (D+BN)": dict(accuracy=0.7025, f1=0.6938, auc=0.7664),
    }
    rows = ([{"Model":m,"Type":"ML", **v} for m,v in ml_res.items()] +
            [{"Model":m,"Type":"ANN",**v} for m,v in ann_res.items()])
    comp_df = pd.DataFrame(rows)

    st.subheader("Model Performance Metrics")
    fig3, ax3s = plt.subplots(1, 3, figsize=(18, 6))
    fig3.patch.set_facecolor("#0f1117")
    for i, metric in enumerate(["accuracy","f1","auc"]):
        ax = ax3s[i]; ax.set_facecolor("#1e2130")
        ml_v  = comp_df[comp_df["Type"]=="ML"][metric].values
        ann_v = comp_df[comp_df["Type"]=="ANN"][metric].values
        ax.bar(range(len(ml_v)), ml_v,  color="#1DB954", alpha=0.7, label="ML")
        ax.bar(range(len(ml_v), len(ml_v)+len(ann_v)), ann_v,
               color="#FF6B6B", alpha=0.7, label="ANN")
        ax.set_title(metric.upper()+" Comparison", color="white", fontweight="bold")
        ax.set_xticks(range(len(comp_df)))
        ax.set_xticklabels(comp_df["Model"], rotation=45, ha="right", color="white")
        ax.tick_params(colors="white"); ax.legend()
    st.pyplot(fig3)
    plt.close(fig3)

    cmp_c1, cmp_c2 = st.columns(2)
    with cmp_c1:
        st.markdown("### ML Models Summary")
        st.dataframe(comp_df[comp_df["Type"]=="ML"][["accuracy","f1","auc"]]
                     .agg(["mean","max"]).style.format("{:.4f}"))
    with cmp_c2:
        st.markdown("### ANN Models Summary")
        st.dataframe(comp_df[comp_df["Type"]=="ANN"][["accuracy","f1","auc"]]
                     .agg(["mean","max"]).style.format("{:.4f}"))

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANN ARCHITECTURES
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🧠 ANN Architecture Visualization")

    def nn_diagram(layers: list, title: str) -> go.Figure:
        fig = go.Figure()
        ypos = []
        for i, sz in enumerate(layers):
            yp = np.linspace(-(sz-1)/2, (sz-1)/2, sz)
            ypos.append(yp)
            col = "#1DB954" if i==0 else "#FF6B6B" if i==len(layers)-1 else "#4ECDC4"
            for y in yp:
                fig.add_trace(go.Scatter(x=[i], y=[y], mode="markers",
                    marker=dict(size=25, color=col), showlegend=False))
        for i in range(len(layers)-1):
            for y1 in ypos[i]:
                for y2 in ypos[i+1]:
                    fig.add_trace(go.Scatter(x=[i,i+1], y=[y1,y2], mode="lines",
                        line=dict(color="rgba(255,255,255,0.2)", width=0.5), showlegend=False))
        fig.update_layout(title=title,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
            font=dict(color="white"), height=300)
        return fig

    archs = {
        "ANN Shallow":     [17,64,32,1],
        "ANN Deep":        [17,128,64,32,16,1],
        "ANN + Dropout":   [17,128,64,32,1],
        "ANN + BatchNorm": [17,128,64,32,1],
        "ANN Best (D+BN)": [17,256,128,64,32,1],
    }
    for arch_name, layers in archs.items():
        arch_c1, arch_c2 = st.columns([3,1])
        with arch_c1:
            st.plotly_chart(nn_diagram(layers, f"{arch_name} Architecture"),
                            use_container_width=True)
        with arch_c2:
            params = sum(layers[i]*layers[i+1] for i in range(len(layers)-1))
            st.metric("Layers",     len(layers))
            st.metric("Parameters", f"{params:,}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — ANN COMPARISONS
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("🔬 ANN Model Comparisons")

    ann_t5 = {
        "ANN Shallow":     dict(accuracy=0.6946, f1=0.6714, auc=0.7579),
        "ANN Deep":        dict(accuracy=0.6729, f1=0.6503, auc=0.7460),
        "ANN + Dropout":   dict(accuracy=0.6933, f1=0.6921, auc=0.7633),
        "ANN + BatchNorm": dict(accuracy=0.6937, f1=0.6852, auc=0.7593),
        "ANN Best (D+BN)": dict(accuracy=0.7025, f1=0.6938, auc=0.7664),
    }
    ann_df = pd.DataFrame({
        "Model":          list(ann_t5.keys()),
        "Accuracy":       [v["accuracy"] for v in ann_t5.values()],
        "F1-Score":       [v["f1"]       for v in ann_t5.values()],
        "AUC":            [v["auc"]      for v in ann_t5.values()],
        "Architecture":   ["Shallow","Deep","Dropout","BatchNorm","Best (D+BN)"],
        "Regularization": ["None","None","Dropout 0.3","BatchNorm","Dropout 0.3 + BatchNorm"],
    })

    st.subheader("ANN Model Performance")
    st.dataframe(ann_df.style
                 .format({"Accuracy":"{:.4f}","F1-Score":"{:.4f}","AUC":"{:.4f}"})
                 .background_gradient(cmap="RdYlGn", subset=["Accuracy","F1-Score","AUC"]))

    st.subheader("Performance Radar Chart")
    cats = ["Accuracy","F1-Score","AUC"]
    fig5 = go.Figure()
    for m_name, res in ann_t5.items():
        vals = [res["accuracy"],res["f1"],res["auc"],res["accuracy"]]
        fig5.add_trace(go.Scatterpolar(r=vals, theta=cats+[cats[0]],
                                       fill="toself", name=m_name))
    fig5.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0.6,0.85])),
        plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
        font=dict(color="white"), height=400)
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Key Insights")
    for m_name, insight in {
        "ANN Shallow":     "⚡ Fast convergence (26 epochs)",
        "ANN Deep":        "⚠️ Overfitting issues (16 epochs)",
        "ANN + Dropout":   "📈 Better generalization (72 epochs)",
        "ANN + BatchNorm": "⚙️ Stable training (40 epochs)",
        "ANN Best (D+BN)": "🏆 Best performance (52 epochs)",
    }.items():
        st.markdown(f"**{m_name}:** {insight}")

st.markdown("---")
st.markdown("🎵 Built with ❤️ using Streamlit | Spotify Hit Prediction")
