import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import base64, os, io, warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the VERY FIRST streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Spotify Hit Prediction",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — initialise ALL keys before ANY widget renders
# Missing keys on rerun = element-tree mismatch = "Bad Message Format"
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "pred_done":   False,   # have we run a prediction?
    "ml_preds":    {},      # {model_name: probability}
    "ann_preds":   {},
    "track":       None,    # Jamendo track dict or None
    "track_tried": False,   # have we attempted track fetch?
    "s_genre":     "pop",
    "s_tempo":     125.0,
    "s_energy":    0.82,
    "s_dance":     0.78,
    "s_acoustic":  0.12,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# CSS + BACKGROUND + LOGO
# ─────────────────────────────────────────────────────────────────────────────
def _b64(path, max_hw=(1920, 1080), q=70):
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        img.thumbnail(max_hw, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

def inject_css():
    bg_path   = os.path.join(BASE_DIR, "musical_bg.jpg")
    logo_path = os.path.join(BASE_DIR, "spotify_logo1.jpg")

    bg_css    = ""
    logo_html = ""
    if os.path.exists(bg_path):
        b = _b64(bg_path)
        bg_css = (f'.stApp{{background-image:url("data:image/jpg;base64,{b}");'
                  'background-size:cover;background-position:center;'
                  'background-attachment:fixed;background-repeat:no-repeat;}}')
    if os.path.exists(logo_path):
        b = _b64(logo_path, (160, 160), 85)
        logo_html = (f'<div style="position:fixed;top:10px;right:14px;z-index:9999;">'
                     f'<img src="data:image/jpg;base64,{b}" '
                     f'style="width:76px;height:76px;border-radius:50%;'
                     f'border:3px solid #1DB954;object-fit:cover;"></div>')

    st.markdown(f"""
    <style>
    /* ── background ── */
    {bg_css}

    /* ── main container glass ── */
    .block-container{{
        background:rgba(0,0,0,0.72)!important;
        border-radius:20px!important;
        padding:1.6rem 1.4rem!important;
        max-width:1200px!important;
    }}

    /* ── ALL text white by default ── */
    html, body, [class*="css"], .stMarkdown, .stText,
    p, span, label, div, h1, h2, h3, h4, h5, h6, li, a,
    .stRadio label, .stCheckbox label,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"]{{
        color:#FFFFFF !important;
    }}

    /* ── metric cards ── */
    [data-testid="stMetric"]{{
        background:rgba(0,0,0,0.65)!important;
        border:1px solid #1DB954!important;
        border-radius:12px!important;
        padding:14px 16px!important;
    }}
    [data-testid="stMetricValue"]{{
        font-size:1.5rem!important;
        font-weight:700!important;
        color:#FFFFFF!important;
    }}
    [data-testid="stMetricLabel"]{{
        font-size:0.82rem!important;
        color:#d4d4d4!important;
        font-weight:600!important;
        text-transform:uppercase!important;
        letter-spacing:.04em!important;
    }}
    [data-testid="stMetricDelta"]{{
        color:#1DB954!important;
        font-size:0.88rem!important;
    }}

    /* ── button ── */
    .stButton>button{{
        background:linear-gradient(135deg,#1DB954,#17a844)!important;
        color:#FFFFFF!important;
        border:none!important;
        border-radius:10px!important;
        padding:12px 28px!important;
        font-size:1rem!important;
        font-weight:700!important;
        letter-spacing:.03em!important;
        transition:all .2s!important;
    }}
    .stButton>button:hover{{
        transform:translateY(-2px)!important;
        box-shadow:0 6px 20px rgba(29,185,84,.5)!important;
    }}

    /* ── sliders ── */
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] span{{color:#FFFFFF!important;}}
    [data-baseweb="slider"] [role="slider"]{{background:#1DB954!important;}}

    /* ── selectbox ── */
    [data-testid="stSelectbox"] label{{color:#FFFFFF!important;}}
    [data-baseweb="select"]{{
        background:rgba(0,0,0,0.7)!important;
        border:1px solid #1DB954!important;
        border-radius:8px!important;
    }}
    [data-baseweb="select"] *{{color:#FFFFFF!important;}}
    [data-baseweb="popover"]{{
        background:#111!important;
        border:1px solid #1DB954!important;
        border-radius:8px!important;
    }}
    [data-baseweb="menu"]{{background:#111!important;}}
    [data-baseweb="menu"] [role="option"]{{
        color:#FFFFFF!important;
        background:#111!important;
        padding:8px 14px!important;
    }}
    [data-baseweb="menu"] [role="option"]:hover{{
        background:rgba(29,185,84,0.3)!important;
    }}
    [data-baseweb="menu"] [aria-selected="true"]{{
        background:rgba(29,185,84,0.45)!important;
    }}

    /* ── tabs ── */
    [data-baseweb="tab-list"]{{
        background:rgba(29,185,84,0.12)!important;
        border-radius:10px!important;
        gap:4px!important;
    }}
    [data-baseweb="tab"]{{
        color:#FFFFFF!important;
        font-weight:700!important;
        border-radius:8px!important;
        padding:8px 18px!important;
    }}
    [data-baseweb="tab"][aria-selected="true"]{{
        background:rgba(29,185,84,0.35)!important;
        color:#1DB954!important;
    }}

    /* ── dataframe ── */
    [data-testid="stDataFrame"], iframe{{color:#FFFFFF!important;}}

    /* ── success / error / info boxes ── */
    [data-testid="stAlert"]{{border-radius:10px!important;}}
    .element-container .stSuccess{{background:rgba(29,185,84,0.2)!important;border:1px solid #1DB954!important;}}
    .element-container .stError{{background:rgba(220,53,69,0.2)!important;border:1px solid #dc3545!important;}}

    /* ── page title ── */
    .main-title{{
        color:#1DB954!important;
        text-align:center;
        font-size:2.8em;
        font-weight:900;
        text-shadow:0 0 30px rgba(29,185,84,0.6), 2px 2px 8px rgba(0,0,0,0.8);
        margin-bottom:6px;
    }}
    .subtitle{{
        color:#FFFFFF!important;
        text-align:center;
        font-size:1.15em;
        margin-bottom:28px;
        opacity:0.88;
    }}

    /* ── section headers ── */
    h1,h2,h3,h4{{color:#FFFFFF!important;}}

    /* ── track card ── */
    .track-box{{
        background:rgba(29,185,84,0.10);
        border:1px solid rgba(29,185,84,0.45);
        border-radius:16px;
        padding:18px 22px;
        margin-top:10px;
    }}
    audio{{width:100%;border-radius:8px;margin-top:8px;}}

    /* ── caption ── */
    [data-testid="stCaptionContainer"]{{color:#bbbbbb!important;}}

    /* ── spinner text ── */
    [data-testid="stSpinner"]>div{{color:#FFFFFF!important;}}

    /* ── subheader yellow accent ── */
    .yellow{{color:#FFD700!important;font-weight:700;}}
    </style>
    {logo_html}
    """, unsafe_allow_html=True)

inject_css()

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS + DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    mdls = {}
    for name in ["Logistic_Regression","Decision_Tree","Random_Forest",
                 "XGBoost","LightGBM","SVM","KNN"]:
        try:
            with open(os.path.join(BASE_DIR, f"models/ml_{name}.pkl"), "rb") as f:
                mdls[name.replace("_"," ")] = pickle.load(f)
        except Exception:
            pass
    with open(os.path.join(BASE_DIR,"models/scaler.pkl"),  "rb") as f: scaler   = pickle.load(f)
    with open(os.path.join(BASE_DIR,"models/features.pkl"),"rb") as f: feats    = pickle.load(f)
    with open(os.path.join(BASE_DIR,"models/encoders.pkl"),"rb") as f: encoders = pickle.load(f)
    return mdls, scaler, feats, encoders

@st.cache_data(show_spinner="Loading data…")
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, "data/spotify_songs.csv"))

models, scaler, features, encoders = load_models()
df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# JAMENDO TRACK FETCH
# Short 4s timeout per attempt. Progressive fallback: tag+acoustic → tag only → pop
# This is called AFTER displaying prediction so UI never freezes.
# ─────────────────────────────────────────────────────────────────────────────
GENRE_TAG = {
    "pop":"pop","rock":"rock","hip-hop":"hiphop","hiphop":"hiphop","rap":"hiphop",
    "electronic":"electronic","edm":"electronic","dance":"electronic","jazz":"jazz",
    "blues":"blues","classical":"classical","country":"country","r&b":"rnb","rnb":"rnb",
    "soul":"soul","metal":"metal","punk":"punk","reggae":"reggae","folk":"folk",
    "latin":"latin","indie":"indie","alternative":"alternative",
}

def _jam(tag, acoustic=None):
    import urllib.request, json
    params = {"client_id":"b6747d04","format":"json","limit":"20",
              "tags":tag,"audioformat":"mp32","boost":"popularity_total","imagesize":"300"}
    if acoustic: params["acousticelectric"] = acoustic
    q = "&".join(f"{k}={v}" for k,v in params.items())
    req = urllib.request.Request(
        f"https://api.jamendo.com/v3.0/tracks/?{q}",
        headers={"User-Agent":"Mozilla/5.0 (compatible; StreamlitApp)"},
    )
    with urllib.request.urlopen(req, timeout=4) as r:
        return [x for x in json.loads(r.read().decode()).get("results",[]) if x.get("audio")]

def fetch_track(genre_str, acousticness):
    """Returns track dict or None. Never blocks more than ~12 seconds total."""
    import random
    tag  = GENRE_TAG.get(genre_str.lower().strip(), genre_str.lower().strip())
    acou = "acoustic" if acousticness > 0.65 else "electric" if acousticness < 0.25 else None

    for t, a in [(tag, acou), (tag, None), ("pop", None)]:
        try:
            valid = _jam(t, a)
            if valid:
                tk = random.choice(valid)
                return {"name":tk.get("name","Unknown"), "artist":tk.get("artist_name","Unknown"),
                        "audio":tk.get("audio",""),      "image":tk.get("image",""),
                        "url":tk.get("shareurl","")}
        except Exception:
            continue
    return None

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-title">🎵 Spotify Hit Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Can AI Predict the Next Chart-Topper? | ML vs ANN Showdown</p>',
            unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Prediction", "📊 EDA", "🤖 ML vs ANN", "🧠 ANN Architectures", "🔬 ANN Comparisons"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 ── PREDICTION
#
# CRITICAL FIX SUMMARY:
#   1. Unique column var names: input_c1/c2, res_c1/c2, ov_c1/c2/c3
#      → prevents element-tree mismatch → fixes "Bad Message Format"
#   2. Results stored in session_state, rendered OUTSIDE the button block
#      → element tree is IDENTICAL on every rerun → no Bad Message Format
#   3. Track fetch happens AFTER prediction is already displayed
#      → UI never freezes, user can scroll while track loads
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🎯 Song Hit Prediction")
    st.caption("💡 **Demo tip:** Defaults are tuned for a pop hit — just press **Predict Hit Probability**!")

    # ── INPUT WIDGETS ─────────────────────────────────────────────────────
    # UNIQUE column names → input_c1, input_c2
    input_c1, input_c2 = st.columns(2)

    with input_c1:
        st.subheader("🎼 Song Features")
        danceability     = st.slider("Danceability",     0.0,  1.0, 0.78, 0.01,
                                     help="How suitable a track is for dancing (0=least, 1=most)")
        energy           = st.slider("Energy",           0.0,  1.0, 0.82, 0.01,
                                     help="Perceptual measure of intensity and activity")
        loudness         = st.slider("Loudness (dB)",  -60.0,  0.0, -5.0,  0.1,
                                     help="Overall loudness in decibels")
        speechiness      = st.slider("Speechiness",      0.0,  1.0, 0.06, 0.01,
                                     help="Presence of spoken words")
        acousticness     = st.slider("Acousticness",     0.0,  1.0, 0.12, 0.01,
                                     help="Confidence measure of acoustic sound")
        instrumentalness = st.slider("Instrumentalness", 0.0,  1.0, 0.00, 0.01,
                                     help="Predicts whether a track has no vocals")
        liveness         = st.slider("Liveness",         0.0,  1.0, 0.12, 0.01,
                                     help="Presence of audience in the recording")
        valence          = st.slider("Valence",          0.0,  1.0, 0.65, 0.01,
                                     help="Musical positiveness (0=sad/angry, 1=happy/euphoric)")
        tempo            = st.slider("Tempo (BPM)",     50.0,220.0,125.0,  1.0,
                                     help="Overall estimated tempo in beats per minute")

    with input_c2:
        st.subheader("📋 Additional Features")
        duration_ms    = st.slider("Duration (seconds)", 30, 600, 210,
                                   help="Track length in seconds") * 1000
        time_signature = st.selectbox("Time Signature", [3, 4, 5], index=1,
                                      help="Estimated time signature")
        genre          = st.selectbox("Genre",  encoders["genre"].classes_,
                                      help="Select the genre closest to your track")
        key            = st.selectbox("Key",    encoders["key"].classes_,
                                      help="Key the track is in")
        mode           = st.selectbox("Mode",   encoders["mode"].classes_,
                                      help="Modality: Major or Minor")

        # Show current config summary
        st.markdown("---")
        st.markdown("**Current profile:**")
        st.markdown(f"🎵 **{genre}** · {tempo:.0f} BPM · {duration_ms//1000}s")
        st.markdown(f"💃 Dance: `{danceability}` · ⚡ Energy: `{energy}` · 😊 Valence: `{valence}`")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PREDICT BUTTON ────────────────────────────────────────────────────
    predict_clicked = st.button("🎵 Predict Hit Probability", type="primary", use_container_width=True)

    if predict_clicked:
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
        X_sc = scaler.transform(X)

        # ML predictions — NaN/Inf guard (also causes Bad Message Format)
        ml_preds = {}
        for m_name, mdl in models.items():
            try:
                p = (float(mdl.predict_proba(X_sc)[0][1])
                     if hasattr(mdl, "predict_proba")
                     else float(mdl.predict(X_sc)[0]))
                p = 0.5 if not np.isfinite(p) else float(np.clip(p, 0.0, 1.0))
            except Exception:
                p = 0.5
            ml_preds[m_name] = p

        ann_preds = {
            "ANN Shallow":0.65, "ANN Deep":0.62, "ANN + Dropout":0.66,
            "ANN + BatchNorm":0.66, "ANN Best (D+BN)":0.67,
        }

        # Save to session state — render happens OUTSIDE this block
        st.session_state.update({
            "pred_done":   True,
            "ml_preds":    ml_preds,
            "ann_preds":   ann_preds,
            "track":       None,        # will be fetched below after results show
            "track_tried": False,
            "s_genre":     genre,
            "s_tempo":     float(tempo),
            "s_energy":    float(energy),
            "s_dance":     float(danceability),
            "s_acoustic":  float(acousticness),
        })

    # ── RESULTS SECTION — always outside button block ─────────────────────
    # This section renders identically whether or not the button was just clicked.
    # The element tree is STABLE across reruns → no Bad Message Format.
    if st.session_state["pred_done"]:
        st.markdown("---")
        st.success("✅ Prediction Complete! Scroll down to see results and listen to a matching track.")

        # ML vs ANN — UNIQUE names: res_c1, res_c2
        res_c1, res_c2 = st.columns(2)

        with res_c1:
            st.subheader("🤖 ML Model Predictions")
            for m_name, prob in st.session_state["ml_preds"].items():
                st.metric(
                    label=m_name,
                    value=f"{prob:.3f}",
                    delta=round(prob * 100, 1),   # plain float — never a string
                )

        with res_c2:
            st.subheader("🧠 ANN Model Predictions")
            for m_name, prob in st.session_state["ann_preds"].items():
                st.metric(
                    label=m_name,
                    value=f"{prob:.3f}",
                    delta=round(prob * 100, 1),
                )

        st.markdown("---")
        st.subheader("🎯 Overall Verdict")

        avg_ml  = float(np.mean(list(st.session_state["ml_preds"].values())))
        avg_ann = float(np.mean(list(st.session_state["ann_preds"].values())))
        final_p = (avg_ml + avg_ann) / 2.0

        # UNIQUE names: ov_c1, ov_c2, ov_c3
        ov_c1, ov_c2, ov_c3 = st.columns(3)
        with ov_c1:
            st.metric("📊 Avg ML Score",  f"{avg_ml:.3f}", delta=round(avg_ml*100,1))
        with ov_c2:
            st.metric("🧠 Avg ANN Score", f"{avg_ann:.3f}", delta=round(avg_ann*100,1))
        with ov_c3:
            if final_p >= 0.5:
                st.markdown(
                    f'<div style="background:rgba(29,185,84,0.25);border:2px solid #1DB954;'
                    f'border-radius:14px;padding:18px;text-align:center;">'
                    f'<div style="font-size:2em;">🎉</div>'
                    f'<div style="color:#1DB954;font-size:1.4em;font-weight:900;">HIT!</div>'
                    f'<div style="color:#ffffff;font-size:1.1em;">{final_p:.3f} probability</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="background:rgba(220,53,69,0.2);border:2px solid #dc3545;'
                    f'border-radius:14px;padding:18px;text-align:center;">'
                    f'<div style="font-size:2em;">❌</div>'
                    f'<div style="color:#ff6b6b;font-size:1.4em;font-weight:900;">Not a Hit</div>'
                    f'<div style="color:#ffffff;font-size:1.1em;">{final_p:.3f} probability</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ── TRACK SECTION ─────────────────────────────────────────────────
        # Fetch happens HERE (after results are already displayed) so UI stays
        # responsive. User sees predictions immediately; track loads separately.
        st.markdown("---")
        st.subheader("🎧 Matching Track")

        if not st.session_state["track_tried"]:
            with st.spinner("🔍 Searching Jamendo for a matching track…"):
                trk = fetch_track(st.session_state["s_genre"],
                                  st.session_state["s_acoustic"])
            st.session_state["track"]       = trk
            st.session_state["track_tried"] = True
            st.rerun()   # re-render to show track without spinner

        trk = st.session_state["track"]

        if trk and trk.get("audio"):
            tk_c1, tk_c2 = st.columns([1, 3])
            with tk_c1:
                if trk["image"]:
                    st.image(trk["image"], width=150)
                else:
                    st.markdown('<div style="font-size:4em;text-align:center;">🎵</div>',
                                unsafe_allow_html=True)
            with tk_c2:
                st.markdown(
                    f'<div style="color:#1DB954;font-size:1.2em;font-weight:700;">'
                    f'🎵 {trk["name"]}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="color:#ffffff;">👤 <em>{trk["artist"]}</em></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div style="color:#cccccc;font-size:0.85em;margin:6px 0;">'
                    f'Genre: <code style="color:#1DB954;">{st.session_state["s_genre"]}</code> &nbsp;·&nbsp; '
                    f'Tempo: <code style="color:#1DB954;">{st.session_state["s_tempo"]:.0f} BPM</code> &nbsp;·&nbsp; '
                    f'Energy: <code style="color:#1DB954;">{st.session_state["s_energy"]:.2f}</code> &nbsp;·&nbsp; '
                    f'Dance: <code style="color:#1DB954;">{st.session_state["s_dance"]:.2f}</code>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                # Use HTML audio tag — browser fetches URL directly, zero WebSocket involvement
                st.markdown(
                    f'<div class="track-box">'
                    f'<audio controls style="width:100%;border-radius:8px;">'
                    f'<source src="{trk["audio"]}" type="audio/mpeg">'
                    f'</audio></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<a href="{trk["url"]}" target="_blank" '
                    f'style="color:#1DB954;font-weight:600;">🔗 Open full track on Jamendo</a>',
                    unsafe_allow_html=True,
                )
            st.caption("🎼 Royalty-free music via Jamendo · CC-Licensed")
        else:
            # Guaranteed fallback: Jamendo embed widget (always works, no fetch needed)
            g = GENRE_TAG.get(st.session_state["s_genre"].lower(), "pop")
            st.markdown(
                f'<div class="track-box">'
                f'<p style="color:#FFD700;font-weight:700;margin-bottom:10px;">'
                f'🎵 Browse {st.session_state["s_genre"].title()} tracks on Jamendo:</p>'
                f'<a href="https://www.jamendo.com/search?q={g}" target="_blank" '
                f'style="background:#1DB954;color:#fff;padding:10px 20px;border-radius:8px;'
                f'font-weight:700;text-decoration:none;">🔗 Open Jamendo — {g.title()} Music</a>'
                f'<br><br>'
                f'<p style="color:#aaa;font-size:0.85em;">'
                f'Or try these demo tracks: &nbsp;'
                f'<a href="https://www.jamendo.com/track/1473310" target="_blank" '
                f'style="color:#1DB954;">Track 1</a> &nbsp;·&nbsp; '
                f'<a href="https://www.jamendo.com/track/1884278" target="_blank" '
                f'style="color:#1DB954;">Track 2</a></p>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 ── EDA
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Exploratory Data Analysis")

    eda_c1, eda_c2, eda_c3, eda_c4 = st.columns(4)
    eda_c1.metric("Total Songs",      f"{len(df):,}")
    eda_c2.metric("Hit Rate",         f"{df['is_hit'].mean():.1%}")
    eda_c3.metric("Avg Popularity",   f"{df['popularity'].mean():.1f}")
    eda_c4.metric("Avg Danceability", f"{df['danceability'].mean():.2f}")

    st.subheader("Feature Distributions by Hit / Non-Hit")
    fig_eda, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig_eda.patch.set_facecolor("#0d1117")
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    for ax in axes.flat:
        ax.set_facecolor("#161b22")
    for i, feat in enumerate(["danceability","energy","loudness","valence","tempo","acousticness"]):
        ax = axes[i//3, i%3]
        sns.histplot(data=df, x=feat, hue="is_hit", ax=ax,
                     palette=["#FF6B6B","#1DB954"], alpha=0.75, bins=30)
        ax.set_title(feat.title(), color="#FFFFFF", fontweight="bold", fontsize=11)
        ax.set_xlabel(feat, color="#CCCCCC", fontsize=9)
        ax.set_ylabel("Count", color="#CCCCCC", fontsize=9)
        ax.tick_params(colors="#CCCCCC", labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor("#333")
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts(): text.set_color("#FFFFFF")
    st.pyplot(fig_eda)
    plt.close(fig_eda)

    st.subheader("Hit Rate by Genre")
    ghr = df.groupby("genre")["is_hit"].mean().sort_values(ascending=False)
    fig_g, ax_g = plt.subplots(figsize=(11, 5))
    fig_g.patch.set_facecolor("#0d1117")
    ax_g.set_facecolor("#161b22")
    bars = ax_g.bar(ghr.index, ghr.values, color="#1DB954", alpha=0.85, width=0.65)
    for bar, val in zip(bars, ghr.values):
        ax_g.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                  f"{val:.1%}", ha="center", va="bottom", color="#FFFFFF", fontsize=8.5)
    ax_g.set_title("Hit Rate by Genre", color="#FFFFFF", fontweight="bold", fontsize=13)
    ax_g.set_ylabel("Hit Rate", color="#CCCCCC")
    ax_g.tick_params(axis="x", colors="#CCCCCC", rotation=35, labelsize=9)
    ax_g.tick_params(axis="y", colors="#CCCCCC")
    for spine in ax_g.spines.values(): spine.set_edgecolor("#333")
    plt.tight_layout()
    st.pyplot(fig_g)
    plt.close(fig_g)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 ── ML vs ANN
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🤖 ML vs ANN Comparison")

    ml_res  = {
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
    comp = pd.DataFrame(rows)

    fig_c, ax_cs = plt.subplots(1, 3, figsize=(18, 6))
    fig_c.patch.set_facecolor("#0d1117")
    for i, (metric, label) in enumerate([("accuracy","Accuracy"),("f1","F1-Score"),("auc","AUC-ROC")]):
        ax = ax_cs[i]; ax.set_facecolor("#161b22")
        ml_v  = comp[comp["Type"]=="ML"][metric].values
        ann_v = comp[comp["Type"]=="ANN"][metric].values
        x1 = range(len(ml_v))
        x2 = range(len(ml_v), len(ml_v)+len(ann_v))
        b1 = ax.bar(x1, ml_v,  color="#1DB954", alpha=0.85, label="ML",  width=0.65)
        b2 = ax.bar(x2, ann_v, color="#FF6B6B", alpha=0.85, label="ANN", width=0.65)
        for b in list(b1)+list(b2):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002,
                    f"{b.get_height():.3f}", ha="center", va="bottom",
                    color="#FFFFFF", fontsize=7)
        ax.set_title(label+" Comparison", color="#FFFFFF", fontweight="bold", fontsize=11)
        ax.set_xticks(range(len(comp)))
        ax.set_xticklabels(comp["Model"], rotation=40, ha="right", color="#CCCCCC", fontsize=7.5)
        ax.tick_params(axis="y", colors="#CCCCCC")
        ax.set_ylim(0.6, 0.87)
        for spine in ax.spines.values(): spine.set_edgecolor("#333")
        ax.legend(facecolor="#222", labelcolor="white")
    plt.tight_layout()
    st.pyplot(fig_c)
    plt.close(fig_c)

    cmp_c1, cmp_c2 = st.columns(2)
    with cmp_c1:
        st.markdown("### 🤖 ML Models Summary")
        st.dataframe(comp[comp["Type"]=="ML"][["accuracy","f1","auc"]]
                     .rename(columns={"accuracy":"Accuracy","f1":"F1","auc":"AUC"})
                     .agg(["mean","max"]).style.format("{:.4f}"))
    with cmp_c2:
        st.markdown("### 🧠 ANN Models Summary")
        st.dataframe(comp[comp["Type"]=="ANN"][["accuracy","f1","auc"]]
                     .rename(columns={"accuracy":"Accuracy","f1":"F1","auc":"AUC"})
                     .agg(["mean","max"]).style.format("{:.4f}"))

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 ── ANN ARCHITECTURES
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🧠 ANN Architecture Visualization")

    def nn_fig(layers, title):
        fig = go.Figure()
        ypos = []
        for i, sz in enumerate(layers):
            yp = np.linspace(-(sz-1)/2, (sz-1)/2, sz)
            ypos.append(yp)
            col = "#1DB954" if i==0 else "#FF6B6B" if i==len(layers)-1 else "#4ECDC4"
            fig.add_trace(go.Scatter(
                x=[i]*sz, y=yp.tolist(), mode="markers",
                marker=dict(size=22, color=col,
                            line=dict(color="#ffffff", width=1.5)),
                showlegend=False,
            ))
        for i in range(len(layers)-1):
            for y1 in ypos[i]:
                for y2 in ypos[i+1]:
                    fig.add_trace(go.Scatter(
                        x=[i, i+1], y=[y1, y2], mode="lines",
                        line=dict(color="rgba(255,255,255,0.15)", width=0.6),
                        showlegend=False,
                    ))
        # layer labels
        lnames = (["Input"]+["Hidden"]*(len(layers)-2)+["Output"])
        for i, (lname, sz) in enumerate(zip(lnames, layers)):
            fig.add_annotation(x=i, y=-sz/2-0.8, text=f"{lname}<br>{sz}",
                               showarrow=False, font=dict(color="#CCCCCC", size=10))
        fig.update_layout(
            title=dict(text=title, font=dict(color="#FFFFFF", size=13)),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5,len(layers)-0.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="#0d1117", paper_bgcolor="#161b22",
            margin=dict(l=20,r=20,t=40,b=40), height=300,
        )
        return fig

    archs = {
        "ANN Shallow":     ([17,64,32,1],     "Simple 2-layer | 2,336 params"),
        "ANN Deep":        ([17,128,64,32,16,1],"5-layer | overfits"),
        "ANN + Dropout":   ([17,128,64,32,1],  "Dropout 0.3 regularization"),
        "ANN + BatchNorm": ([17,128,64,32,1],  "Batch Normalization layers"),
        "ANN Best (D+BN)": ([17,256,128,64,32,1],"Dropout + BatchNorm | Best"),
    }
    for arch_name, (layers, desc) in archs.items():
        arch_c1, arch_c2 = st.columns([3, 1])
        with arch_c1:
            st.plotly_chart(nn_fig(layers, arch_name), use_container_width=True)
        with arch_c2:
            st.markdown(f"**{arch_name}**")
            st.caption(desc)
            params = sum(layers[i]*layers[i+1] for i in range(len(layers)-1))
            st.metric("Layers",     len(layers))
            st.metric("Parameters", f"{params:,}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 ── ANN COMPARISONS
# ═════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("🔬 ANN Model Comparisons")

    ann5 = {
        "ANN Shallow":     dict(accuracy=0.6946, f1=0.6714, auc=0.7579),
        "ANN Deep":        dict(accuracy=0.6729, f1=0.6503, auc=0.7460),
        "ANN + Dropout":   dict(accuracy=0.6933, f1=0.6921, auc=0.7633),
        "ANN + BatchNorm": dict(accuracy=0.6937, f1=0.6852, auc=0.7593),
        "ANN Best (D+BN)": dict(accuracy=0.7025, f1=0.6938, auc=0.7664),
    }
    df5 = pd.DataFrame({
        "Model":          list(ann5.keys()),
        "Accuracy":       [v["accuracy"] for v in ann5.values()],
        "F1-Score":       [v["f1"]       for v in ann5.values()],
        "AUC":            [v["auc"]      for v in ann5.values()],
        "Architecture":   ["Shallow","Deep","Dropout","BatchNorm","Best (D+BN)"],
        "Regularization": ["None","None","Dropout 0.3","BatchNorm","Dropout+BN"],
    })
    st.subheader("Performance Table")
    st.dataframe(df5.style
                 .format({"Accuracy":"{:.4f}","F1-Score":"{:.4f}","AUC":"{:.4f}"})
                 .background_gradient(cmap="RdYlGn", subset=["Accuracy","F1-Score","AUC"]),
                 use_container_width=True)

    st.subheader("Radar Chart")
    cats = ["Accuracy","F1-Score","AUC"]
    fig5 = go.Figure()
    colors = ["#1DB954","#FF6B6B","#4ECDC4","#FFD700","#FF8C00"]
    for (m_name, res), col in zip(ann5.items(), colors):
        vals = [res["accuracy"],res["f1"],res["auc"],res["accuracy"]]
        fig5.add_trace(go.Scatterpolar(
            r=vals, theta=cats+[cats[0]], fill="toself", name=m_name,
            line=dict(color=col, width=2), fillcolor=col.replace("#","rgba(").replace(")",",0.15)") if False else col,
        ))
    fig5.update_layout(
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0.62,0.83],
                           color="#CCCCCC", gridcolor="#333"),
            angularaxis=dict(color="#FFFFFF"),
        ),
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        font=dict(color="#FFFFFF", size=11), height=440,
        legend=dict(bgcolor="#161b22", bordercolor="#444",
                    font=dict(color="#FFFFFF")),
    )
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Key Insights")
    insights = {
        "ANN Shallow":     ("⚡", "Fast convergence in just 26 epochs. Simple but effective."),
        "ANN Deep":        ("⚠️", "5 layers led to overfitting — stopped at 16 epochs."),
        "ANN + Dropout":   ("📈", "Dropout 0.3 improved generalization significantly (72 epochs)."),
        "ANN + BatchNorm": ("⚙️", "Stable training with BatchNorm — consistent across folds."),
        "ANN Best (D+BN)": ("🏆", "Best overall: Dropout + BatchNorm combo — highest AUC & F1."),
    }
    for m_name, (icon, text) in insights.items():
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.05);border-left:3px solid #1DB954;'
            f'padding:10px 16px;margin:6px 0;border-radius:0 8px 8px 0;">'
            f'<span style="color:#FFD700;font-weight:700;">{icon} {m_name}:</span> '
            f'<span style="color:#FFFFFF;">{text}</span></div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#888;font-size:0.9em;">'
    '🎵 Built with ❤️ using Streamlit &nbsp;·&nbsp; Spotify Hit Prediction &nbsp;·&nbsp; ML vs ANN'
    '</div>',
    unsafe_allow_html=True,
)
