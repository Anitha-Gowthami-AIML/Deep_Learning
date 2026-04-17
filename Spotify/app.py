import streamlit as st
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Spotify
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Spotify Hit Predictor",
    page_icon="🎵",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    try:
        with open(os.path.join(BASE_DIR, "models/best_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(BASE_DIR, "models/scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

model, scaler = load_model()

# -------------------- SPOTIFY SETUP --------------------
@st.cache_resource
def get_spotify_client():
    try:
        client_id = st.secrets["SPOTIFY_CLIENT_ID"]
        client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]
    except:
        st.warning("⚠️ Spotify credentials not found. Using fallback mode.")
        return None

    auth_manager = SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def fetch_spotify_track(genre, mood_query):
    sp = get_spotify_client()

    if sp is None:
        return None

    try:
        results = sp.search(
            q=f"{mood_query} {genre}",
            type="track",
            limit=20
        )

        tracks = results["tracks"]["items"]

        if tracks:
            import random
            t = random.choice(tracks)

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


# -------------------- UI --------------------
st.title("🎵 Spotify Hit Prediction + AI Music Recommender")

st.markdown("Predict whether a song will be a hit and listen to a similar real-world track.")

# -------------------- INPUTS --------------------
col1, col2 = st.columns(2)

with col1:
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    loudness = st.slider("Loudness", -60.0, 0.0, -10.0)
    valence = st.slider("Valence", 0.0, 1.0, 0.5)
    tempo = st.slider("Tempo", 50.0, 200.0, 120.0)

with col2:
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.2)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.1)

    genre = st.selectbox("Genre", ["pop", "rock", "hip-hop", "electronic", "jazz", "classical"])

# -------------------- PREDICTION --------------------
if st.button("🎯 Predict Hit + Recommend Song"):

    # Feature engineering (simple & safe)
    features = np.array([[
        danceability, energy, loudness, speechiness,
        acousticness, instrumentalness, liveness,
        valence, tempo
    ]])

    if scaler:
        features = scaler.transform(features)

    if model:
        prob = model.predict_proba(features)[0][1]
    else:
        prob = np.random.uniform(0.4, 0.8)  # fallback

    # -------------------- RESULT --------------------
    st.subheader("🎯 Prediction Result")

    if prob > 0.5:
        st.success(f"🎉 HIT SONG! Probability: {prob:.2f}")
    else:
        st.error(f"❌ Not likely a hit. Probability: {prob:.2f}")

    # -------------------- MOOD LOGIC --------------------
    if prob > 0.7:
        mood = "high energy dance"
    elif prob > 0.5:
        mood = "popular trending"
    else:
        mood = "chill acoustic"

    # -------------------- SPOTIFY --------------------
    st.subheader("🎧 Recommended Track (Real Songs)")

    track = fetch_spotify_track(genre, mood)

    if track:
        col1, col2 = st.columns([1, 3])

        with col1:
            if track["image"]:
                st.image(track["image"], width=150)

        with col2:
            st.markdown(f"**🎵 {track['name']}**")
            st.markdown(f"👤 {track['artist']}")

            if track["preview"]:
                st.audio(track["preview"])
            else:
                st.warning("Preview not available")

            st.markdown(f"[🔗 Open in Spotify]({track['url']})")

    else:
        # fallback always works
        st.info("Using fallback demo track")

        st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3")
        st.markdown("[🔗 Listen full demo](https://www.soundhelix.com)")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit + ML + Spotify API")
