# 🎵 Spotify Hit Prediction — ML vs ANN

Can AI Predict the Next Chart-Topper? A comprehensive comparison of Machine Learning and Artificial Neural Network models for Spotify song hit prediction.

## 🚀 Features

- **Song Hit Prediction**: Input song features and get predictions from multiple ML and ANN models
- **Exploratory Data Analysis**: Interactive visualizations of the Spotify dataset
- **ML vs ANN Comparison**: Detailed performance comparison between traditional ML and deep learning models
- **ANN Architecture Visualization**: Beautiful visualizations of different neural network architectures
- **ANN Model Comparisons**: In-depth analysis of different ANN variants (Shallow, Deep, Dropout, BatchNorm, etc.)

## 📊 Dataset

- **Source**: Synthetic Spotify-style dataset (12,000 songs)
- **Features**: 17 audio features including danceability, energy, loudness, tempo, etc.
- **Target**: Binary classification (Hit vs Not Hit based on popularity)

## 🤖 Models

### Machine Learning Models
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- SVM
- KNN

### Artificial Neural Networks
- ANN Shallow (2 hidden layers)
- ANN Deep (4 hidden layers)
- ANN + Dropout
- ANN + Batch Normalization
- ANN Best (Dropout + BatchNorm)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd spotify_prediction_final
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## 📁 Project Structure

```
spotify_prediction_final/
├── app.py                    # Main Streamlit application
├── Spotify_ML_vs_ANN.ipynb   # Jupyter notebook with model training
├── requirements.txt          # Python dependencies
├── data/
│   └── spotify_songs.csv     # Synthetic dataset (12,000 songs)
├── spotify_data.csv          # Real Spotify dataset (1.1M songs) - not used by app
├── musical_bg.jpg           # Background image
├── spotify_logo1.jpg        # Logo image
├── models/                  # Saved models and preprocessors
│   ├── ml_*.pkl            # ML models
│   ├── ann_*.pkl           # ANN models
│   ├── scaler.pkl          # Feature scaler
│   ├── encoders.pkl        # Label encoders
│   └── features.pkl        # Feature list
└── README.md               # This file
```

## 🎯 Usage

1. **Prediction Tab**: Input song features and get hit probability predictions
2. **EDA Tab**: Explore the dataset with interactive visualizations
3. **ML vs ANN Tab**: Compare performance metrics between ML and ANN models
4. **ANN Architectures Tab**: Visualize different neural network architectures
5. **ANN Comparisons Tab**: Detailed analysis of ANN model variants

## 🚀 Deployment

### Local Deployment
```bash
# Using Python
pip install -r requirements.txt
streamlit run app.py

# Using the run script
chmod +x run.sh
./run.sh
```

### Docker Deployment
```bash
# Build the image
docker build -t spotify-hit-prediction .

# Run the container
docker run -p 8501:8501 spotify-hit-prediction
```

### Streamlit Cloud Deployment
1. Push code to GitHub repository
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub account
4. Select the repository and set main file path to `app.py`
5. Click Deploy!

### Heroku Deployment
1. Create a `Procfile`:
   ```
   web: sh setup.sh && streamlit run app.py
   ```
2. Create `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [global]\n\
   \n\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   \n\
   " > ~/.streamlit/config.toml
   ```
3. Deploy to Heroku using the CLI or GitHub integration

## 📈 Model Performance

| Model | Accuracy | F1-Score | AUC |
|-------|----------|----------|-----|
| LightGBM | 0.7450 | 0.7358 | 0.8209 |
| XGBoost | 0.7408 | 0.7319 | 0.8188 |
| ANN Best (D+BN) | 0.7025 | 0.6938 | 0.7664 |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Data Science IITG for the project framework
- Spotify for the inspiration
- TensorFlow and scikit-learn communities

---

Built with ❤️ for the Data Science IITG Deployment Project