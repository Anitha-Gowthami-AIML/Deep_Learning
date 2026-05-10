"""
╔══════════════════════════════════════════════════════════════════════════╗
║   Bank Customer Churn Prediction — Streamlit App                        ║
║   Matches : ann_churn_modelling.py  (17 models)                         ║
║   Tabs    : ANN Viz Lab · EDA · Phase A (11 models) ·                   ║
║             Phase B (6 models) · Keras Tuner · Save/Load ·              ║
║             Grand Comparison · Live Predictor                           ║
║   Run     : streamlit run app.py                                        ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os, math, warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏦 ANN Churn Lab",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp{background:#0f1117;color:#e0e0e0}
  [data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0a0f1e 0%,#0d1530 100%);
    border-right:1px solid rgba(80,160,255,.15)
  }
  .stTabs [data-baseweb="tab-list"]{background:#0d1530;border-radius:8px;padding:4px;gap:3px;display:flex;flex-wrap:nowrap;overflow-x:auto;overflow-y:hidden;scrollbar-width:thin;scrollbar-color:rgba(120,180,255,.6) transparent;}
  .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar{height:8px}
  .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track{background:transparent}
  .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb{background:rgba(120,180,255,.4);border-radius:10px}
  .stTabs [data-baseweb="tab"]{color:#4a7ab5;border-radius:6px;font-weight:600;font-size:12px;padding:5px 10px;white-space:nowrap;}
  .stTabs [aria-selected="true"]{background:rgba(60,120,255,.25)!important;color:#80c0ff!important}
  [data-testid="metric-container"]{
    background:rgba(20,40,100,.25);
    border:1px solid rgba(80,160,255,.2);
    border-radius:10px;padding:12px
  }
  h1{color:#80c0ff!important} h2{color:#60a0e0!important} h3{color:#5090d0!important}
  [data-testid="stDataFrame"]{border:1px solid rgba(60,120,200,.3);border-radius:8px}
  .stButton>button{
    background:linear-gradient(135deg,#1a3a8a,#2255cc);
    color:white;border:1px solid rgba(80,150,255,.4);border-radius:8px;font-weight:600
  }
  .stButton>button:hover{background:linear-gradient(135deg,#2255cc,#3366ee)}
  .winner-badge{
    background:linear-gradient(135deg,#1a4a1a,#225522);
    border:2px solid #44bb44;border-radius:12px;padding:16px 24px;text-align:center
  }
  .model-card{
    background:rgba(20,40,80,.25);border:1px solid rgba(80,160,255,.18);
    border-radius:8px;padding:10px 14px;margin:4px 0
  }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════
SEED       = 42
MODEL_DIR  = "saved_models"
DATA_FILE  = "Churn_Modeling.csv"

# Exact filenames written by ann_churn_modelling.py
MODEL_FILES = {
    "A1 — Batch GD (SGD)":             "A1_BatchGD_SGD.keras",
    "A2 — Stochastic GD (SGD)":        "A2_StochasticGD_SGD.keras",
    "A3 — Mini-Batch Adam":            "A3_Adam.keras",
    "A4 — Mini-Batch RMSprop":         "A4_RMSprop.keras",
    "A5 — Early Stopping":             "A5_EarlyStopping.keras",
    "A6 — Dropout (0.3+0.2)":          "A6_Dropout.keras",
    "A7 — Glorot Uniform (Xavier)":    "A7_GlorotUniform.keras",
    "A8 — He Normal (Kaiming)":        "A8_HeNormal.keras",
    "A9 — He Uniform":                 "A9_HeUniform.keras",
    "A10 — Random Normal":             "A10_RandomNormal.keras",
    "A11 — Keras Tuner (Phase A)":     "A11_KerasTuner.keras",
    "B1 — Class Weights":              "B1_ClassWeights.keras",
    "B2 — SMOTE":                      "B2_SMOTE.keras",
    "B3 — Random Oversampling":        "B3_RandomOver.keras",
    "B4 — Random Undersampling":       "B4_RandomUnder.keras",
    "B5 — SMOTEENN":                   "B5_SMOTEENN.keras",
    "B6 — Keras Tuner (Phase B)":      "B6_KerasTuner.keras",
}

PHASE_A_KEYS = [k for k in MODEL_FILES if k.startswith("A")]
PHASE_B_KEYS = [k for k in MODEL_FILES if k.startswith("B")]

# ── Dark-theme helper for matplotlib ──────────────────────────────────────
def _dark_fig(*args, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor("#0d1530")
    if isinstance(ax, np.ndarray):
        for a in ax.flatten():
            a.set_facecolor("#0a0f1e")
            for sp in a.spines.values(): sp.set_edgecolor("#1e3a6a")
            a.tick_params(colors="#80a0c0")
    else:
        ax.set_facecolor("#0a0f1e")
        for sp in ax.spines.values(): sp.set_edgecolor("#1e3a6a")
        ax.tick_params(colors="#80a0c0")
    return fig, ax

# ══════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_raw_data():
    if not os.path.exists(DATA_FILE):
        return None
    return pd.read_csv(DATA_FILE)

@st.cache_data
def preprocess_data(df_raw):
    df = df_raw.copy()
    df.drop(columns=["RowNumber","CustomerId","Surname"], inplace=True)
    df.rename(columns={"Exited":"Churn"}, inplace=True)
    feature_cols = df.select_dtypes(include="number").drop("Churn",axis=1).columns
    for col in feature_cols:
        if abs(df[col].skew()) > 1:
            df[col] = np.log1p(df[col])
    df = pd.get_dummies(df, columns=["Geography"], drop_first=True)
    df["Gender"] = df["Gender"].map({"Male":0,"Female":1})
    for col in ["CreditScore","Age","Balance","EstimatedSalary"]:
        Q1,Q3 = df[col].quantile(.25), df[col].quantile(.75)
        df[col] = df[col].clip(Q1-1.5*(Q3-Q1), Q3+1.5*(Q3-Q1))
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)
    return X, y, X_tr_s, X_te_s, y_tr, y_te, sc, list(X.columns)

@st.cache_resource
def load_keras_model(path):
    try:
        from tensorflow.keras.models import load_model
        return load_model(path)
    except Exception as e:
        warnings.warn(f"Failed to load Keras model '{path}': {e}")
        return None

@st.cache_resource
def load_scaler():
    p = os.path.join(MODEL_DIR, "scaler.pkl")
    if os.path.exists(p):
        with open(p,"rb") as f: return pickle.load(f)
    return None

def get_available_models():
    return {n: os.path.join(MODEL_DIR,f)
            for n,f in MODEL_FILES.items()
            if os.path.exists(os.path.join(MODEL_DIR,f))}

def evaluate_model(model, X_te_s, y_te, threshold=0.5):
    y_prob = model.predict(X_te_s, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    rep = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_te, y_prob)
    return {
        "Accuracy":          round(rep["accuracy"],4),
        "Precision(Churn)":  round(rep["1"]["precision"],4),
        "Recall(Churn)":     round(rep["1"]["recall"],4),
        "F1(Churn)":         round(rep["1"]["f1-score"],4),
        "ROC-AUC":           round(auc,4),
    }, y_prob, y_pred

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 ANN Churn Lab")
    st.markdown("---")
    df_raw = load_raw_data()
    if df_raw is not None:
        st.success(f"✅ Dataset  {df_raw.shape[0]:,} rows · {df_raw.shape[1]} cols")
    else:
        st.error(f"❌ `{DATA_FILE}` not found")

    avail = get_available_models()
    st.markdown(f"### 💾 Models: {len(avail)}/{len(MODEL_FILES)}")

    # Phase A group
    st.markdown("<span style='color:#ffffff'>**Phase A — No Imbalance**</span>", unsafe_allow_html=True)
    for k in PHASE_A_KEYS:
        icon = "✅" if k in avail else "❌"
        st.markdown(f"<span style='font-size:11px;color:#ffffff'>{icon} {k}</span>", unsafe_allow_html=True)

    st.markdown("<span style='color:#ffffff'>**Phase B — Imbalance Handled**</span>", unsafe_allow_html=True)
    for k in PHASE_B_KEYS:
        icon = "✅" if k in avail else "❌"
        st.markdown(f"<span style='font-size:11px;color:#ffffff'>{icon} {k}</span>", unsafe_allow_html=True)

    st.markdown("---")
    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)
    st.markdown("---")
    st.markdown("""
    **📖 Quick Start**
    1. Run `ann_churn_modelling.py` to train & save models
    2. Place `Churn_Modeling.csv` here
    3. Explore all tabs!

    **📁 Required**
    - `Churn_Modeling.csv`
    - `saved_models/*.keras`
    - `saved_models/scaler.pkl`
    """)

# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='text-align:center;letter-spacing:3px;'>
🏦 Bank Customer Churn — ANN Deep Learning Lab
</h1>
<p style='text-align:center;color:#4a7ab5;font-size:14px;margin-top:-8px;'>
17 Models · EDA · Gradient Descents · Weight Init · Imbalance Handling · Keras Tuner · Live Predictor
</p>
<hr style='border-color:rgba(80,160,255,.2);margin:8px 0 20px 0;'>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🌌 ANN Viz Lab",
    "📊 EDA Dashboard",
    "🤖 Phase A — 11 Models",
    "⚖️ Phase B — Imbalance",
    "🔧 Keras Tuner",
    "💾 Save & Load",
    "🏆 Grand Comparison",
    "🔮 Live Predictor",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — ANN VIZ LAB  (Galaxy + MLP labs fully embedded, no external files)
# ══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("🌌 ANN Interactive Visualization Lab")

    # ── Galaxy Lab ─────────────────────────────────────────────────────
    st.subheader("🌌 ANN Galaxy Lab — Neural Network Tour")
    galaxy_html = """<!DOCTYPE html>
<html><head><style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#000;overflow:hidden;font-family:'Courier New',monospace}
canvas{display:block}
#guide-panel{position:absolute;bottom:20px;left:50%;transform:translateX(-50%);
  background:rgba(0,5,20,.85);border:1px solid rgba(80,160,255,.3);border-radius:12px;
  padding:14px 22px;color:#a0c8ff;font-size:12px;max-width:520px;width:90%;text-align:center;
  pointer-events:auto;box-shadow:0 0 30px rgba(40,100,255,.15)}
#guide-title{font-size:10px;color:#4a90d9;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px}
#guide-text{line-height:1.6;color:#c8e0ff;font-size:12px}
#nav-buttons{display:flex;gap:10px;justify-content:center;margin-top:10px}
.nav-btn{background:rgba(40,80,180,.3);border:1px solid rgba(80,160,255,.4);border-radius:6px;
  color:#90c0ff;padding:5px 14px;font-size:11px;cursor:pointer;font-family:'Courier New',monospace;transition:all .2s}
.nav-btn:hover{background:rgba(60,120,255,.3);color:#fff}
.nav-btn.active{background:rgba(80,150,255,.4);color:#fff;border-color:#80b0ff}
#tooltip{position:absolute;background:rgba(0,5,25,.92);border:1px solid rgba(100,180,255,.4);
  border-radius:8px;padding:8px 12px;color:#c0dcff;font-size:11px;pointer-events:none;
  max-width:200px;display:none;z-index:100;line-height:1.5}
#title-bar{position:absolute;top:14px;left:50%;transform:translateX(-50%);text-align:center;pointer-events:none}
#main-title{font-size:16px;font-weight:bold;color:#80c0ff;letter-spacing:3px;text-shadow:0 0 20px rgba(100,160,255,.6)}
#sub-title{font-size:10px;color:#4a7ab0;letter-spacing:2px;margin-top:2px}
#step-indicator{position:absolute;top:14px;right:16px;color:#3a6090;font-size:10px;letter-spacing:1px}
#legend{position:absolute;top:60px;left:14px;display:flex;flex-direction:column;gap:5px}
.legend-item{display:flex;align-items:center;gap:7px;font-size:10px;color:#5a8aaa}
.legend-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
</style></head><body>
<canvas id='c'></canvas>
<div style='position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none'>
  <div id='title-bar'>
    <div id='main-title'>⬡ ANN GALAXY LAB</div>
    <div id='sub-title'>ARTIFICIAL NEURAL NETWORK — INTERACTIVE TOUR</div>
  </div>
  <div id='step-indicator'></div>
  <div id='legend'>
    <div class='legend-item'><div class='legend-dot' style='background:#4488ff'></div>Input layer</div>
    <div class='legend-item'><div class='legend-dot' style='background:#aa44ff'></div>Hidden layers</div>
    <div class='legend-item'><div class='legend-dot' style='background:#ff6644'></div>Output layer</div>
    <div class='legend-item'><div class='legend-dot' style='background:#22ccaa'></div>Active signal</div>
  </div>
  <div id='guide-panel' style='pointer-events:auto'>
    <div id='guide-title'>◈ NEURAL GUIDE</div>
    <div id='guide-text'>Welcome to ANN Galaxy Lab. Click any neuron to fire it, or follow the guided tour.</div>
    <div id='nav-buttons'>
      <button class='nav-btn' onclick='prevStep()'>◀ Prev</button>
      <button class='nav-btn active' onclick='autoTour()'>▶ Auto Tour</button>
      <button class='nav-btn' onclick='nextStep()'>Next ▶</button>
    </div>
  </div>
  <div id='tooltip'></div>
</div>
<script>
const canvas=document.getElementById('c');const ctx=canvas.getContext('2d');
let W,H,cx,cy,time=0,stepIdx=0,autoMode=false,autoTimer=null;
let stars=[],neurons=[],connections=[],signals=[],hoveredNeuron=null,highlightLayer=null,highlightMode=null;
const LAYERS=[
  {name:'Input Layer',x:.12,count:4,color:'#4488ff',glow:'#2255cc'},
  {name:'Hidden 1',  x:.33,count:5,color:'#aa44ff',glow:'#6622aa'},
  {name:'Hidden 2',  x:.54,count:5,color:'#aa44ff',glow:'#6622aa'},
  {name:'Hidden 3',  x:.75,count:4,color:'#cc66ff',glow:'#8833cc'},
  {name:'Output',    x:.90,count:2,color:'#ff6644',glow:'#cc3322'},
];
const NEURON_INFO={
  input:{title:'Input Neuron',desc:'Receives raw data. Passes signal forward unchanged.'},
  hidden:{title:'Hidden Neuron',desc:'Applies weights×inputs+bias then activation. Learns patterns.'},
  output:{title:'Output Neuron',desc:'Final prediction. Trained via backpropagation.'},
};
const TOUR=[
  {title:'🌌 WELCOME TO ANN GALAXY LAB',text:'Each glowing star is a neuron. Light streams are signals flowing left→right. Click any neuron to explore it.',highlight:null},
  {title:'◈ INPUT LAYER',text:'Blue stars = Input Neurons. They receive raw features — CreditScore, Age, Balance. Each holds one value.',layer:0},
  {title:'◈ WEIGHTS & CONNECTIONS',text:'Glowing lines are weighted synapses. Each weight W amplifies or dampens the signal. Training adjusts these.',highlight:'connections'},
  {title:'◈ HIDDEN LAYER 1',text:'First hidden layer: neurons compute Σ(wᵢxᵢ)+b then ReLU activation. Detects low-level patterns.',layer:1},
  {title:'◈ HIDDEN LAYER 2',text:'Combines features from layer 1. Detects higher-level patterns — customer behavior clusters.',layer:2},
  {title:'◈ HIDDEN LAYER 3',text:'Even more abstract. Deep = hierarchical abstraction.',layer:3},
  {title:'◈ FORWARD PROPAGATION',text:'Watch signals travel left→right. Each neuron transforms the signal. This is a forward pass = one prediction.',highlight:'signal'},
  {title:'◈ OUTPUT LAYER',text:'Red star = Output Neuron. Sigmoid gives churn probability. Above 0.5 = predicted churn.',layer:4},
  {title:'◈ BACKPROPAGATION',text:'After prediction, loss flows backward (right→left). Gradients assign blame to each weight. This is learning.',highlight:'back'},
  {title:'⬡ FULL NETWORK',text:'Input → Hidden Layers → Output. Every churn prediction flows through this galaxy of neurons.',highlight:null},
];
function resize(){
  W=canvas.width=canvas.parentElement?canvas.parentElement.offsetWidth:780;
  H=canvas.height=480;cx=W/2;cy=H/2;buildNetwork();
}
function buildNetwork(){
  stars=[];
  for(let i=0;i<180;i++) stars.push({x:Math.random()*W,y:Math.random()*H,r:Math.random()*1.2+.2,a:Math.random(),sp:Math.random()*.003+.001,ph:Math.random()*Math.PI*2});
  neurons=[];connections=[];
  const netH=H*.55,netTop=H*.11;
  LAYERS.forEach((layer,li)=>{
    const lx=W*layer.x;layer.neurons=[];
    for(let i=0;i<layer.count;i++){
      const ny=netTop+(netH/(layer.count-1))*i;
      const n={x:lx,y:ny,r:9,baseR:9,layer:li,idx:i,color:layer.color,glow:layer.glow,
               pulse:Math.random()*Math.PI*2,active:0,
               activation:(Math.random()*.7+.2).toFixed(2),bias:((Math.random()-.5)*.5).toFixed(2),
               type:li===0?'input':li===LAYERS.length-1?'output':'hidden'};
      layer.neurons.push(n);neurons.push(n);
    }
  });
  LAYERS.forEach((layer,li)=>{
    if(li<LAYERS.length-1){
      const nxt=LAYERS[li+1];
      layer.neurons.forEach(n1=>nxt.neurons.forEach(n2=>connections.push({n1,n2,weight:((Math.random()*2-1)).toFixed(2),alpha:Math.random()*.2+.04})));
    }
  });
}
function spawnSignal(fwd=true){
  const src=fwd?LAYERS[0]:LAYERS[LAYERS.length-1];
  const n=src.neurons[Math.floor(Math.random()*src.neurons.length)];
  signals.push({neuron:n,layer:n.layer,progress:0,forward:fwd,color:fwd?'#22ccaa':'#ff9944',alpha:.9});
}
let hasSpoken = false;
function applyStep(s){
  highlightLayer=s.layer!==undefined?s.layer:null;highlightMode=s.highlight||null;
  document.getElementById('guide-title').textContent=s.title;
  document.getElementById('guide-text').textContent=s.text;
  document.getElementById('step-indicator').textContent=`STEP ${stepIdx+1}/${TOUR.length}`;
  if(highlightMode==='signal') for(let i=0;i<6;i++) setTimeout(()=>spawnSignal(true),i*200);
  if(highlightMode==='back')   for(let i=0;i<6;i++) setTimeout(()=>spawnSignal(false),i*200);
  // Speak the text
  if ('speechSynthesis' in window && hasSpoken) {
    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(s.text);
    speechSynthesis.speak(utterance);
  }
  hasSpoken = true;
}
function nextStep(){stepIdx=(stepIdx+1)%TOUR.length;applyStep(TOUR[stepIdx]);}
function prevStep(){stepIdx=(stepIdx-1+TOUR.length)%TOUR.length;applyStep(TOUR[stepIdx]);}
function autoTour(){
  autoMode=!autoMode;const btn=document.querySelectorAll('.nav-btn')[1];
  if(autoMode){btn.textContent='⏸ Pause';btn.classList.add('active');autoTimer=setInterval(nextStep,4000);}
  else{btn.textContent='▶ Auto Tour';clearInterval(autoTimer);}
}
function getNeuronAt(mx,my){
  for(const n of neurons){const dx=mx-n.x,dy=my-n.y;if(Math.sqrt(dx*dx+dy*dy)<n.r+12)return n;}return null;
}
canvas.addEventListener('mousemove',e=>{
  const r=canvas.getBoundingClientRect();const mx=e.clientX-r.left,my=e.clientY-r.top;
  hoveredNeuron=getNeuronAt(mx,my);const tip=document.getElementById('tooltip');
  if(hoveredNeuron){
    const info=NEURON_INFO[hoveredNeuron.type];
    tip.innerHTML=`<strong style='color:#80c0ff'>${info.title}</strong><br><span style='color:#5a8aaa'>Layer:${LAYERS[hoveredNeuron.layer].name}</span><br>${info.desc}<br><span style='color:#22ccaa;font-size:10px'>act:${hoveredNeuron.activation} bias:${hoveredNeuron.bias}</span>`;
    tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY-55)+'px';
    canvas.style.cursor='pointer';
  }else{tip.style.display='none';canvas.style.cursor='default';}
});
canvas.addEventListener('click',e=>{
  const r=canvas.getBoundingClientRect();const n=getNeuronAt(e.clientX-r.left,e.clientY-r.top);
  if(n){n.active=1.5;connections.filter(c=>c.n1===n).forEach(c=>signals.push({conn:c,progress:0,forward:true,color:'#22ccaa',alpha:1}));}
});
function draw(){
  time+=.016;ctx.clearRect(0,0,W,H);
  const bg=ctx.createRadialGradient(cx,cy,0,cx,cy,Math.max(W,H)*.7);
  bg.addColorStop(0,'#000814');bg.addColorStop(1,'#000305');
  ctx.fillStyle=bg;ctx.fillRect(0,0,W,H);
  stars.forEach(s=>{
    s.a=.3+.5*Math.abs(Math.sin(time*s.sp*30+s.ph));
    ctx.beginPath();ctx.arc(s.x,s.y,s.r,0,Math.PI*2);
    ctx.fillStyle=`rgba(160,200,255,${s.a*.7})`;ctx.fill();
  });
  connections.forEach(conn=>{
    const{n1,n2}=conn;
    const isHL=(highlightLayer!==null&&(n1.layer===highlightLayer||n2.layer===highlightLayer));
    const isFd=highlightLayer!==null&&!isHL&&highlightMode!=='connections';
    let alpha=isFd?.02:(highlightMode==='connections'?.22:conn.alpha);
    if(hoveredNeuron&&(n1===hoveredNeuron||n2===hoveredNeuron))alpha=.7;
    ctx.beginPath();ctx.moveTo(n1.x,n1.y);ctx.lineTo(n2.x,n2.y);
    ctx.strokeStyle=n1.color+Math.floor(alpha*255).toString(16).padStart(2,'0');
    ctx.lineWidth=.6;ctx.stroke();
  });
  signals=signals.filter(s=>{
    if(s.conn){
      s.progress+=.025;
      if(s.progress>=1){s.conn.n2.active=Math.max(s.conn.n2.active,1);return false;}
      const{n1,n2}=s.conn;const sx=n1.x+(n2.x-n1.x)*s.progress,sy=n1.y+(n2.y-n1.y)*s.progress;
      const trail=ctx.createRadialGradient(sx,sy,0,sx,sy,5);
      trail.addColorStop(0,`rgba(34,204,170,${s.alpha})`);trail.addColorStop(1,'rgba(34,204,170,0)');
      ctx.fillStyle=trail;ctx.beginPath();ctx.arc(sx,sy,5,0,Math.PI*2);ctx.fill();return true;
    }
    s.progress+=.02;
    if(s.progress>=1){
      const nextLi=s.forward?s.layer+1:s.layer-1;
      if(nextLi>=0&&nextLi<LAYERS.length){
        const nl=LAYERS[nextLi];const nn=nl.neurons[Math.floor(Math.random()*nl.neurons.length)];
        nn.active=Math.max(nn.active,1);
        const conn=s.forward?connections.find(c=>c.n1===s.neuron&&c.n2===nn):connections.find(c=>c.n2===s.neuron&&c.n1===nn);
        if(conn)signals.push({conn,progress:0,forward:s.forward,color:s.color,alpha:.9});
        if(nextLi<LAYERS.length-1&&nextLi>0)signals.push({neuron:nn,layer:nextLi,progress:0,forward:s.forward,color:s.color,alpha:.9});
      }return false;
    }return true;
  });
  neurons.forEach(n=>{
    const isHL=highlightLayer!==null&&n.layer===highlightLayer;
    const isFd=highlightLayer!==null&&!isHL&&highlightMode!=='connections'&&highlightMode!=='signal'&&highlightMode!=='back';
    n.active=Math.max(0,n.active-.015);n.pulse+=.02;
    const ps=1+.12*Math.sin(n.pulse)+n.active*.5,r=n.baseR*ps*(n===hoveredNeuron?1.4:1);
    const fd=isFd?.15:1;
    const glR=r*(3+n.active*2);
    if(!isFd){
      const grd=ctx.createRadialGradient(n.x,n.y,r*.5,n.x,n.y,glR);
      const gc=n.active>.3?'34,204,170':n.color==='#4488ff'?'40,100,255':n.color==='#ff6644'?'255,100,60':'140,60,255';
      grd.addColorStop(0,`rgba(${gc},${(.35+n.active*.4)*fd})`);grd.addColorStop(1,'rgba(0,0,0,0)');
      ctx.fillStyle=grd;ctx.beginPath();ctx.arc(n.x,n.y,glR,0,Math.PI*2);ctx.fill();
    }
    const bg2=ctx.createRadialGradient(n.x-r*.3,n.y-r*.3,0,n.x,n.y,r);
    const bc=n.active>.3?'#22ccaa':n.color;
    bg2.addColorStop(0,bc+'ff');bg2.addColorStop(.6,bc+'bb');bg2.addColorStop(1,n.glow+'88');
    ctx.globalAlpha=fd;ctx.fillStyle=bg2;ctx.beginPath();ctx.arc(n.x,n.y,r,0,Math.PI*2);ctx.fill();
    if(n===hoveredNeuron||n.active>.3){ctx.strokeStyle=n.active>.3?'rgba(34,204,170,.9)':'rgba(255,255,255,.5)';ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(n.x,n.y,r+3,0,Math.PI*2);ctx.stroke();}
    ctx.globalAlpha=1;
  });
  LAYERS.forEach((layer,li)=>{
    const lx=W*layer.x;const topY=Math.min(...layer.neurons.map(n=>n.y))-22;
    const isHL=highlightLayer===li;const isFd=highlightLayer!==null&&!isHL;
    ctx.fillStyle=isFd?'rgba(60,100,150,.3)':isHL?'#ffffff':'rgba(120,180,255,.7)';
    ctx.font=`${isHL?'bold ':' '}10px 'Courier New'`;ctx.textAlign='center';ctx.fillText(layer.name,lx,topY);
  });
  if(Math.random()<.012)spawnSignal(true);
  requestAnimationFrame(draw);
}
window.addEventListener('resize',resize);resize();draw();applyStep(TOUR[0]);
</script></body></html>"""

    st.components.v1.html(galaxy_html, height=520, scrolling=False)
    st.markdown("---")

    # ── Architecture diagram ───────────────────────────────────────────
    st.subheader("📐 ANN Architecture — Churn Model")
    fig, ax = _dark_fig(figsize=(14, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    layer_config = [
        ("Input\n(11 features)", 11, 1.0, "#1a3a6a", "steelblue"),
        ("Hidden 1\n(64 ReLU)",  8,  2.8, "#1a5a4a", "#27ae60"),
        ("Dropout\n0.30",         8,  4.2, "#4a2a1a", "#e67e22"),
        ("Hidden 2\n(32 ReLU)",  6,  5.6, "#1a5a4a", "#27ae60"),
        ("Dropout\n0.30",         6,  7.0, "#4a2a1a", "#e67e22"),
        ("Hidden 3\n(16 ReLU)",  4,  8.2, "#1a5a4a", "#27ae60"),
        ("Output\n(1 Sigmoid)",  1,  9.5, "#4a1a5a", "#9b59b6"),
    ]
    node_pos = []
    for (lbl, n, xp, bg_col, ec) in layer_config:
        dn = min(n, 6); gap = 7.0 / (dn + 1)
        ys = [gap * (i + 1) + 0.5 for i in range(dn)]
        node_pos.append((xp, ys, ec))
        for y in ys:
            circ = plt.Circle((xp, y), 0.27, color=bg_col, ec=ec, lw=1.5, zorder=3)
            ax.add_patch(circ)
        ax.text(xp, 0.05, lbl, ha="center", va="bottom", fontsize=8,
                color="white", fontfamily="monospace")
    for i in range(len(node_pos) - 1):
        x1, ys1, _ = node_pos[i]; x2, ys2, _ = node_pos[i + 1]
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1, x2], [y1, y2], color="#1e3a6a", lw=0.4, alpha=0.5, zorder=1)
    ax.set_title("ANN Architecture — Bank Churn (Phase B Models)",
                 color="#80c0ff", fontsize=12, fontweight="bold", pad=10)
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("📊 EDA Dashboard")
    if df_raw is None:
        st.error(f"Dataset `{DATA_FILE}` not found.")
        st.stop()

    # KPIs
    total = len(df_raw); churned = int(df_raw["Exited"].sum())
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total Customers",   f"{total:,}")
    k2.metric("Churned",            f"{churned:,}")
    k3.metric("Churn Rate",         f"{churned/total*100:.1f}%")
    k4.metric("Avg Balance",        f"${df_raw['Balance'].mean():,.0f}")
    k5.metric("Avg Age",            f"{df_raw['Age'].mean():.1f} yrs")
    st.markdown("---")

    # Row 1
    c1,c2,c3 = st.columns(3)
    with c1:
        fig,ax = _dark_fig(figsize=(5,4))
        counts = df_raw["Exited"].value_counts()
        ax.pie(counts.values, labels=["Stayed","Churned"],
               colors=["#3498db","#e74c3c"], autopct="%1.1f%%", startangle=90,
               wedgeprops={"edgecolor":"#0d1530","linewidth":2},
               textprops={"color":"#c0d0e0","fontsize":10})
        ax.set_title("Target Distribution", color="#80c0ff", fontweight="bold")
        st.pyplot(fig)
    with c2:
        fig,ax = _dark_fig(figsize=(5,4))
        geo = df_raw.groupby("Geography")["Exited"].mean()*100
        bars = ax.bar(geo.index, geo.values, color=["#e74c3c","#f39c12","#27ae60"])
        ax.set_title("Churn Rate by Geography (%)", color="#80c0ff", fontweight="bold")
        ax.set_ylabel("Churn Rate %", color="#80a0c0")
        for bar,v in zip(bars, geo.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.3, f"{v:.1f}%", ha="center", color="#c0d0e0", fontsize=9)
        st.pyplot(fig)
    with c3:
        fig,ax = _dark_fig(figsize=(5,4))
        gen = df_raw.groupby("Gender")["Exited"].mean()*100
        ax.bar(gen.index, gen.values, color=["#3498db","#e91e63"])
        ax.set_title("Churn Rate by Gender (%)", color="#80c0ff", fontweight="bold")
        ax.set_ylabel("Churn Rate %", color="#80a0c0")
        for i,v in enumerate(gen.values):
            ax.text(i, v+.3, f"{v:.1f}%", ha="center", color="#c0d0e0", fontsize=10)
        st.pyplot(fig)

    # Row 2
    c4,c5,c6 = st.columns(3)
    with c4:
        fig,ax = _dark_fig(figsize=(5,4))
        df_raw[df_raw["Exited"]==0]["Age"].hist(bins=25, alpha=.7, label="Stayed",  color="#3498db", ax=ax)
        df_raw[df_raw["Exited"]==1]["Age"].hist(bins=25, alpha=.7, label="Churned", color="#e74c3c", ax=ax)
        ax.set_title("Age Distribution by Churn", color="#80c0ff", fontweight="bold")
        ax.legend(facecolor="#0d1530", labelcolor="#c0d0e0")
        st.pyplot(fig)
    with c5:
        fig,ax = _dark_fig(figsize=(5,4))
        df_raw[df_raw["Exited"]==0]["Balance"].hist(bins=25, alpha=.7, label="Stayed",  color="#3498db", ax=ax)
        df_raw[df_raw["Exited"]==1]["Balance"].hist(bins=25, alpha=.7, label="Churned", color="#e74c3c", ax=ax)
        ax.set_title("Balance Distribution by Churn", color="#80c0ff", fontweight="bold")
        ax.legend(facecolor="#0d1530", labelcolor="#c0d0e0")
        st.pyplot(fig)
    with c6:
        fig,ax = _dark_fig(figsize=(5,4))
        sns.countplot(x="NumOfProducts", hue="Exited", data=df_raw,
                      palette=["#3498db","#e74c3c"], ax=ax)
        ax.set_title("Num Products vs Churn", color="#80c0ff", fontweight="bold")
        ax.legend(["Stayed","Churned"], facecolor="#0d1530", labelcolor="#c0d0e0")
        ax.set_facecolor("#0a0f1e")
        st.pyplot(fig)

    # Row 3
    c7,c8,c9 = st.columns(3)
    with c7:
        fig,ax = _dark_fig(figsize=(5,4))
        tenure_churn = df_raw.groupby("Tenure")["Exited"].mean()*100
        ax.plot(tenure_churn.index, tenure_churn.values, "o-", color="#f39c12", lw=2.5, markersize=6)
        ax.fill_between(tenure_churn.index, tenure_churn.values, alpha=.15, color="#f39c12")
        ax.set_title("Churn Rate by Tenure", color="#80c0ff", fontweight="bold")
        ax.set_xlabel("Tenure (years)", color="#80a0c0")
        ax.grid(alpha=.15, color="#3a6aaa")
        st.pyplot(fig)
    with c8:
        fig,ax = _dark_fig(figsize=(5,4))
        act = df_raw.groupby("IsActiveMember")["Exited"].mean()*100
        ax.bar(["Inactive","Active"], act.values, color=["#e74c3c","#27ae60"])
        ax.set_title("Active Member vs Churn %", color="#80c0ff", fontweight="bold")
        for i,v in enumerate(act.values):
            ax.text(i, v+.3, f"{v:.1f}%", ha="center", color="#c0d0e0", fontsize=11, fontweight="bold")
        st.pyplot(fig)
    with c9:
        fig,ax = _dark_fig(figsize=(5,4))
        cr = df_raw.groupby("HasCrCard")["Exited"].mean()*100
        ax.bar(["No Card","Has Card"], cr.values, color=["#9b59b6","#3498db"])
        ax.set_title("Credit Card vs Churn %", color="#80c0ff", fontweight="bold")
        for i,v in enumerate(cr.values):
            ax.text(i, v+.3, f"{v:.1f}%", ha="center", color="#c0d0e0", fontsize=11, fontweight="bold")
        st.pyplot(fig)

    # Correlation heatmap
    st.subheader("🔗 Correlation Matrix")
    fig,ax = _dark_fig(figsize=(12,7))
    num = df_raw.select_dtypes(include=[np.number])
    mask = np.triu(np.ones_like(num.corr(), dtype=bool))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="RdYlBu_r",
                linewidths=.5, mask=mask, vmin=-1, vmax=1, ax=ax, annot_kws={"size":8})
    ax.set_title("Correlation Matrix (Lower Triangle)", color="#80c0ff", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#80a0c0")
    st.pyplot(fig)

    # Skewness
    st.subheader("📐 Skewness Analysis")
    num_cols = df_raw.select_dtypes(include=[np.number]).columns
    skew = df_raw[num_cols].skew().sort_values()
    fig,ax = _dark_fig(figsize=(10,4))
    colors_sk = ["#e74c3c" if abs(v)>1 else "#27ae60" for v in skew.values]
    ax.barh(skew.index, skew.values, color=colors_sk)
    ax.axvline(x=1,  color="yellow", ls="--", alpha=.6, label="|skew|>1 → log1p")
    ax.axvline(x=-1, color="yellow", ls="--", alpha=.6)
    ax.set_title("Feature Skewness (Red = log1p applied)", color="#80c0ff", fontweight="bold")
    ax.legend(facecolor="#0d1530", labelcolor="#c0d0e0")
    st.pyplot(fig)
    st.caption("Features with |skew| > 1 have log1p transformation applied in preprocessing.")

    # Pairplot key features
    st.subheader("🔍 Key Feature Pairplot")
    key_cols = ["Age","Balance","CreditScore","EstimatedSalary","Exited"]
    fig = plt.figure(figsize=(10,8))
    fig.patch.set_facecolor("#0d1530")
    g = sns.pairplot(df_raw[key_cols], hue="Exited", palette=["#3498db","#e74c3c"],
                     diag_kind="kde", plot_kws={"alpha":.35,"s":15})
    g.fig.patch.set_facecolor("#0d1530")
    g.fig.suptitle("Pairplot — Key Features vs Churn", y=1.02, fontsize=11, color="#80c0ff", fontweight="bold")
    st.pyplot(g.fig)

    # Raw data explorer
    st.subheader("🔍 Raw Data Explorer")
    n_rows = st.slider("Rows to show:", 5, 100, 20, key="eda_rows")
    st.dataframe(df_raw.head(n_rows), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — PHASE A (11 MODELS)
# ══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("🤖 Phase A — 11 Models Without Imbalance Handling")

    # Model descriptions
    model_desc = {
        "A1 — Batch GD (SGD)":          ("Batch GD", "SGD with batch_size=len(X_train). Full dataset per step. Slow but stable.",            "#1a3a6a"),
        "A2 — Stochastic GD (SGD)":     ("Stochastic GD", "SGD with batch_size=1. One sample per step. Noisy but escapes local minima.",      "#1a4a2a"),
        "A3 — Mini-Batch Adam":         ("Mini-Batch Adam", "Adam with batch_size=32. Best of both worlds — fast & adaptive.",                 "#2a1a4a"),
        "A4 — Mini-Batch RMSprop":      ("Mini-Batch RMSprop", "RMSprop with batch_size=32. Adaptive LR, good for non-stationary problems.",  "#3a2a1a"),
        "A5 — Early Stopping":          ("Early Stopping", "patience=10, restore_best_weights. Stops before overfitting.",                    "#1a3a4a"),
        "A6 — Dropout (0.3+0.2)":       ("Dropout", "Dropout(0.3) + Dropout(0.2). Randomly mutes neurons to prevent memorization.",          "#3a1a3a"),
        "A7 — Glorot Uniform (Xavier)": ("Glorot Uniform", "Xavier init. Designed for sigmoid/tanh. Balances variance across layers.",        "#1a4a1a"),
        "A8 — He Normal (Kaiming)":     ("He Normal", "Best for ReLU. Accounts for the 'dying ReLU' variance problem.",                       "#4a1a1a"),
        "A9 — He Uniform":              ("He Uniform", "Uniform variant of He init. Good for ReLU family activations.",                       "#2a3a1a"),
        "A10 — Random Normal":          ("Random Normal", "Baseline init. Mean=0, Std=0.05. No special variance scaling.",                    "#3a3a1a"),
        "A11 — Keras Tuner (Phase A)":  ("Keras Tuner", "Best HP from 15-trial RandomSearch on raw imbalanced data.",                        "#1a2a4a"),
    }

    # Cards
    cols = st.columns(3)
    for i, (name, (short, desc, bg)) in enumerate(model_desc.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='model-card' style='background:{bg}40;'>
              <b style='color:#80c0ff;'>{name}</b><br>
              <small style='color:#8090b0;'>{desc}</small>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if df_raw is None:
        st.error("Dataset not found.")
    else:
        data = preprocess_data(df_raw)
        X, y, X_tr_s, X_te_s, y_tr, y_te, sc, feat_names = data
        avail_a = {k: v for k,v in get_available_models().items() if k.startswith("A")}

        if not avail_a:
            st.warning("No Phase A models found. Run `ann_churn_modelling.py` first.")
        else:
            res_a = []
            failed_a = []
            for name, path in avail_a.items():
                m = load_keras_model(path)
                if m:
                    met, _, _ = evaluate_model(m, X_te_s, y_te, threshold)
                    res_a.append({"Model": name, **met})
                else:
                    failed_a.append(name)

            if res_a:
                df_a = pd.DataFrame(res_a).set_index("Model")
                st.subheader("📋 Phase A — Results Table")
                st.dataframe(df_a.style.highlight_max(
                    subset=["Accuracy","Precision(Churn)","Recall(Churn)","F1(Churn)","ROC-AUC"],
                    color="#1a5a1a"), use_container_width=True)

                st.subheader("📊 Phase A — Metric Charts")
                fig, axes = _dark_fig(2, 3, figsize=(18, 9))
                metrics_show = ["Accuracy","Precision(Churn)","Recall(Churn)","F1(Churn)","ROC-AUC"]
                colors_a = ["#3498db","#27ae60","#e74c3c","#f39c12","#9b59b6"]
                for ax, metric, color in zip(axes.flatten()[:5], metrics_show, colors_a):
                    bars = ax.barh(df_a.index, df_a[metric], color=color)
                    ax.set_title(metric, color="#80c0ff", fontweight="bold")
                    ax.set_xlim(0, 1.12)
                    for bar in bars:
                        w = bar.get_width()
                        ax.text(w+.005, bar.get_y()+bar.get_height()/2, f"{w:.4f}",
                                va="center", fontsize=7, color="#c0d0e0")
                axes.flatten()[-1].axis("off")
                plt.suptitle("Phase A — 11 ANN Models Without Imbalance Handling",
                             color="#80c0ff", fontsize=12, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig)

                # Deep-dive
                st.subheader("🔬 Model Deep-Dive")
                sel_a = st.selectbox("Select Phase A model:", list(avail_a.keys()), key="deep_a")
                if sel_a:
                    m = load_keras_model(avail_a[sel_a])
                    if m:
                        met, y_prob, y_pred = evaluate_model(m, X_te_s, y_te, threshold)
                        c1,c2,c3,c4,c5 = st.columns(5)
                        for col, (k,v) in zip([c1,c2,c3,c4,c5], met.items()):
                            col.metric(k, f"{v:.4f}")
                        fpr, tpr, _ = roc_curve(y_te, y_prob)
                        cm = confusion_matrix(y_te, y_pred)
                        fig, axes = _dark_fig(1, 2, figsize=(12, 4))
                        ConfusionMatrixDisplay(cm, display_labels=["Stayed","Churned"]).plot(
                            ax=axes[0], cmap="Blues", colorbar=False)
                        axes[0].set_title(f"Confusion Matrix — {sel_a}", color="#80c0ff", fontsize=9)
                        axes[0].tick_params(colors="#80a0c0")
                        axes[1].plot(fpr, tpr, color="#e74c3c", lw=2.5, label=f"AUC={met['ROC-AUC']:.3f}")
                        axes[1].plot([0,1],[0,1],"--", color="#4a7ab5", alpha=.5)
                        axes[1].set_title("ROC Curve", color="#80c0ff")
                        axes[1].set_xlabel("FPR", color="#80a0c0"); axes[1].set_ylabel("TPR", color="#80a0c0")
                        axes[1].legend(facecolor="#0a0f1e", labelcolor="#c0d0e0")
                        plt.tight_layout(); st.pyplot(fig)
            elif failed_a:
                st.error(
                    "Phase A models were found but failed to load. "
                    "Check the terminal logs for TensorFlow/model compatibility issues."
                )

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — PHASE B (6 MODELS)
# ══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("⚖️ Phase B — 6 Models With Imbalance Handling")
    #st.info("ℹ️ SMOTE + Class Weight combined is intentionally excluded per project requirements.")

    # Technique cards
    tech_cards = [
        ("⚖️ Class Weights",      "#1a3a6a", "Penalises churn misclassification ~4× in the loss. No data change."),
        ("🔬 SMOTE",              "#1a4a2a", "Generates synthetic minority samples by interpolating between real ones."),
        ("📈 Random Oversample",  "#3a2a1a", "Duplicates existing minority rows until classes balance."),
        ("📉 Random Undersample", "#3a1a1a", "Removes majority rows randomly. Smaller training set."),
        ("🔀 SMOTEENN",           "#2a1a3a", "SMOTE + Edited Nearest Neighbours cleaning. Balanced AND cleaner."),
        ("🔧 Keras Tuner B",      "#1a2a3a", "Best HP from 15-trial RandomSearch on the top Phase B technique."),
    ]
    cols_cards = st.columns(3)
    for i, (title, bg, desc) in enumerate(tech_cards):
        with cols_cards[i % 3]:
            st.markdown(f"""
            <div class='model-card' style='background:{bg}40;'>
              <b style='color:#80c0ff;'>{title}</b><br>
              <small style='color:#8090b0;'>{desc}</small>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if df_raw is None:
        st.error("Dataset not found.")
    else:
        data = preprocess_data(df_raw)
        X, y, X_tr_s, X_te_s, y_tr, y_te, sc, feat_names = data
        avail_b = {k: v for k,v in get_available_models().items() if k.startswith("B")}

        if not avail_b:
            st.warning("No Phase B models found. Run `ann_churn_modelling.py` first.")
        else:
            res_b = []
            failed_b = []
            for name, path in avail_b.items():
                m = load_keras_model(path)
                if m:
                    met, _, _ = evaluate_model(m, X_te_s, y_te, threshold)
                    res_b.append({"Model": name, **met})
                else:
                    failed_b.append(name)

            if res_b:
                df_b = pd.DataFrame(res_b).set_index("Model")
                st.subheader("📋 Phase B — Results Table")
                st.dataframe(df_b.style.highlight_max(
                    subset=["Accuracy","Precision(Churn)","Recall(Churn)","F1(Churn)","ROC-AUC"],
                    color="#1a5a1a"), use_container_width=True)

                st.subheader("📊 Phase B — Metric Charts")
                fig, axes = _dark_fig(2, 3, figsize=(18, 9))
                metrics_show = ["Accuracy","Precision(Churn)","Recall(Churn)","F1(Churn)","ROC-AUC"]
                colors_b = ["#e74c3c","#27ae60","#f39c12","#e91e63","#9b59b6"]
                for ax, metric, color in zip(axes.flatten()[:5], metrics_show, colors_b):
                    bars = ax.barh(df_b.index, df_b[metric], color=color)
                    ax.set_title(metric, color="#80c0ff", fontweight="bold")
                    ax.set_xlim(0, 1.12)
                    for bar in bars:
                        w = bar.get_width()
                        ax.text(w+.005, bar.get_y()+bar.get_height()/2, f"{w:.4f}",
                                va="center", fontsize=8, color="#c0d0e0")
                axes.flatten()[-1].axis("off")
                plt.suptitle("Phase B — 6 ANN Models With Imbalance Handling",
                             color="#80c0ff", fontsize=12, fontweight="bold")
                plt.tight_layout(); st.pyplot(fig)

                # Deep-dive
                st.subheader("🔬 Model Deep-Dive")
                sel_b = st.selectbox("Select Phase B model:", list(avail_b.keys()), key="deep_b")
                if sel_b:
                    m = load_keras_model(avail_b[sel_b])
                    if m:
                        met, y_prob, y_pred = evaluate_model(m, X_te_s, y_te, threshold)
                        c1,c2,c3,c4,c5 = st.columns(5)
                        for col, (k,v) in zip([c1,c2,c3,c4,c5], met.items()):
                            col.metric(k, f"{v:.4f}")
                        fpr, tpr, _ = roc_curve(y_te, y_prob)
                        cm = confusion_matrix(y_te, y_pred)
                        fig, axes = _dark_fig(1, 2, figsize=(12, 4))
                        ConfusionMatrixDisplay(cm, display_labels=["Stayed","Churned"]).plot(
                            ax=axes[0], cmap="Reds", colorbar=False)
                        axes[0].set_title(f"Confusion Matrix — {sel_b}", color="#80c0ff", fontsize=9)
                        axes[0].tick_params(colors="#80a0c0")
                        axes[1].plot(fpr, tpr, color="#27ae60", lw=2.5, label=f"AUC={met['ROC-AUC']:.3f}")
                        axes[1].plot([0,1],[0,1],"--", color="#4a7ab5", alpha=.5)
                        axes[1].set_title("ROC Curve", color="#80c0ff")
                        axes[1].set_xlabel("FPR", color="#80a0c0"); axes[1].set_ylabel("TPR", color="#80a0c0")
                        axes[1].legend(facecolor="#0a0f1e", labelcolor="#c0d0e0")
                        plt.tight_layout(); st.pyplot(fig)
            elif failed_b:
                st.error(
                    "Phase B models were found but failed to load. "
                    "Check the terminal logs for TensorFlow/model compatibility issues."
                )

# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — KERAS TUNER
# ══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("🔧 Keras Tuner — Hyperparameter Tuning")

    col1, col2 = st.columns(2)
    def hp_table(title, border_color, data_rows):
        rows_html = "".join(
            f"<tr{'  style=background:rgba(255,255,255,.03)' if i%2 else ''}>"
            f"<td style='padding:6px 10px;'>{r[0]}</td>"
            f"<td style='padding:6px 10px;color:#22ccaa;'>{r[1]}</td></tr>"
            for i,r in enumerate(data_rows)
        )
        return f"""
        <div style="background:#0d1a2a;border:1px solid {border_color};
                    border-radius:10px;padding:14px;font-size:12px;color:#c0d8f0;">
          <div style="color:{border_color};font-weight:bold;margin-bottom:8px;">{title}</div>
          <table style="width:100%;border-collapse:collapse;">
            <tr style="border-bottom:1px solid {border_color}40;">
              <th style="text-align:left;padding:5px 10px;color:{border_color};">HP</th>
              <th style="text-align:left;padding:5px 10px;color:{border_color};">Search Space</th>
            </tr>{rows_html}
          </table></div>"""

    HP_ROWS = [
        ("Hidden Layers", "1 · 2 · 3"),
        ("Neurons/Layer",  "16 · 32 · 64 · 128"),
        ("Activation",     "relu · tanh"),
        ("Dropout Rate",   "0.0 → 0.5 (step 0.1)"),
        ("Learning Rate",  "0.001 · 0.0005 · 0.0001"),
        ("Optimizer",      "Adam · RMSprop"),
        ("Objective",      "val_accuracy"),
        ("Max Trials",     "15"),
    ]
    with col1:
        st.subheader("🔷 Phase A — Raw Data")
        st.markdown(hp_table("Phase A Tuning Space","rgba(80,160,255,.6)",
                             HP_ROWS+[("Data","Raw (imbalanced)")]), unsafe_allow_html=True)
    with col2:
        st.subheader("🔶 Phase B — Best Resampled")
        st.markdown(hp_table("Phase B Tuning Space","rgba(180,80,255,.6)",
                             HP_ROWS+[("Data","Best B1–B5 technique")]), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🗺️ Search Space Visualized")
    fig, axes = _dark_fig(1, 3, figsize=(16, 5))
    layers=[1,2,3]; units=[16,32,64,128]
    combos=[[l*u for u in units] for l in layers]
    bottoms=np.zeros(len(units))
    for l_idx,(l,cmbs) in enumerate(zip(layers,combos)):
        axes[0].bar([f"{u}n" for u in units], cmbs, bottom=bottoms, label=f"{l} layer(s)", alpha=.8)
        bottoms+=cmbs
    axes[0].set_title("Layer × Neuron Combinations", color="#80c0ff", fontweight="bold")
    axes[0].legend(facecolor="#0a0f1e", labelcolor="#c0d0e0", fontsize=8)

    lrs=[0.001,0.0005,0.0001]
    axes[1].barh(["0.001 (Adam default)","0.0005 (Half)","0.0001 (Slow)"], lrs,
                 color=["#3498db","#27ae60","#e74c3c"])
    axes[1].set_title("Learning Rate Options", color="#80c0ff", fontweight="bold")

    drops=[0.0,0.1,0.2,0.3,0.4,0.5]
    axes[2].bar([str(d) for d in drops], drops, color="#9b59b6")
    axes[2].axhline(.3, color="#e74c3c", ls="--", alpha=.6, label="Default used")
    axes[2].set_title("Dropout Rate Options", color="#80c0ff", fontweight="bold")
    axes[2].legend(facecolor="#0a0f1e", labelcolor="#c0d0e0")
    plt.tight_layout(); st.pyplot(fig)

    st.subheader("📈 How KerasTuner Works")
    st.markdown("""
    | Step | What Happens |
    |------|-------------|
    | 1 | Samples a random HP combination from the search space |
    | 2 | Builds & trains the model up to 50 epochs with EarlyStopping(patience=8) |
    | 3 | Records `val_accuracy` |
    | 4 | Repeats for 15 trials |
    | 5 | Selects the trial with highest `val_accuracy` |
    | 6 | Rebuilds best model, trains fully up to 100 epochs with EarlyStopping(patience=10) |
    | 7 | Saves as `A11_KerasTuner.keras` or `B6_KerasTuner.keras` |
    """)

    col_ta, col_tb = st.columns(2)
    with col_ta:
        st.markdown("**Phase A Tuner**")
        if os.path.exists("kt_phase_a"):
            st.success("✅ `kt_phase_a/` found")
        else:
            st.warning("Run script to generate tuner results.")
    with col_tb:
        st.markdown("**Phase B Tuner**")
        if os.path.exists("kt_phase_b"):
            st.success("✅ `kt_phase_b/` found")
        else:
            st.warning("Run script to generate tuner results.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — SAVE & LOAD
# ══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("💾 Save & Load Models")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📥 Saving")
        st.markdown("""
        <div style="background:#0d1a2a;border:1px solid rgba(80,160,255,.25);
                    border-radius:10px;padding:16px;font-size:12px;color:#c0d8f0;">
          <div style="color:#80c0ff;font-weight:bold;margin-bottom:8px;">How models are saved:</div>
          <p>📦 <b style='color:#22ccaa'>Format:</b> <code>.keras</code> (TF 2.x recommended)</p>
          <p>📂 <b style='color:#22ccaa'>Location:</b> <code>saved_models/</code></p>
          <p>🔢 <b style='color:#22ccaa'>Total models:</b> 17 (A1–A11 + B1–B6)</p>
          <p>⚖️ <b style='color:#22ccaa'>Scaler:</b> <code>scaler.pkl</code></p>
          <p>🏷️ <b style='color:#22ccaa'>Features:</b> <code>feature_names.pkl</code></p>
          <p>💡 <b style='color:#f39c12'>Tip:</b> ModelCheckpoint saves best epoch automatically</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.subheader("📤 Loading & Inference")
        st.markdown("""
        <div style="background:#0d2a1a;border:1px solid rgba(80,255,160,.2);
                    border-radius:10px;padding:16px;font-size:12px;color:#c0d8f0;">
          <div style="color:#22ccaa;font-weight:bold;margin-bottom:8px;">Inference pipeline:</div>
          <p>1️⃣ <b style='color:#80c0ff'>Load</b> — <code>load_model('saved_models/X.keras')</code></p>
          <p>2️⃣ <b style='color:#80c0ff'>Scaler</b> — <code>pickle.load('scaler.pkl')</code></p>
          <p>3️⃣ <b style='color:#80c0ff'>Scale</b> — <code>scaler.transform(X_new)</code></p>
          <p>4️⃣ <b style='color:#80c0ff'>Predict</b> — <code>model.predict(X_scaled)</code></p>
          <p>5️⃣ <b style='color:#80c0ff'>Threshold</b> — prob ≥ 0.5 → Churn</p>
          <p>💡 <b style='color:#f39c12'>Tip:</b> Use the 🔮 Live Predictor tab to test live!</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📁 Model File Status (17 Models)")

    file_data = []
    for name, fname in MODEL_FILES.items():
        fp = os.path.join(MODEL_DIR, fname)
        exists = os.path.exists(fp)
        size = os.path.getsize(fp)/1024 if exists else 0
        file_data.append({
            "Phase":       "A" if name.startswith("A") else "B",
            "Model Name":  name,
            "File":        fname,
            "Status":      "✅ Found" if exists else "❌ Missing",
            "Size (KB)":   f"{size:.1f}" if exists else "—",
        })
    df_files = pd.DataFrame(file_data)
    st.dataframe(df_files, use_container_width=True)

    found = sum(1 for r in file_data if "✅" in r["Status"])
    if found == len(MODEL_FILES):
        st.success(f"✅ All {len(MODEL_FILES)} models found!")
    else:
        st.warning(f"⚠️ {found}/{len(MODEL_FILES)} models found. Run `ann_churn_modelling.py` first.")

    st.subheader("🔄 Live Re-Load Verification")
    if st.button("🔄 Load All 17 Models & Verify"):
        if df_raw is None:
            st.error("Dataset not found.")
        else:
            data = preprocess_data(df_raw)
            X, y, X_tr_s, X_te_s, y_tr, y_te, sc, _ = data
            avail = get_available_models()
            if not avail:
                st.error("No saved models found.")
            else:
                vres = []
                prog = st.progress(0)
                for i,(name,path) in enumerate(avail.items()):
                    m = load_keras_model(path)
                    if m:
                        met, _, _ = evaluate_model(m, X_te_s, y_te)
                        vres.append({"Model":name,**met,"Status":"✅ OK"})
                    else:
                        vres.append({"Model":name,"Status":"❌ Failed"})
                    prog.progress((i+1)/len(avail))
                st.success(f"✅ Verified {len(vres)} models!")
                st.dataframe(pd.DataFrame(vres), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 7 — GRAND COMPARISON
# ══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("🏆 Grand Final Comparison — All 17 Models")

    if df_raw is None:
        st.error("Dataset not found.")
    else:
        data = preprocess_data(df_raw)
        X, y, X_tr_s, X_te_s, y_tr, y_te, sc, _ = data
        avail = get_available_models()

        if not avail:
            st.warning("No saved models found. Run `ann_churn_modelling.py` first.")
        else:
            all_res = []
            for name, path in avail.items():
                m = load_keras_model(path)
                if m:
                    met, _, _ = evaluate_model(m, X_te_s, y_te, threshold)
                    phase = "A — No Imbalance" if name.startswith("A") else "B — Imbalance Handled"
                    all_res.append({"Model":name,"Phase":phase,**met})

            if all_res:
                df_all = pd.DataFrame(all_res)
                df_sorted = df_all.sort_values("F1(Churn)", ascending=False).reset_index(drop=True)

                # KPIs
                st.subheader("🎯 Best Model per Metric")
                metrics_kpi = ["Accuracy","Precision(Churn)","Recall(Churn)","F1(Churn)","ROC-AUC"]
                kpi_cols = st.columns(5)
                for col, metric in zip(kpi_cols, metrics_kpi):
                    best = df_all.loc[df_all[metric].idxmax()]
                    col.metric(f"Best {metric}", f"{best[metric]:.4f}", best["Model"].split(" — ")[0])

                st.markdown("---")

                # Full table
                st.subheader("📋 Complete Results — Ranked by F1(Churn)")
                df_disp = df_sorted.set_index("Model")
                st.dataframe(
                    df_disp[["Phase","Accuracy","Precision(Churn)","Recall(Churn)","F1(Churn)","ROC-AUC"]]
                    .style.highlight_max(subset=metrics_kpi, color="#1a5a1a")
                    .highlight_min(subset=metrics_kpi, color="#5a1a1a"),
                    use_container_width=True
                )

                # Grouped bar chart
                st.subheader("📊 All 17 Models — Side-by-Side")
                fig, axes = _dark_fig(1, 5, figsize=(28, 9))
                bar_colors = ["#3498db" if r.startswith("A") else "#e74c3c" for r in df_sorted["Model"]]
                for ax, metric in zip(axes, metrics_kpi):
                    bars = ax.barh(df_sorted["Model"], df_sorted[metric], color=bar_colors)
                    ax.set_title(metric, color="#80c0ff", fontweight="bold", fontsize=9)
                    ax.set_xlim(0, 1.14)
                    ax.tick_params(labelsize=7)
                    for bar in bars:
                        w = bar.get_width()
                        ax.text(w+.003, bar.get_y()+bar.get_height()/2, f"{w:.3f}",
                                va="center", fontsize=6.5, color="#c0d0e0")
                from matplotlib.patches import Patch
                legend_els = [Patch(facecolor="#3498db", label="Phase A — No Imbalance"),
                              Patch(facecolor="#e74c3c", label="Phase B — Imbalance Handled")]
                fig.legend(handles=legend_els, loc="lower center", ncol=2, fontsize=10,
                           facecolor="#0d1530", labelcolor="#c0d0e0", bbox_to_anchor=(.5,-0.04))
                plt.suptitle("🏆 Grand Final — All 17 ANN Models",
                             color="#80c0ff", fontsize=14, fontweight="bold")
                plt.tight_layout(); st.pyplot(fig)

                # Heatmap
                st.subheader("🌡️ Performance Heatmap")
                fig, ax = _dark_fig(figsize=(14, 8))
                heat = df_sorted.set_index("Model")[metrics_kpi]
                sns.heatmap(heat, annot=True, fmt=".3f", cmap="YlOrRd",
                            vmin=.3, vmax=1.0, linewidths=.5, ax=ax, annot_kws={"size":9})
                ax.set_title("Model Performance Heatmap (all 17)", color="#80c0ff", fontsize=13, fontweight="bold")
                ax.tick_params(colors="#80a0c0", labelsize=8)
                st.pyplot(fig)

                # Scatter plots
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.subheader("🔵 F1 vs ROC-AUC")
                    fig, ax = _dark_fig(figsize=(7, 6))
                    for _, row in df_all.iterrows():
                        c = "#3498db" if row["Model"].startswith("A") else "#e74c3c"
                        ax.scatter(row["ROC-AUC"], row["F1(Churn)"], s=100, color=c, zorder=5)
                        ax.annotate(row["Model"].split(" — ")[0],
                                    (row["ROC-AUC"], row["F1(Churn)"]),
                                    textcoords="offset points", xytext=(5,3), fontsize=7, color="#c0d0e0")
                    ax.set_xlabel("ROC-AUC", color="#80a0c0")
                    ax.set_ylabel("F1(Churn)", color="#80a0c0")
                    ax.set_title("F1 vs ROC-AUC", color="#80c0ff", fontweight="bold")
                    ax.grid(alpha=.15, color="#3a6aaa")
                    ax.legend(handles=[Patch(facecolor="#3498db", label="Phase A"),
                                       Patch(facecolor="#e74c3c", label="Phase B")],
                              facecolor="#0a0f1e", labelcolor="#c0d0e0")
                    st.pyplot(fig)
                with col_s2:
                    st.subheader("📌 Precision vs Recall")
                    fig, ax = _dark_fig(figsize=(7, 6))
                    for _, row in df_all.iterrows():
                        c = "#3498db" if row["Model"].startswith("A") else "#e74c3c"
                        ax.scatter(row["Precision(Churn)"], row["Recall(Churn)"], s=100, color=c, zorder=5)
                        ax.annotate(row["Model"].split(" — ")[0],
                                    (row["Precision(Churn)"], row["Recall(Churn)"]),
                                    textcoords="offset points", xytext=(5,3), fontsize=7, color="#c0d0e0")
                    ax.set_xlabel("Precision(Churn)", color="#80a0c0")
                    ax.set_ylabel("Recall(Churn)", color="#80a0c0")
                    ax.set_title("Precision vs Recall", color="#80c0ff", fontweight="bold")
                    ax.grid(alpha=.15, color="#3a6aaa")
                    ax.legend(handles=[Patch(facecolor="#3498db", label="Phase A"),
                                       Patch(facecolor="#e74c3c", label="Phase B")],
                              facecolor="#0a0f1e", labelcolor="#c0d0e0")
                    st.pyplot(fig)

                # Radar chart
                st.subheader("🕸️ Radar — Best A vs Best B")
                best_a = df_all[df_all["Model"].str.startswith("A")].sort_values("ROC-AUC", ascending=False).iloc[0]
                best_b = df_all[df_all["Model"].str.startswith("B")].sort_values("ROC-AUC", ascending=False).iloc[0]
                N = len(metrics_kpi)
                angles = [n/float(N)*2*math.pi for n in range(N)] + [0]
                fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
                fig.patch.set_facecolor("#0d1530"); ax.set_facecolor("#050a1a")
                for lbl, row, col in [
                    (f"Best A: {best_a['Model']}", best_a, "#3498db"),
                    (f"Best B: {best_b['Model']}", best_b, "#e74c3c"),
                ]:
                    vals = [row[m] for m in metrics_kpi] + [row[metrics_kpi[0]]]
                    ax.plot(angles, vals, lw=2.5, color=col, label=lbl)
                    ax.fill(angles, vals, alpha=.15, color=col)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(metrics_kpi, color="#80c0ff", fontsize=10)
                ax.set_ylim(0,1)
                ax.set_title("Best Phase A vs Best Phase B", color="#80c0ff", fontsize=11, fontweight="bold", pad=20)
                ax.legend(loc="upper right", bbox_to_anchor=(1.45,1.1), fontsize=9,
                          facecolor="#0d1530", labelcolor="#c0d0e0")
                ax.grid(color="#1e3a6a", alpha=.5); ax.spines["polar"].set_color("#1e3a6a")
                ax.tick_params(colors="#4a6a8a")
                st.pyplot(fig)

                # Winner
                winner = df_all.sort_values("ROC-AUC", ascending=False).iloc[0]
                st.markdown("---")
                st.markdown(f"""
                <div class='winner-badge'>
                  <h2>🏆 Overall Best Model</h2>
                  <h3 style='color:#44ff44;'>{winner['Model']}</h3>
                  <p style='color:#80ff80;'>Phase: {winner['Phase']}</p>
                  <table style='margin:auto;color:#c0e0c0;font-size:15px;'>
                    <tr><td>Accuracy</td><td>&nbsp;&nbsp;{winner['Accuracy']:.4f}</td></tr>
                    <tr><td>Recall(Churn)</td><td>&nbsp;&nbsp;{winner['Recall(Churn)']:.4f}</td></tr>
                    <tr><td>F1(Churn)</td><td>&nbsp;&nbsp;{winner['F1(Churn)']:.4f}</td></tr>
                    <tr><td><b>ROC-AUC</b></td><td>&nbsp;&nbsp;<b>{winner['ROC-AUC']:.4f}</b></td></tr>
                  </table>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 8 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("🔮 Live Customer Churn Predictor")
    st.markdown("Enter a customer's details and get an instant churn probability from any of the 17 saved models.")
    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(32,82,164,.18);border:1px solid rgba(80,160,255,.3);border-radius:14px;padding:16px;text-align:center;">
      <strong style="color:#c8e4ff;font-size:15px;">⬇️ Live Predictor controls are below — fill the form and click the big button.</strong>
      <div style="color:#d0e8ff;margin-top:6px;font-size:13px;">There is also a full-width predict button above the form for easier discovery.</div>
    </div>
    """, unsafe_allow_html=True)

    avail = get_available_models()
    if not avail:
        st.warning("No saved models found. Run `ann_churn_modelling.py` first.")
    else:
        if "live_predict_click" not in st.session_state:
            st.session_state.live_predict_click = False

        def _set_live_predict():
            st.session_state.live_predict_click = True

        st.button("🔮 Predict Churn Probability", key="live_predict_top",
                  on_click=_set_live_predict, use_container_width=True)
        col_form, col_result = st.columns([1, 1])

        with col_form:
            st.subheader("👤 Customer Details")
            credit_score = st.slider("Credit Score",       300, 900, 650)
            age          = st.slider("Age",                 18,  90,  38)
            tenure       = st.slider("Tenure (years)",       0,  10,   5)
            balance      = st.number_input("Balance ($)",   0.0, 300000.0, 75000.0, 1000.0)
            n_products   = st.selectbox("Num Products",    [1, 2, 3, 4])
            has_cr_card  = st.selectbox("Has Credit Card?",["Yes","No"])
            is_active    = st.selectbox("Is Active Member?",["Yes","No"])
            est_salary   = st.number_input("Estimated Salary ($)", 0.0, 250000.0, 80000.0, 1000.0)
            geography    = st.selectbox("Geography",       ["France","Germany","Spain"])
            gender       = st.selectbox("Gender",          ["Male","Female"])
            st.markdown("---")
            sel_pred = st.selectbox("🤖 Model to use:", list(avail.keys()))
            st.button("🔮 Predict Churn Probability", key="live_predict_bottom",
                      on_click=_set_live_predict, use_container_width=True)
            predict_btn = st.session_state.live_predict_click
            if predict_btn:
                st.session_state.live_predict_click = False

        with col_result:
            st.subheader("📊 Prediction Result")
            if not predict_btn:
                st.info("Enter customer details on the left and click 'Predict Churn Probability' to get live churn output.")
            if predict_btn:
                raw_input = {
                    "CreditScore":       credit_score,
                    "Gender":            1 if gender == "Female" else 0,
                    "Age":               age,
                    "Tenure":            tenure,
                    "Balance":           balance,
                    "NumOfProducts":     n_products,
                    "HasCrCard":         1 if has_cr_card == "Yes" else 0,
                    "IsActiveMember":    1 if is_active == "Yes" else 0,
                    "EstimatedSalary":   est_salary,
                    "Geography_Germany": 1 if geography == "Germany" else 0,
                    "Geography_Spain":   1 if geography == "Spain" else 0,
                }
                inp_df = pd.DataFrame([raw_input])
                # Apply same log1p as preprocessing
                for col_name in ["Balance","EstimatedSalary"]:
                    inp_df[col_name] = np.log1p(inp_df[col_name])

                sc_loaded = load_scaler()
                if sc_loaded is None and df_raw is not None:
                    sc_loaded = preprocess_data(df_raw)[6]

                if sc_loaded is None:
                    st.error("Scaler not found. Run the script first.")
                else:
                    try:
                        X_inp = sc_loaded.transform(inp_df)
                    except Exception as e:
                        st.error(f"Scaler error: {e}"); X_inp = None

                    if X_inp is not None:
                        m = load_keras_model(avail[sel_pred])
                        if m:
                            prob = float(m.predict(X_inp, verbose=0).flatten()[0])
                            pred = int(prob >= threshold)
                            gc   = "#e74c3c" if prob > .5 else "#27ae60"
                            rl   = "HIGH RISK" if prob>.7 else "MEDIUM RISK" if prob>.4 else "LOW RISK"
                            rc   = "#e74c3c" if prob>.7 else "#f39c12" if prob>.4 else "#27ae60"

                            st.markdown(f"""
                            <div style="text-align:center;padding:20px;
                                        background:rgba(20,40,80,.3);border-radius:12px;
                                        border:1px solid {gc}40;">
                              <h2 style="color:{gc};">{"🚨 WILL CHURN" if pred else "✅ WILL STAY"}</h2>
                              <h1 style="font-size:54px;color:{gc};margin:10px 0;">{prob*100:.1f}%</h1>
                              <p style="color:#80a0c0;">Churn Probability</p>
                              <span style="background:{rc}33;border:1px solid {rc};color:{rc};
                                           border-radius:20px;padding:4px 16px;font-weight:bold;">
                                {rl}
                              </span>
                            </div>""", unsafe_allow_html=True)

                            # Probability bar
                            fig, ax = _dark_fig(figsize=(8, 1.6))
                            ax.barh(["Churn Prob"], [prob],       color=gc,       height=.5)
                            ax.barh(["Churn Prob"], [1 - prob], left=[prob], color="#1e3a6a", height=.5)
                            ax.axvline(x=threshold, color="yellow", ls="--", lw=1.5,
                                       label=f"Threshold {threshold}")
                            ax.set_xlim(0, 1)
                            ax.set_title(f"P(Churn)={prob:.4f}  |  Threshold={threshold}  |  Model: {sel_pred}",
                                         color="#80c0ff", fontsize=9)
                            ax.legend(facecolor="#0d1530", labelcolor="#c0d0e0", fontsize=8)
                            plt.tight_layout(); st.pyplot(fig)

                            # All models comparison for this customer
                            st.markdown("---")
                            st.subheader("🔄 Compare All Models for This Customer")
                            if st.button("Run All 17 Models on This Customer"):
                                all_preds = []
                                for mname, mpath in avail.items():
                                    mm = load_keras_model(mpath)
                                    if mm:
                                        p = float(mm.predict(X_inp, verbose=0).flatten()[0])
                                        all_preds.append({"Model": mname,
                                                          "P(Churn)": round(p,4),
                                                          "Prediction": "🚨 Churn" if p>=threshold else "✅ Stay"})
                                df_preds = pd.DataFrame(all_preds).sort_values("P(Churn)", ascending=False)
                                st.dataframe(df_preds, use_container_width=True)

                                fig, ax = _dark_fig(figsize=(10, 6))
                                colors_pred = ["#e74c3c" if p>=threshold else "#27ae60"
                                               for p in df_preds["P(Churn)"]]
                                ax.barh(df_preds["Model"], df_preds["P(Churn)"], color=colors_pred)
                                ax.axvline(x=threshold, color="yellow", ls="--", lw=1.5,
                                           label=f"Threshold {threshold}")
                                ax.set_xlim(0, 1)
                                ax.set_title("All Models — Churn Probability for This Customer",
                                             color="#80c0ff", fontweight="bold")
                                ax.legend(facecolor="#0d1530", labelcolor="#c0d0e0")
                                plt.tight_layout(); st.pyplot(fig)

                            st.subheader("🔎 Input Summary")
                            summary_df = pd.DataFrame([{
                                "Credit Score": credit_score,
                                "Gender": gender,
                                "Age": age,
                                "Tenure": tenure,
                                "Balance": balance,
                                "Num Products": n_products,
                                "Has Credit Card": has_cr_card,
                                "Is Active Member": is_active,
                                "Estimated Salary": est_salary,
                                "Geography": geography,
                            }])
                            st.table(summary_df)
                        else:
                            st.error("Failed to load model.")

        # Batch prediction
        st.markdown("---")
        st.subheader("📂 Batch Prediction — Multi-row Customer Scoring")
        st.markdown("Upload a CSV containing customer feature rows and score them all at once with the selected model.")
        st.markdown("The CSV should include the original input columns from `Churn_Modeling.csv` except `Exited`, for example `CreditScore`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, and `Geography`. ")
        uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
        if uploaded:
            batch_df = pd.read_csv(uploaded)
            st.subheader("Uploaded Data Preview")
            st.dataframe(batch_df.head(), use_container_width=True)
            required_cols = ["CreditScore","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Geography"]
            if not set(required_cols).issubset(batch_df.columns):
                missing = [c for c in required_cols if c not in batch_df.columns]
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                st.info(f"Uploaded {len(batch_df)} rows. The selected model will score these rows after preprocessing.")
                if st.button("Run Batch Prediction", key="batch_predict"):
                    batch_processed = batch_df.copy()
                    if "Exited" in batch_processed.columns:
                        batch_processed = batch_processed.drop(columns=["Exited"])
                    batch_processed["Gender"] = batch_processed["Gender"].map({"Male":0,"Female":1})
                    batch_processed["HasCrCard"] = batch_processed["HasCrCard"].map({"Yes":1,"No":0}).fillna(batch_processed["HasCrCard"])
                    batch_processed["IsActiveMember"] = batch_processed["IsActiveMember"].map({"Yes":1,"No":0}).fillna(batch_processed["IsActiveMember"])
                    batch_processed["Geography_Germany"] = (batch_processed["Geography"] == "Germany").astype(int)
                    batch_processed["Geography_Spain"] = (batch_processed["Geography"] == "Spain").astype(int)
                    for col_name in ["Balance","EstimatedSalary"]:
                        batch_processed[col_name] = np.log1p(batch_processed[col_name])

                    feature_cols = [c for c in feat_names if c in batch_processed.columns]
                    missing_features = [c for c in feat_names if c not in batch_processed.columns]
                    if missing_features:
                        st.error(f"Missing processed feature columns: {', '.join(missing_features)}")
                    else:
                        X_batch = batch_processed[feature_cols]
                        sc_loaded = load_scaler()
                        if sc_loaded is None and df_raw is not None:
                            sc_loaded = preprocess_data(df_raw)[6]
                        if sc_loaded is None:
                            st.error("Scaler not found. Run the script first.")
                        else:
                            X_batch_scaled = sc_loaded.transform(X_batch)
                            mm = load_keras_model(avail[sel_pred])
                            if mm is None:
                                st.error("Failed to load the selected model for batch scoring.")
                            else:
                                probs = mm.predict(X_batch_scaled, verbose=0).flatten()
                                preds = (probs >= threshold).astype(int)
                                result_df = batch_df.copy()
                                result_df["P(Churn)"] = np.round(probs, 4)
                                result_df["Prediction"] = np.where(preds == 1, "🚨 Churn", "✅ Stay")
                                st.subheader("Batch Prediction Results")
                                st.dataframe(result_df, use_container_width=True)
                                csv = result_df.to_csv(index=False).encode("utf-8")
                                st.download_button("Download scored CSV", data=csv, file_name="batch_predictions.csv", mime="text/csv")
