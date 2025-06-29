import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import torch
import torch.nn as nn
import timm
import tempfile
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── CONFIGURE PATHS ─────────────────────────────────────────────────────────────
# ── POINT AT YOUR STREAMLIT APP LOCATION ───────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
TAXONOMY_CSV = PROJECT_ROOT / "Data" / "taxonomy.csv"
MODEL_PATH   = PROJECT_ROOT / "Code" / "model_best.pth"

SAMPLE_RATE  = 32000
MAX_DURATION = 60       # in seconds

# ── STREAMLIT PAGE SETUP ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bird Sound Classifier",
    page_icon="🐦",
    layout="wide",
)

st.title("🐦 Bird Sound Classifier")
st.markdown(f"""
Upload up to {MAX_DURATION}s of audio (MP3, WAV, OGG, FLAC)
and this app will tell you which bird it most likely is, using an EfficientNet-based model.
""")

# ── LOAD TAXONOMY ───────────────────────────────────────────────────────────────
@st.cache_data
def load_taxonomy(path):
    df = pd.read_csv(path)
    return df

if not TAXONOMY_CSV.exists():
    st.error(f"Cannot find taxonomy file at\n→ {TAXONOMY_CSV}")
    st.stop()

taxonomy_df = load_taxonomy(TAXONOMY_CSV)
species_ids   = taxonomy_df['primary_label'].tolist()
species_names = taxonomy_df['common_name'].tolist()
class_names   = taxonomy_df['class_name'].tolist()

# ── MODEL DEFINITION ────────────────────────────────────────────────────────────
class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.model_name, pretrained=False,
            in_chans=cfg.in_channels
        )
        # strip out default head
        if hasattr(self.backbone, 'classifier'):
            feat_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            feat_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            feat_dim = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        # some timm models return dicts
        if isinstance(feats, dict):
            feats = feats['features']
        if feats.dim() == 4:
            feats = self.pool(feats).flatten(1)
        return self.classifier(feats)

class CFG:
    model_name   = 'efficientnet_b0'
    in_channels  = 1
    device       = 'cuda' if torch.cuda.is_available() else 'cpu'
    FS           = SAMPLE_RATE
    N_FFT        = 1024
    HOP_LENGTH   = 512
    N_MELS       = 128
    FMIN, FMAX   = 50, 14_000
    TARGET_SHAPE = (256, 256)
    WINDOW_SIZE  = 5   # seconds

cfg = CFG()

import pickle

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Cannot find model file at {MODEL_PATH}")
        return None

    try:
        # turn off the restricted weights-only unpickler
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=cfg.device,
            weights_only=False
        )
    except TypeError:
        # older PyTorch that doesn't accept weights_only arg
        with open(MODEL_PATH, 'rb') as f:
            checkpoint = pickle.load(f)

    # extract whatever your checkpoint key is
    if 'model_state_dict' in checkpoint:
        sd = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        sd = checkpoint['state_dict']
    else:
        sd = checkpoint

    model = BirdCLEFModel(cfg, len(species_ids))
    model.load_state_dict(sd)
    model.to(cfg.device).eval()
    return model

model = load_model()
if model is None:
    st.stop()

# ── AUDIO PROCESSING UTILITIES ─────────────────────────────────────────────────
def load_audio(audio_bytes, fmt):
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        wav, _ = librosa.load(tmp.name, sr=cfg.FS, mono=True)
    os.unlink(tmp.name)
    return wav

def to_melspec(y):
    m = librosa.feature.melspectrogram(
        y=y, sr=cfg.FS,
        n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN, fmax=cfg.FMAX,
        power=2.0
    )
    db = librosa.power_to_db(m, ref=np.max)
    norm = (db - db.min()) / (db.max() - db.min() + 1e-8)
    return norm

def preprocess(y):
    # trim or pad
    max_len = cfg.FS * MAX_DURATION
    if len(y) > max_len:
        y = y[:max_len]
    elif len(y) < cfg.FS * cfg.WINDOW_SIZE:
        pad = cfg.FS*cfg.WINDOW_SIZE - len(y)
        y = np.pad(y, (0,pad), mode='constant')
    spec = to_melspec(y)
    spec = cv2.resize(spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
    return spec.astype(np.float32)

def predict(y):
    spec = preprocess(y)
    t = torch.tensor(spec).unsqueeze(0).unsqueeze(0).to(cfg.device)
    with torch.no_grad():
        out = model(t)
        prob = torch.sigmoid(out).cpu().numpy().squeeze()
    df = pd.DataFrame({
        'species_id': species_ids,
        'common_name': species_names,
        'class_name': class_names,
        'probability': prob
    }).sort_values('probability', ascending=False)
    return df

# ── PLOTTING HELPERS ───────────────────────────────────────────────────────────
def show_waveform(y):
    fig, ax = plt.subplots(figsize=(10,2))
    librosa.display.waveshow(y, sr=cfg.FS, ax=ax)
    ax.set(title="Waveform", xlabel="Time (s)")
    st.pyplot(fig)

def show_spec(y):
    spec = to_melspec(y)
    fig, ax = plt.subplots(figsize=(10,3))
    img = librosa.display.specshow(spec, y_axis='mel', x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title="Mel-Spectrogram")
    st.pyplot(fig)

def show_top(df, n=10):
    top = df.head(n)
    fig, ax = plt.subplots(figsize=(8,4))
    sns.barplot(x='probability', y='common_name', data=top, ax=ax)
    ax.set(xlim=(0,1), title=f"Top {n} Predictions")
    st.pyplot(fig)

# ── USER INTERFACE ─────────────────────────────────────────────────────────────
st.header("Upload an audio file")
f = st.file_uploader("Choose MP3, WAV, OGG or FLAC", type=['mp3','wav','ogg','flac'])

if f:
    st.audio(f)
    data = f.read()
    ext  = f.name.split('.')[-1].lower()
    wav  = load_audio(data, ext)
    dur  = len(wav)/cfg.FS

    if dur > MAX_DURATION:
        st.warning(f"Trimming to first {MAX_DURATION}s (you uploaded {dur:.1f}s)")
        wav = wav[:cfg.FS*MAX_DURATION]

    c1, c2 = st.columns(2)
    c1.write(f"Duration: {dur:.2f}s")
    c2.write(f"Sample rate: {cfg.FS} Hz")

    st.subheader("Audio Analysis")
    show_waveform(wav)
    show_spec(wav)

    st.subheader("Predictions")
    res = predict(wav)
    best = res.iloc[0]
    st.success(f"**{best.common_name}**  —  {best.probability:.1%}")
    show_top(res)

    with st.expander("View all results"):
        st.dataframe(res, use_container_width=True)

st.markdown("---")
st.markdown("**About:** Trained on BirdCLEF 2025 with EfficientNet-B0.")
