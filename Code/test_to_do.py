#!/usr/bin/env python
# coding: utf-8

# # **BirdCLEF 2025 Inference Notebook**
# This notebook runs inference on BirdCLEF 2025 test soundscapes and generates a submission file. It supports both single model inference and ensemble inference with multiple models. You can find the pre-processing and training processes in the following notebooks:
#
# - [Transforming Audio-to-Mel Spec. | BirdCLEF'25](https://www.kaggle.com/code/kadircandrisolu/transforming-audio-to-mel-spec-birdclef-25)
# - [EfficientNet B0 Pytorch [Train] | BirdCLEF'25](https://www.kaggle.com/code/kadircandrisolu/efficientnet-b0-pytorch-train-birdclef-25)
#
# **Features**
# - Audio Preprocessing
# - Test-Time Augmentation (TTA)

# In[1]:


import os
import gc
import warnings
import logging
import time
import math
import cv2
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score #added

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)


# In[2]:


class CFG:
    # test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    # submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    # taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    # model_path = '/kaggle/input/birdclef-2025-efficientnet-b0'

    OR_PATH = os.getcwd()
    os.chdir("..")
    ROOT_PATH = os.getcwd()
    sep = os.path.sep

    # Base Data directory
    DATA_DIR = os.path.join(ROOT_PATH, 'Data') + sep

    test_soundscapes = os.path.join(DATA_DIR, 'test_soundscapes')
    submission_csv = os.path.join(DATA_DIR, 'sample_submission.csv')
    taxonomy_csv = os.path.join(DATA_DIR, 'taxonomy.csv')

    CODE_DIR = os.path.join(ROOT_PATH, 'Code') + sep
    model_path = CODE_DIR

    # Audio parameters
    FS = 32000
    WINDOW_SIZE = 5

    # Mel spectrogram parameters
    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000
    TARGET_SHAPE = (256, 256)

    model_name = 'efficientnet_b0'
    in_channels = 1
    device = 'cpu'

    # Inference parameters
    batch_size = 16
    use_tta = False
    tta_count = 3
    threshold = 0.5

    use_specific_folds = False  # If False, use all found models
    folds = [0, 4]  # Used only if use_specific_folds is True

    debug = False
    debug_count = 3


cfg = CFG()

# In[3]:


print(f"Using device: {cfg.device}")
print(f"Loading taxonomy data...")
taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
species_ids = taxonomy_df['primary_label'].tolist()
num_classes = len(species_ids)
print(f"Number of classes: {num_classes}")


# In[4]:


class BirdCLEFModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=False,
            in_chans=cfg.in_channels,
            drop_rate=0.0,
            drop_path_rate=0.0
        )

        if 'efficientnet' in cfg.model_name:
            backbone_out = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif 'resnet' in cfg.model_name:
            backbone_out = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            backbone_out = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0, '')

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, num_classes)

    def forward(self, x):
        features = self.backbone(x)

        if isinstance(features, dict):
            features = features['features']

        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)

        logits = self.classifier(features)
        return logits


# In[5]:


def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    return mel_spec_norm


def process_audio_segment(audio_data, cfg):
    """Process audio segment to get mel spectrogram"""
    if len(audio_data) < cfg.FS * cfg.WINDOW_SIZE:
        audio_data = np.pad(audio_data,
                            (0, cfg.FS * cfg.WINDOW_SIZE - len(audio_data)),
                            mode='constant')

    mel_spec = audio2melspec(audio_data, cfg)

    # Resize if needed
    if mel_spec.shape != cfg.TARGET_SHAPE:
        mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

    return mel_spec.astype(np.float32)


# In[6]:


def find_model_files(cfg):
    """
    Find only 'model_best.pth' in the model directory
    """
    model_dir = Path(cfg.model_path)

    model_best_path = model_dir / 'model_best.pth'

    if model_best_path.exists():
        return [str(model_best_path)]
    else:
        print(f"Warning: model_best.pth not found under {cfg.model_path}!")
        return []



def load_models(cfg, num_classes):
    """
    Load all found model files and prepare them for ensemble
    """
    models = []

    model_files = find_model_files(cfg)

    if not model_files:
        print(f"Warning: No model files found under {cfg.model_path}!")
        return models

    print(f"Found a total of {len(model_files)} model files.")

    if cfg.use_specific_folds:
        filtered_files = []
        for fold in cfg.folds:
            fold_files = [f for f in model_files if f"fold{fold}" in f]
            filtered_files.extend(fold_files)
        model_files = filtered_files
        print(f"Using {len(model_files)} model files for the specified folds ({cfg.folds}).")

    for model_path in model_files:
        try:
            print(f"Loading model: {model_path}")
            # checkpoint = torch.load(model_path, map_location=torch.device(cfg.device))
            checkpoint = torch.load(model_path, map_location=torch.device(cfg.device), weights_only=False)

            model = BirdCLEFModel(cfg, num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(cfg.device)
            model.eval()

            models.append(model)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")

    return models


def predict_on_spectrogram(audio_path, models, cfg, species_ids):
    """Process a single audio file and predict species presence for each 5-second segment"""
    predictions = []
    row_ids = []
    soundscape_id = Path(audio_path).stem

    try:
        # print(f"Processing {soundscape_id}") #commented
        pass #added
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)

        total_segments = int(len(audio_data) / (cfg.FS * cfg.WINDOW_SIZE))

        for segment_idx in range(total_segments):
            start_sample = segment_idx * cfg.FS * cfg.WINDOW_SIZE
            end_sample = start_sample + cfg.FS * cfg.WINDOW_SIZE
            segment_audio = audio_data[start_sample:end_sample]

            end_time_sec = (segment_idx + 1) * cfg.WINDOW_SIZE
            row_id = f"{soundscape_id}_{end_time_sec}"
            row_ids.append(row_id)

            if cfg.use_tta:
                all_preds = []

                for tta_idx in range(cfg.tta_count):
                    mel_spec = process_audio_segment(segment_audio, cfg)
                    mel_spec = apply_tta(mel_spec, tta_idx)

                    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    mel_spec = mel_spec.to(cfg.device)

                    if len(models) == 1:
                        with torch.no_grad():
                            outputs = models[0](mel_spec)
                            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            all_preds.append(probs)
                    else:
                        segment_preds = []
                        for model in models:
                            with torch.no_grad():
                                outputs = model(mel_spec)
                                probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                                segment_preds.append(probs)

                        avg_preds = np.mean(segment_preds, axis=0)
                        all_preds.append(avg_preds)

                final_preds = np.mean(all_preds, axis=0)
            else:
                mel_spec = process_audio_segment(segment_audio, cfg)

                mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                mel_spec = mel_spec.to(cfg.device)

                if len(models) == 1:
                    with torch.no_grad():
                        outputs = models[0](mel_spec)
                        final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()
                else:
                    segment_preds = []
                    for model in models:
                        with torch.no_grad():
                            outputs = model(mel_spec)
                            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                            segment_preds.append(probs)

                    final_preds = np.mean(segment_preds, axis=0)

            predictions.append(final_preds)

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

    return row_ids, predictions


# In[7]:


def apply_tta(spec, tta_idx):
    """Apply test-time augmentation"""
    if tta_idx == 0:
        # Original spectrogram
        return spec
    elif tta_idx == 1:
        # Time shift (horizontal flip)
        return np.flip(spec, axis=1)
    elif tta_idx == 2:
        # Frequency shift (vertical flip)
        return np.flip(spec, axis=0)
    else:
        return spec


def run_inference(cfg, models, species_ids):
    """Run inference on all test soundscapes"""
    test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))

    if cfg.debug:
        print(f"Debug mode enabled, using only {cfg.debug_count} files")
        test_files = test_files[:cfg.debug_count]

    print(f"Found {len(test_files)} test soundscapes")

    all_row_ids = []
    all_predictions = []

    for audio_path in tqdm(test_files):
        row_ids, predictions = predict_on_spectrogram(str(audio_path), models, cfg, species_ids)
        all_row_ids.extend(row_ids)
        all_predictions.extend(predictions)

    return all_row_ids, all_predictions


def create_submission(row_ids, predictions, species_ids, cfg):
    """Create submission dataframe"""
    print("Creating submission dataframe...")

    submission_dict = {'row_id': row_ids}

    for i, species in enumerate(species_ids):
        submission_dict[species] = [pred[i] for pred in predictions]

    submission_df = pd.DataFrame(submission_dict)

    submission_df.set_index('row_id', inplace=True)

    sample_sub = pd.read_csv(cfg.submission_csv, index_col='row_id')

    missing_cols = set(sample_sub.columns) - set(submission_df.columns)
    if missing_cols:
        print(f"Warning: Missing {len(missing_cols)} species columns in submission")
        for col in missing_cols:
            submission_df[col] = 0.0

    submission_df = submission_df[sample_sub.columns]

    submission_df = submission_df.reset_index()

    return submission_df

def compute_macro_auc(y_true, y_pred): #added
    """
    Compute macro-averaged ROC-AUC score, skipping classes with no positive labels.
    """
    aucs = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) == 0:
            continue
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        aucs.append(auc)
    return np.mean(aucs) if aucs else 0.0



# In[8]:


def main():
    start_time = time.time()
    print("Starting BirdCLEF-2025 inference...")
    print(f"TTA enabled: {cfg.use_tta} (variations: {cfg.tta_count if cfg.use_tta else 0})")

    meta_df = pd.read_csv(os.path.join(cfg.DATA_DIR, 'test_metadata.csv'))
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()

    models = load_models(cfg, num_classes)

    if not models:
        print("No models found! Please check model paths.")
        return

    print(f"Model usage: {'Single model' if len(models) == 1 else f'Ensemble of {len(models)} models'}")

    row_ids, predictions = run_inference(cfg, models, species_ids)

    submission_df = create_submission(row_ids, predictions, species_ids, cfg)

    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    # ================== Compute ROC-AUC =================== -added
    print("Computing validation ROC-AUC...")

    # Strip .ogg from filename → to match row_id prefix
    meta_df['prefix'] = meta_df['filename'].str.replace('.ogg', '', regex=False)
    prefix_to_label = dict(zip(meta_df['prefix'], meta_df['primary_label']))

    # Create y_true: one-hot encoded ground truth
    y_true = []
    for row_id in submission_df['row_id']:
        prefix = row_id.rsplit('_', 1)[0]  # remove trailing time (_5, _10, etc.)
        label = prefix_to_label.get(prefix, None)
        one_hot = [1 if label == sp else 0 for sp in species_ids]
        y_true.append(one_hot)

    y_true = np.array(y_true)
    y_pred = submission_df[species_ids].values

    val_auc = compute_macro_auc(y_true, y_pred)
    print(f"Validation ROC-AUC (macro, skipped empty classes): {val_auc:.4f}")

    end_time = time.time()
    print(f"Inference completed in {(end_time - start_time) / 60:.2f} minutes")


# In[9]:


if __name__ == "__main__":
    main()

# In[ ]:




