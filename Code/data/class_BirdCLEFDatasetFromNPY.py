import pandas as pd
import numpy as np
import torch
import random
from utils.utils_preprocessing import process_audio_file
from torch.utils.data import Dataset
import librosa



class BirdCLEFDatasetFromNPY(Dataset):
    """
    PyTorch Dataset class for BirdCLEF 2025 audio classification using precomputed spectrograms.

    Args:
        df (pd.DataFrame): DataFrame containing metadata and file information.
        cfg (object): Configuration object with necessary parameters (e.g., paths, flags).
        spectrograms (dict, optional): Dictionary of precomputed spectrograms keyed by sample name.
        mode (str): Dataset mode, either "train" or "validation". Affects augmentations and logging.
    """

    def __init__(self, df, cfg, spectrograms=None, mode="train"):
        """
        Initialize the dataset, loading taxonomy, generating sample names, and validating spectrograms.

        Args:
            df (pd.DataFrame): Input dataframe with columns like 'filename' and 'primary_label'.
            cfg (object): Config object containing parameters like train_datadir, seed, debug, etc.
            spectrograms (dict, optional): Dictionary of spectrogram arrays keyed by samplename.
            mode (str): Indicates "train" or "validation", used to apply augmentations conditionally.
        """

        self.df = df
        self.cfg = cfg
        self.mode = mode
        self.spectrograms = spectrograms

        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        if 'filepath' not in self.df.columns:
            self.df['filepath'] = self.cfg.train_datadir + '/' + self.df.filename

        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])

        sample_names = set(self.df['samplename'])
        if self.spectrograms:
            found_samples = sum(1 for name in sample_names if name in self.spectrograms)
            print(f"Found {found_samples} matching spectrograms for {mode} dataset out of {len(self.df)} samples")

        if cfg.debug:
            self.df = self.df.sample(min(1000, len(self.df)), random_state=cfg.seed).reset_index(drop=True)

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: Dictionary with keys:
                - 'melspec' (torch.Tensor): Spectrogram tensor with shape (1, n_mels, time).
                - 'target' (torch.Tensor): One-hot encoded target vector.
                - 'filename' (str): Filename of the original audio sample.
        """
        row = self.df.iloc[idx]
        samplename = row['samplename']
        spec = None

        if self.spectrograms and samplename in self.spectrograms:
            spec = self.spectrograms[samplename]
        elif not self.cfg.LOAD_DATA:
            spec = process_audio_file(row['filepath'], self.cfg)

        if spec is None:
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
            if self.mode == "train":
                print(f"Warning: Spectrogram for {samplename} not found and could not be generated")

        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self.apply_spec_augmentations(spec)

        target = self.encode_label(row['primary_label'])

        if 'secondary_labels' in row and row['secondary_labels'] not in [[''], None, np.nan]:
            if isinstance(row['secondary_labels'], str):
                secondary_labels = eval(row['secondary_labels'])
            else:
                secondary_labels = row['secondary_labels']

            for label in secondary_labels:
                if label in self.label_to_idx:
                    target[self.label_to_idx[label]] = 1.0

        return {
            'melspec': spec,
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': row['filename']
        }

    def apply_spec_augmentations(self, spec):
        """
        Apply basic data augmentations to the spectrogram:
        - Time masking
        - Frequency masking
        - Random brightness/contrast

        Args:
            spec (torch.Tensor): Input spectrogram of shape (1, n_mels, time).

        Returns:
            torch.Tensor: Augmented spectrogram.
        """
        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                width = random.randint(5, 20)
                start = random.randint(0, spec.shape[2] - width)
                spec[0, :, start:start+width] = 0

        if random.random() < 0.5:
            num_masks = random.randint(1, 3)
            for _ in range(num_masks):
                height = random.randint(5, 20)
                start = random.randint(0, spec.shape[1] - height)
                spec[0, start:start+height, :] = 0

        if random.random() < 0.5:
            gain = random.uniform(0.8, 1.2)
            bias = random.uniform(-0.1, 0.1)
            spec = spec * gain + bias
            spec = torch.clamp(spec, 0, 1)

        return spec

    def encode_label(self, label):
        """
        Convert a primary label into a one-hot encoded target vector.

        Args:
            label (str): Bird species label.

        Returns:
            np.ndarray: One-hot encoded vector of size `num_classes`.
        """
        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target
