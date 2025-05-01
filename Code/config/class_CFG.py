import os
import torch

class CFG:
    seed = 42
    debug = False
    apex = False
    print_freq = 100
    num_workers = 2

    # OUTPUT_DIR = '/kaggle/working/'
    #
    # train_datadir = '/kaggle/input/birdclef-2025/train_audio'
    # train_csv = '/kaggle/input/birdclef-2025/train.csv'
    # test_soundscapes = '/kaggle/input/birdclef-2025/test_soundscapes'
    # submission_csv = '/kaggle/input/birdclef-2025/sample_submission.csv'
    # taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    #
    # spectrogram_npy = '/kaggle/input/birdclef25-mel-spectrograms/birdclef2025_melspec_5sec_256_256.npy'

    # Save current working directory and go one level up to project root
    OR_PATH = os.getcwd()
    os.chdir("..")
    ROOT_PATH = os.getcwd()
    sep = os.path.sep

    # Base Data directory
    DATA_DIR = os.path.join(ROOT_PATH, 'Data') + sep

    # Define paths to files and folders inside the Data folder
    train_datadir = os.path.join(DATA_DIR, 'train_audio')
    train_csv = os.path.join(DATA_DIR, 'train.csv')
    test_soundscapes = os.path.join(DATA_DIR, 'test_soundscapes')
    submission_csv = os.path.join(DATA_DIR, 'sample_submission.csv')
    taxonomy_csv = os.path.join(DATA_DIR, 'taxonomy.csv')
    recording_location_txt = os.path.join(DATA_DIR, 'recording_location.txt')
    train_soundscapes = os.path.join(DATA_DIR, 'train_soundscapes')

    # Mel-spectrogram .npy file
    spectrogram_npy = os.path.join(DATA_DIR, 'melspec', 'birdclef2025_melspec_5sec_256_256.npy')

    # Return to original directory
    os.chdir(OR_PATH)

    model_name = 'efficientnet_b0'
    pretrained = True
    in_channels = 1

    LOAD_DATA = True
    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)

    N_FFT = 1024
    HOP_LENGTH = 512
    N_MELS = 128
    FMIN = 50
    FMAX = 14000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10
    batch_size = 32
    criterion = 'BCEWithLogitsLoss'

    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]

    optimizer = 'AdamW'
    lr = 5e-4
    weight_decay = 1e-5

    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs

    aug_prob = 0.5
    mixup_alpha = 0.5

    def update_debug_settings(self):
        if self.debug:
            self.epochs = 2
            self.selected_folds = [0]

