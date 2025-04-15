import os

class CFG:
    """
        Configuration settings for BirdCLEF audio classification training.

        Attributes:
        -----------
        seed : Random seed for reproducibility.
        debug : If True, enables quick testing mode.
        apex : Use NVIDIA Apex for mixed precision.
        print_freq : Print log every N steps.
        num_workers : DataLoader worker count.

        OR_PATH, ROOT_PATH : Paths for navigating project structure.
        DATA_DIR : Main data folder path.
        train_datadir, train_csv, test_soundscapes, submission_csv,
        taxonomy_csv, recording_location_txt, train_soundscapes : Data file paths.

        model_name : Backbone architecture (e.g., 'efficientnet_b0').
        pretrained : Load pretrained weights.
        in_channels : Input channels (1 = grayscale).

        LOAD_DATA : Whether to load data.
        FS : Sampling rate.
        TARGET_DURATION : Clip duration in seconds.
        TARGET_SHAPE : Shape of spectrogram input.
        N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX : Audio processing params.

        device : 'cuda' if available, else 'cpu'.
        epochs : Training epochs.
        batch_size : Batch size.
        criterion : Loss function.
        n_fold : Total folds.
        selected_folds : Folds used for training.

        optimizer : Optimizer type (e.g., AdamW).
        lr : Learning rate.
        weight_decay : L2 regularization.
        scheduler : LR scheduler.
        min_lr : Minimum learning rate.
        T_max : Max iterations for cosine scheduler.

        aug_prob : Data augmentation probability.
        mixup_alpha : Mixup augmentation strength.

        Methods:
        --------
        update_debug_settings(): Adjusts config for debug mode.
        """

    seed = 42
    debug = True  
    apex = False
    print_freq = 100
    num_workers = 2



    # Save current working directory and go one level up to project root
    OR_PATH = os.getcwd()
    os.chdir("..")
    ROOT_PATH = os.getcwd()
    sep = os.path.sep

    DATA_DIR = os.path.join(ROOT_PATH, 'Data') + sep

    # Define paths to files and folders inside the Data folder
    train_datadir = os.path.join(DATA_DIR, 'train_audio')
    train_csv = os.path.join(DATA_DIR, 'train.csv')
    test_soundscapes = os.path.join(DATA_DIR, 'test_soundscapes')
    submission_csv = os.path.join(DATA_DIR, 'sample_submission.csv')
    taxonomy_csv = os.path.join(DATA_DIR, 'taxonomy.csv')
    recording_location_txt = os.path.join(DATA_DIR, 'recording_location.txt')
    train_soundscapes = os.path.join(DATA_DIR, 'train_soundscapes')

    # Optional: return to original path
    os.chdir(OR_PATH)

    #====================================================================================#
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

cfg = CFG()


