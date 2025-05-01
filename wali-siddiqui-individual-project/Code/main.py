import pandas as pd

from config.class_CFG import CFG
from utils.util_set_seed import set_seed
from train.util_run_training import run_training
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    cfg = CFG()
    set_seed(cfg.seed)

    print("\nLoading training data...")
    train_df = pd.read_csv(cfg.train_csv)
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)

    print("\nStarting training...")
    print(f"LOAD_DATA is set to {cfg.LOAD_DATA}")
    if cfg.LOAD_DATA:
        print("Using pre-computed mel spectrograms from NPY file")
    else:
        print("Will generate spectrograms on-the-fly during training")

    run_training(train_df, cfg)

    print("\nTraining complete!")
