
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = '../Data'
train_audio_dir = os.path.join(DATA_DIR, 'train_audio')
test_soundscapes_dir = os.path.join(DATA_DIR, 'test_soundscapes')
train_csv_path = os.path.join(DATA_DIR, 'train.csv')
train_split_csv_path = os.path.join(DATA_DIR, 'train_split.csv')
test_split_csv_path = os.path.join(DATA_DIR, 'test_split.csv')
test_metadata_csv_path = os.path.join(DATA_DIR, 'test_metadata.csv')

# Parameters
test_size_ratio = 0.2  # 20% validation
random_seed = 42

# Step 1: Load the full train.csv
print("Loading original train.csv...")
train_df = pd.read_csv(train_csv_path)

# Step 2: Stratified split into train / valid
print(f"Splitting into {100*(1-test_size_ratio):.0f}% train and {100*test_size_ratio:.0f}% valid...")
train_split, test_split = train_test_split(
    train_df,
    test_size=test_size_ratio,
    stratify=train_df['primary_label'],
    random_state=random_seed
)

# Step 3: Save split CSVs
train_split.to_csv(train_split_csv_path, index=False)
test_split.to_csv(test_split_csv_path, index=False)
print(f"Saved train_split.csv ({len(train_split)}) and test_split.csv ({len(test_split)})")

# Step 4: Move validation audio files to test_soundscapes
print(f"Copying validation audio files into {test_soundscapes_dir}...")
os.makedirs(test_soundscapes_dir, exist_ok=True)

metadata = []
for idx, row in test_split.iterrows():
    original_file = os.path.join(train_audio_dir, row['filename'])
    new_name = f"soundscape_{idx:06d}.ogg"
    new_path = os.path.join(test_soundscapes_dir, new_name)

    if os.path.exists(original_file):
        shutil.copy(original_file, new_path)
    else:
        print(f"Warning: Missing file {original_file}")

    metadata.append({
        'filename': new_name,
        'primary_label': row['primary_label'],
    })

# Step 5: Save test metadata CSV
test_meta_df = pd.DataFrame(metadata)
test_meta_df.to_csv(test_metadata_csv_path, index=False)
print(f"Saved test_metadata.csv ({len(test_meta_df)} entries)")

print("\nDone! The test set is now separated into test_soundscapes.")
#========================================================================#
