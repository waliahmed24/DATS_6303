import os
import shutil
import random
import pandas as pd

# Paths
train_audio_dir = '../Data/train_audio'
test_soundscapes_dir = '../Data/test_soundscapes'
metadata_csv_path = '../Data/test_metadata.csv'
taxonomy_csv_path = '../Data/taxonomy.csv'
num_files = 700

# Load taxonomy to enrich labels
taxonomy_df = pd.read_csv(taxonomy_csv_path)
taxonomy_df = taxonomy_df.set_index('primary_label')  # folder name = primary_label

# Create test folder
os.makedirs(test_soundscapes_dir, exist_ok=True)

# Gather all .ogg files and their labels (primary_label)
ogg_entries = []
for root, _, files in os.walk(train_audio_dir):
    for file in files:
        if file.endswith('.ogg'):
            primary_label = os.path.basename(root)
            full_path = os.path.join(root, file)
            ogg_entries.append((full_path, primary_label))

# Random sample
selected = random.sample(ogg_entries, min(num_files, len(ogg_entries)))

# Prepare metadata
metadata = []
for idx, (src_path, label) in enumerate(selected, 1):
    new_name = f"soundscape_{idx:06d}.ogg"
    dest_path = os.path.join(test_soundscapes_dir, new_name)
    shutil.copy(src_path, dest_path)

    row = taxonomy_df.loc[label] if label in taxonomy_df.index else {}
    entry = {
        'filename': new_name,
        'primary_label': label
    }

    # Add extra fields if available
    for col in ['scientific_name', 'common_name', 'family', 'order']:
        entry[col] = row[col] if col in taxonomy_df.columns else None

    metadata.append(entry)

# Save metadata
df = pd.DataFrame(metadata)
df.to_csv(metadata_csv_path, index=False)

print(f"Created {len(df)} test soundscapes and metadata CSV at {metadata_csv_path}")
