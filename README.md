# DATS_6303

# Final Project - Group 6

This project focuses on training a deep learning model using mel spectrogram images to classify audio data. The dataset used consists of precomputed mel spectrogram images, and the training pipeline is implemented in Python using popular deep learning libraries.


## 🚀 Getting Started

Follow the instructions below to set up and run the project on your machine.

### 1. Clone the Repository

```bash
git clone https://github.com/waliahmed24/Final_Project_Group6.git
```

### 2. Navigate to The Project Directory

```bash
cd Final_Project_Group6
```

### 3. Download The Dataset

```bash
wget -O Data.zip "https://www.dropbox.com/scl/fi/8jd9okreb87ojim7p513u/Data.zip?rlkey=islyrmkgagvnoxd1rdyolcslv&st=w23hljqi&dl=1"
```

### 4. Unzip The Dataset

```bash
unzip Data.zip
```

### 5. Navigate to the Code Directory

```bash
cd Code
```

### 6. Create a Virtual Environment and Install Dependencies

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### 7. Run the Training Script

```bash
python3 main.py
```


This would run training and validation and once done will save the best model based on the validation set accuracy. Refer to the project Directory structure below within the Code folder:

<img width="490" alt="Screenshot 2025-05-01 at 3 16 48 PM" src="https://github.com/user-attachments/assets/83444649-47a8-4d64-be02-6203ad3d48cd" />














