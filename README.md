# Deep-Neural-Network-Based-Speaker-Identification-Systems
Deep learning-based speaker identification system using x-vector (TDNN) embeddings and Probabilistic Linear Discriminant Analysis (PLDA). Includes MFCC feature extraction, custom dataset preprocessing, and evaluation based on accuracy and EER.
 here is the code that i executed for the model:

!pip install -q speechbrain torchaudio librosa tqdm noisereduce pydub tensorflow scikit-learn matplotlib

import os
import numpy as np
import torch
import librosa
import noisereduce as nr
from tqdm import tqdm
from pydub import AudioSegment
from speechbrain.pretrained import EncoderClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive', force_remount=False)

base_dir = "/content/drive/MyDrive/voice_data"
converted_dir = os.path.join(base_dir, "converted_wav")
os.makedirs(converted_dir, exist_ok=True)


def convert_to_wav(file_path, out_dir):
    try:
        file_name = os.path.basename(file_path)
        new_path = os.path.join(out_dir, os.path.splitext(file_name)[0] + ".wav")
        if os.path.exists(new_path):
            return new_path
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(new_path, format="wav")
        return new_path
    except Exception as e:
        print(f"❌ Conversion failed for {file_path}: {e}")
        return None

def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000, mono=True)
        y_denoised = nr.reduce_noise(y=y, sr=sr)
        y_trimmed, _ = librosa.effects.trim(y_denoised, top_db=25)
        y_norm = librosa.util.normalize(y_trimmed)
        return y_norm, sr
    except Exception as e:
        print(f"⚠️ Skipping {file_path}: {e}")
        return None, None

def extract_mfcc_features(y, sr, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=int(0.010*sr), n_fft=int(0.025*sr))
    mfcc = mfcc.T
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=1)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    pooled = np.concatenate([mean, std])
    return pooled

def extract_features(base_dir, use_xfactor=False):
    print(f"📂 Scanning {base_dir} recursively for audio files...\n")
    if use_xfactor:
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    features, labels = [], []
    valid_exts = ('.wav', '.m4a', '.mp3', '.flac')
    all_files = []

    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                all_files.append(os.path.join(root, f))

    print(f"🎧 Found {len(all_files)} audio files in total.\n")

    for file_path in tqdm(all_files, desc="🎵 Processing audio"):
        if not file_path.lower().endswith('.wav'):
            file_path = convert_to_wav(file_path, converted_dir)
            if file_path is None:
                continue

        y, sr = preprocess_audio(file_path)
        if y is None:
            continue

        mfcc_feat = extract_mfcc_features(y, sr)

        if use_xfactor:
            try:
                audio_tensor = torch.tensor(y).unsqueeze(0)
                emb = classifier.encode_batch(audio_tensor).squeeze().detach().numpy()
                combined_feat = np.concatenate([mfcc_feat, emb])
            except Exception as e:
                print(f"❌ Error extracting ECAPA embeddings {file_path}: {e}")
                continue
        else:
            combined_feat = mfcc_feat

        label = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        parent_dir = os.path.basename(os.path.dirname(file_path))
        speaker_label = label if label.startswith("spk-") else parent_dir

        features.append(combined_feat)
        labels.append(speaker_label)

    features = np.array(features)
    labels = np.array(labels)

    if len(features) == 0:
        print("❌ No valid features found.")
    else:
        print(f"✅ Extracted {len(features)} feature vectors of shape {features.shape}")

    return features, labels

def train_dnn(X, y, epochs=50, batch_size=16):  # epochs increased from 20 → 50
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    y_cat = to_categorical(y_enc)

    X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(256, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(y_cat.shape[1], activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # ⏳ Save history for plotting
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=batch_size, verbose=1)

    # ✅ Evaluate model
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

    # 📊 Plot Epochs vs Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Epochs vs Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, le

X, y = extract_features(base_dir, use_xfactor=True)

if len(X) == 0:
    print("❌ No features extracted. Check folder structure or audio formats.")
else:
    print(f"✅ Final dataset: {len(X)} samples, feature dim = {X.shape[1]}")
    model, label_encoder = train_dnn(X, y, epochs=50, batch_size=16)

    model.save(os.path.join(base_dir, "speaker_dnn_mfcc_xfactor.h5"))
    np.save(os.path.join(base_dir, "label_encoder_mfcc_xfactor.npy"), label_encoder.classes_)
    print("✅ Model and label encoder saved successfully.")

