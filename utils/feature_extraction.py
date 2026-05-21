import os
import numpy as np
import librosa
import torch

from tqdm import tqdm
from speechbrain.pretrained import EncoderClassifier

from utils.audio_processing import preprocess_audio

def extract_mfcc_features(y, sr, n_mfcc=20):

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        hop_length=int(0.010 * sr),
        n_fft=int(0.025 * sr)
    )

    mfcc = mfcc.T

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate([mfcc, delta, delta2], axis=1)

    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)

    pooled = np.concatenate([mean, std])

    return pooled


def extract_features(base_dir, use_xfactor=True):

    print(f"Scanning {base_dir} recursively...")

    classifier = None

    if use_xfactor:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )

    features = []
    labels = []

    valid_exts = ('.wav', '.mp3', '.flac', '.m4a')

    all_files = []

    for root, _, files in os.walk(base_dir):

        for f in files:

            if f.lower().endswith(valid_exts):

                all_files.append(os.path.join(root, f))

    print(f"Found {len(all_files)} audio files")

    for file_path in tqdm(all_files):

        y, sr = preprocess_audio(file_path)

        if y is None:
            continue

        mfcc_feat = extract_mfcc_features(y, sr)

        if use_xfactor:

            try:
                audio_tensor = torch.tensor(y).unsqueeze(0)

                emb = classifier.encode_batch(audio_tensor)

                emb = emb.squeeze().detach().numpy()

                combined_feat = np.concatenate([mfcc_feat, emb])

            except Exception as e:
                print(f"Embedding error: {e}")
                continue

        else:
            combined_feat = mfcc_feat

        speaker_label = os.path.basename(os.path.dirname(file_path))

        features.append(combined_feat)
        labels.append(speaker_label)

    features = np.array(features)
    labels = np.array(labels)

    print(f"Extracted features shape: {features.shape}")

    return features, labels