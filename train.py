import os
import numpy as np

from utils.feature_extraction import extract_features
from utils.model_utils import train_dnn

BASE_DIR = "voice_data"

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

X, y = extract_features(BASE_DIR, use_xfactor=True)

if len(X) == 0:

    print("No features extracted.")

else:

    print(f"Dataset size: {len(X)}")

    model, label_encoder = train_dnn(
        X,
        y,
        epochs=50,
        batch_size=16
    )

    model.save(
        os.path.join(
            MODEL_DIR,
            "speaker_dnn_mfcc_xfactor.h5"
        )
    )

    np.save(
        os.path.join(
            MODEL_DIR,
            "label_encoder.npy"
        ),
        label_encoder.classes_
    )

    print("Model saved successfully.")