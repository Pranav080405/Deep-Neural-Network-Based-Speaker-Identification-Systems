import numpy as np
import librosa

from tensorflow.keras.models import load_model

from utils.audio_processing import preprocess_audio
from utils.feature_extraction import extract_mfcc_features

MODEL_PATH = "models/speaker_dnn_mfcc_xfactor.h5"

LABEL_PATH = "models/label_encoder_mfcc_xfactor.npy"

model = load_model(MODEL_PATH)

labels = np.load(LABEL_PATH, allow_pickle=True)

audio_path = input("Enter audio file path: ")

y, sr = preprocess_audio(audio_path)

features = extract_mfcc_features(y, sr)

features = np.expand_dims(features, axis=0)

prediction = model.predict(features)

predicted_class = np.argmax(prediction)

confidence = np.max(prediction)

print("\nPredicted Speaker:", labels[predicted_class])

print(f"Confidence: {confidence*100:.2f}%")