import librosa
import noisereduce as nr

def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000, mono=True)

        # Noise reduction
        y_denoised = nr.reduce_noise(y=y, sr=sr)

        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y_denoised, top_db=25)

        # Normalize
        y_norm = librosa.util.normalize(y_trimmed)

        return y_norm, sr

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None