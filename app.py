import streamlit as st
import numpy as np
import tempfile
import torch
import torch.nn.functional as F

from tensorflow.keras.models import load_model
from speechbrain.pretrained import EncoderClassifier

from utils.audio_processing import preprocess_audio
from utils.feature_extraction import extract_mfcc_features


# Load DNN Model + Labels

MODEL_PATH = "models/speaker_dnn_mfcc_xfactor.h5"

LABEL_PATH = "models/label_encoder_mfcc_xfactor.npy"

model = load_model(MODEL_PATH)

labels = np.load(LABEL_PATH, allow_pickle=True)


# Load ECAPA Embedding Model

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb"
)

# Streamlit UI


st.title("Speaker Identification & Verification System")

st.write(
    "Deep learning based speaker recognition system using "
    "MFCC features, ECAPA embeddings, and DNN classification."
)



st.markdown("---")

st.markdown("""
### System Features

- MFCC Feature Extraction
- ECAPA-TDNN Speaker Embeddings
- Deep Neural Network Classification
- Speaker Verification using Cosine Similarity
- Audio Preprocessing and Noise Reduction
""")



tab1, tab2 = st.tabs([
    "Speaker Identification",
    "Speaker Verification"
])


# TAB 1 — SPEAKER IDENTIFICATION

with tab1:

    st.header("Speaker Identification")

    st.write("Upload a voice sample to identify the speaker.")

    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["wav", "mp3", "m4a", "flac"],
        key="identification"
    )

    if uploaded_file is not None:

        st.audio(uploaded_file)

        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:

            tmp_file.write(uploaded_file.read())

            temp_path = tmp_file.name

        # Preprocess audio
        y, sr = preprocess_audio(temp_path)

        if y is None:

            st.error("Error processing audio.")

        else:


            # MFCC Features

            mfcc_feat = extract_mfcc_features(y, sr)

  
            # ECAPA Embeddings
         
            audio_tensor = torch.tensor(y).unsqueeze(0)

            emb = classifier.encode_batch(audio_tensor)

            emb = emb.squeeze().detach().numpy()

    
            # Combine Features
           

            features = np.concatenate([mfcc_feat, emb])

            features = np.expand_dims(features, axis=0)

           
            prediction = model.predict(features)

            prediction = prediction.flatten()


            sorted_indices = np.argsort(prediction)[::-1]

  
            for idx in sorted_indices:

                label = labels[idx]

                if label != "converted_wav":

                    predicted_class = idx

                    predicted_label = label.split(",")[0]

                    confidence = prediction[idx]

                    break

           
            st.success(f"Predicted Speaker: {predicted_label}")

            st.info(f"Confidence: {confidence*100:.2f}%")

            st.progress(float(confidence))

            # Optional low-confidence warning
            if confidence < 0.40:

                st.warning("Low confidence prediction")


# TAB 2 — SPEAKER VERIFICATION


with tab2:

    st.header("Speaker Verification")

    st.write(
        "Upload two audio samples to verify whether "
        "they belong to the same speaker."
    )

    audio1 = st.file_uploader(
        "Upload First Audio",
        type=["wav", "mp3", "m4a", "flac"],
        key="audio1"
    )

    audio2 = st.file_uploader(
        "Upload Second Audio",
        type=["wav", "mp3", "m4a", "flac"],
        key="audio2"
    )

    if audio1 is not None and audio2 is not None:

        st.subheader("Uploaded Audio Samples")

        st.audio(audio1)

        st.audio(audio2)

        
        with tempfile.NamedTemporaryFile(delete=False) as tmp1:

            tmp1.write(audio1.read())

            path1 = tmp1.name

        with tempfile.NamedTemporaryFile(delete=False) as tmp2:

            tmp2.write(audio2.read())

            path2 = tmp2.name

       
        # Preprocess audio
        y1, sr1 = preprocess_audio(path1)

        y2, sr2 = preprocess_audio(path2)

        if y1 is None or y2 is None:

            st.error("Error processing audio.")

        else:


            # Convert to tensors
            signal1 = torch.tensor(y1).unsqueeze(0)

            signal2 = torch.tensor(y2).unsqueeze(0)

            emb1 = classifier.encode_batch(signal1)

            emb2 = classifier.encode_batch(signal2)

            # Cosine Similarity
            similarity = F.cosine_similarity(
                emb1.squeeze(1),
                emb2.squeeze(1)
            ).item()

            st.subheader("Verification Result")

            st.write(f"Similarity Score: {similarity:.4f}")

            # Similarity interpretation
            if similarity > 0.85:

                st.success("Very High Speaker Match")

            elif similarity > 0.75:

                st.info("Moderate Speaker Match")

            elif similarity > 0.65:

                st.warning("Weak Speaker Match")

            else:

                st.error("Different Speakers")

            # Final decision
            THRESHOLD = 0.75

            if similarity > THRESHOLD:

                st.success("Same Speaker Detected")

            else:

                st.error("Different Speakers")

            # Similarity progress bar
            st.progress(min(max(similarity, 0.0), 1.0))