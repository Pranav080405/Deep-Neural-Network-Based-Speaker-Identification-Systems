
# Hybrid Speaker Identification and Verification System

A deep learning-based speaker recognition framework capable of performing both:

- **Speaker Identification** (Who is speaking?)
- **Speaker Verification** (Are these two voices from the same speaker?)

The system combines **MFCC-based acoustic feature extraction**, **ECAPA-TDNN speaker embeddings**, and a **Deep Neural Network (DNN)** classifier to achieve robust speaker recognition performance across multilingual and real-world speech conditions.

---

# Project Overview

This project was developed as part of a research work on **DNN Embedding-based Speaker Identification Systems** using an in-house multilingual Indian speech corpus.

The framework was designed to:
- Extract speaker-specific vocal characteristics
- Learn discriminative speaker embeddings
- Perform speaker classification using deep neural networks
- Verify speaker similarity using embedding-based cosine similarity scoring

The system supports:
- Hindi
- English
- Telugu
- Multilingual speaker combinations

and was evaluated on a dataset consisting of **51 speakers** collected from diverse linguistic and demographic backgrounds.

---
# End-to-End System Development

One of the most significant aspects of this project is that the entire pipeline was developed completely from scratch, including:

- Dataset collection
- Speech recording
- Data preprocessing
- Feature engineering
- Model training
- Performance evaluation
- Deployment

The project was not built using a pre-existing benchmark dataset alone. Instead, a custom multilingual speaker corpus was created manually for this research work.

---

# Custom Dataset Collection

The speech dataset was personally collected and curated during the project development process.

The dataset creation involved:
- recording voice samples from friends and volunteers
- collecting multilingual utterances
- ensuring speaker diversity
- organizing speaker-wise datasets
- manually verifying recordings

Each speaker contributed approximately:
- 20 spoken sentences
- across 1–3 languages depending on multilingual fluency

Languages included:
- English
- Hindi
- Telugu

The recordings were intentionally collected under varying:
- accents
- speaking styles
- intonations
- linguistic combinations

to improve real-world robustness of the system.

---

# Dataset Engineering Process

The dataset preparation pipeline included:

## Audio Recording
Voice samples were collected using:
- smartphone microphones
- laptop microphones
- online speech sources

---

## Audio Standardization

All recordings were:
- converted to mono
- resampled to 16 kHz
- normalized
- cleaned for noise artifacts

---

## Dataset Organization

The recordings were manually organized speaker-wise into structured directories for supervised learning.

Example:

```text
spk-01/
spk-02/
spk-03/
```

This enabled:
- label encoding
- speaker classification
- embedding learning

---

# End-to-End ML Pipeline Ownership

This project involved complete ownership of the entire machine learning workflow, including:

| Stage | Contribution |
|---|---|
| Dataset Collection | Self-collected multilingual speech corpus |
| Data Cleaning | Audio preprocessing and normalization |
| Feature Engineering | MFCC + ECAPA embeddings |
| Model Design | DNN architecture development |
| Training | TensorFlow/Keras model training |
| Evaluation | Accuracy, loss, EER analysis |
| Verification | Cosine similarity speaker verification |
| Deployment | Streamlit-based web application |
| Debugging | Real-world dataset contamination mitigation |

---

# Practical Engineering Challenges Solved

Several real-world engineering challenges were encountered and addressed during development:

- multilingual speaker variability
- noisy recordings
- speaker imbalance
- preprocessing inconsistencies
- embedding dimensionality alignment
- deployment compatibility
- dataset label contamination during recursive parsing

These challenges provided valuable experience in:
- ML system debugging
- audio AI engineering
- deployment-oriented model design
- practical deep learning workflows

---

# Research and Engineering Focus

This project was developed not only as a deep learning model, but as a complete research-oriented and deployment-ready speaker AI system.

The work combines concepts from:
- Digital Signal Processing (DSP)
- Speech Processing
- Deep Learning
- Speaker Biometrics
- Representation Learning
- AI Deployment Engineering

This end-to-end development approach significantly strengthened understanding of:
- real-world dataset engineering
- speaker embedding systems
- neural network optimization
- inference pipeline design
- production-oriented AI deployment

---


# Key Features

## Speaker Identification
Classifies an uploaded voice sample into one of the known trained speaker classes using:
- MFCC features
- ECAPA embeddings
- DNN classifier

## Speaker Verification
Verifies whether two uploaded voice samples belong to the same speaker using:
- ECAPA-TDNN embeddings
- Cosine similarity scoring

## Audio Preprocessing
Includes:
- Noise reduction
- Silence trimming
- Audio normalization
- 16 kHz mono conversion

## Deep Learning Pipeline
Implements:
- MFCC + Δ + ΔΔ feature extraction
- Speaker embedding generation
- DNN classification
- Similarity-based verification

## Streamlit Deployment
Interactive web-based interface allowing:
- Audio upload
- Real-time prediction
- Similarity scoring
- Confidence visualization

---

# System Architecture

## Speaker Identification Pipeline

```text
Audio Input
    ↓
Preprocessing
    ↓
MFCC Feature Extraction
    ↓
ECAPA-TDNN Embeddings
    ↓
Feature Concatenation
    ↓
DNN Classifier
    ↓
Predicted Speaker
```

---

## Speaker Verification Pipeline

```text
Audio Sample 1
        ↓
ECAPA Embedding

Audio Sample 2
        ↓
ECAPA Embedding
        ↓
Cosine Similarity
        ↓
Same / Different Speaker
```

---

# Dataset Description

The project uses an in-house multilingual speech corpus consisting of:

| Parameter | Details |
|---|---|
| Total Speakers | 51 |
| Languages | English, Hindi, Telugu |
| Gender | Male and Female |
| Age Group | 17–60 years |
| Sampling Rate | 16 kHz Mono |
| Utterances | 20 per speaker per language |
| Audio Duration | 15–20 seconds |

The dataset contains:
- direct microphone recordings
- extracted YouTube speech samples
- multilingual and accent-diverse utterances

This diversity improves robustness and generalization capability under real-world conditions. :contentReference[oaicite:1]{index=1}

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| TensorFlow / Keras | DNN modeling |
| PyTorch | ECAPA embedding extraction |
| SpeechBrain | Pretrained ECAPA-TDNN model |
| Librosa | Audio processing + MFCC extraction |
| NumPy | Numerical operations |
| scikit-learn | Label encoding + metrics |
| Matplotlib | Training visualization |
| Streamlit | Web deployment |
| Google Colab | Initial model training |

<img width="1280" height="756" alt="telegram-cloud-photo-size-5-6084773876299666399-y" src="https://github.com/user-attachments/assets/11bfdc17-7695-4d9b-8103-caf37eba38f5" />

<img width="1280" height="756" alt="telegram-cloud-photo-size-5-6084773876299666400-y" src="https://github.com/user-attachments/assets/c2328ba8-9587-4543-b6dc-003920f510ab" />

---

# Feature Extraction

## MFCC Features

The system extracts:
- 20 MFCC coefficients
- Delta features (Δ)
- Delta-Delta features (ΔΔ)

MFCCs are widely used in speech processing because they:
- approximate human auditory perception
- capture spectral characteristics
- provide robustness under noisy conditions

The addition of Δ and ΔΔ features helps model temporal speech dynamics.

---

# ECAPA-TDNN Speaker Embeddings

The system integrates pretrained **ECAPA-TDNN embeddings** using the SpeechBrain framework.

These embeddings:
- encode speaker-specific vocal characteristics
- generalize well to unseen speakers
- improve verification robustness

ECAPA embeddings are used for:
- feature augmentation in DNN classification
- cosine similarity-based speaker verification

---

# Deep Neural Network Architecture

The DNN classifier consists of:

- Dense Layer (256 neurons)
- Dropout (0.3)
- Dense Layer (128 neurons)
- Dropout (0.3)
- Softmax Output Layer

Activation Functions:
- ReLU
- Softmax

Optimizer:
- Adam Optimizer

Loss Function:
- Categorical Crossentropy

---

# Training Details

The model was trained using:
- 50 epochs
- Batch size = 16
- 80/20 train-test split

Training was conducted on Google Colab using GPU acceleration.

---

# Performance Metrics

The system was evaluated using:
- Accuracy
- Loss
- Equal Error Rate (EER)
- Cosine Similarity Scores

---

# Experimental Results

| Epochs | Accuracy | EER |
|---|---|---|
| 10 | 90.25% | 9.5% |
| 20 | 91.34% | 8.2% |
| 30 | 91.73% | 7.5% |
| 40 | 91.52% | 6.8% |
| 50 | 91.26% | 6.2% |

The model demonstrated:
- stable convergence
- strong generalization
- low overfitting
- reliable speaker discrimination performance

# Training Visualization and Performance Analysis

The following plots illustrate the convergence behavior, training stability, and generalization capability of the proposed DNN-based speaker identification system.

---

## Accuracy vs Epoch

The graph below shows the variation of training and validation accuracy across 50 epochs.

- Training accuracy steadily improves during the early epochs.
- Validation accuracy consistently remains close to or slightly higher than training accuracy.
- The convergence of both curves near 91–92% indicates stable learning and strong generalization without significant overfitting.

![Accuracy vs Epoch](assets/accuracy_vs_epoch.png)

---

## Epochs vs Accuracy

This visualization highlights the progressive increase in classification accuracy with increasing epochs.

The model demonstrates:
- rapid convergence during initial epochs
- stable performance after convergence
- balanced learning behavior

![Epochs vs Accuracy](assets/epochs_vs_accuracy.png)

---

## Model Loss vs Epoch

The loss curves show a rapid reduction in both training and validation loss during early training stages, followed by gradual stabilization.

Observations:
- validation loss remains consistently low
- smooth convergence behavior
- no major oscillations or divergence
- effective optimization using Adam optimizer

![Model Loss vs Epoch](assets/model_loss_vs_epoch.png)

---

## Precision vs Model Loss

This graph illustrates the relationship between validation loss and precision.

As validation loss decreases:
- precision improves significantly
- the model becomes increasingly discriminative
- speaker-specific embedding quality improves

The clustering of points near high precision and low loss demonstrates the robustness of the proposed framework.

![Precision vs Model Loss](assets/precision_vs_model_loss.png)

---

# Model Behavior Analysis

The experimental results indicate:

- Strong convergence behavior
- Stable optimization dynamics
- Effective feature learning
- Minimal overfitting
- Robust generalization capability

The close alignment between training and validation curves confirms that the model learns discriminative speaker representations while maintaining reliable performance on unseen data.

The final system achieved:
- ~91% speaker identification accuracy
- Reduced Equal Error Rate (EER)
- Reliable multilingual speaker discrimination
- Strong verification similarity performance

These results validate the effectiveness of combining:
- MFCC-based acoustic features
- ECAPA-TDNN embeddings
- Deep neural network classification
- Cosine similarity verification

:contentReference[oaicite:2]{index=2}

---

# Verification Performance

The speaker verification module uses cosine similarity scoring.

Typical similarity interpretation:

| Similarity Score | Interpretation |
|---|---|
| 0.85 – 1.00 | Very High Match |
| 0.75 – 0.85 | Moderate Match |
| 0.65 – 0.75 | Weak Match |
| < 0.65 | Different Speakers |

This verification system generalizes well even to previously unseen speakers.

---

# Real-World Applications

This framework can be applied in:

- Voice biometric authentication
- Secure banking systems
- Smart home voice interfaces
- Call center speaker verification
- Forensic speaker analysis
- IoT voice-enabled systems
- Personalized virtual assistants

:contentReference[oaicite:3]{index=3}

---

# Challenges Addressed

The system was designed to handle:
- multilingual speech
- varying accents
- noisy recordings
- channel variability
- speaker diversity

During deployment, a dataset-label contamination issue was identified due to recursive utility-folder parsing. Instead of retraining the entire system, an inference-time filtering mechanism was engineered to ignore invalid utility classes during prediction while preserving the original trained model.

This reflects a practical real-world ML engineering mitigation strategy.

---

# Streamlit Deployment

The project includes an interactive Streamlit application with:

## Speaker Identification Tab
- Upload voice sample
- Predict speaker identity
- Confidence visualization

## Speaker Verification Tab
- Upload two voice samples
- Compute similarity score
- Same/Different speaker decision

---

# Project Structure

```text
speaker-identification-dnn/
│
├── app.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── speaker_dnn_mfcc_xfactor.h5
│   └── label_encoder_mfcc_xfactor.npy
│
├── utils/
│   ├── audio_processing.py
│   ├── feature_extraction.py
│   └── model_utils.py
│
└── assets/
```

---

# Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

# Future Improvements

Future work may include:
- Transformer-based speaker models
- Anti-spoofing mechanisms
- Real-time microphone inference
- Edge AI deployment
- FPGA acceleration
- Domain adaptation
- Noise-robust embeddings
- Mobile deployment

---

# Research Publication

This project is based on the research paper:

**"Performance Evaluation of DNN Embedding-based Speaker Identification System"**

The work evaluates:
- DNN embedding architectures
- multilingual speaker corpora
- MFCC feature extraction
- PLDA-based scoring
- convergence behavior across epochs

:contentReference[oaicite:4]{index=4}

# Local Installation and Execution

## Clone the Repository

```bash
git clone https://github.com/Pranav080405/Deep-Neural-Network-Based-Speaker-Identification-Systems.git
```

---

## Navigate to the Project Directory

```bash
cd Deep-Neural-Network-Based-Speaker-Identification-Systems
```

---

## Create Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Required Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application Locally

```bash
streamlit run app.py
```

---

## Open in Browser

After execution, Streamlit automatically launches the application in the browser.

If it does not open automatically, use:

```text
http://localhost:8501
```

---

# Features Available in Local Deployment

The local application supports:

- Speaker Identification
- Speaker Verification
- MFCC Feature Extraction
- ECAPA-TDNN Embedding Extraction
- Deep Neural Network Inference
- Cosine Similarity Based Verification
- Confidence Score Estimation
- Real-Time Audio Upload and Processing

---

# Project Structure

```text
.
├── app.py
├── requirements.txt
├── models/
│   ├── speaker_dnn_mfcc_xfactor.h5
│   └── label_encoder_mfcc_xfactor.npy
├── utils/
│   ├── audio_processing.py
│   ├── feature_extraction.py
│   └── model_utils.py
├── assets/
├── README.md
└── train.py
```

---

# Important Notes

- The deployed version was tested primarily in a local environment using Streamlit.
- Due to runtime compatibility issues between TensorFlow, SpeechBrain, and certain cloud deployment environments, the local deployment remains the most stable and reliable execution method for demonstration purposes.
- The complete inference pipeline functions correctly in local execution.

=======
