# Structural Damage Detection using CWT-DCNN

## Project Overview

Structural damage detection is vital for ensuring the safety and integrity of buildings and infrastructures. This project presents a machine learning solution that classifies whether a three-story aluminium structure is damaged or undamaged based on vibration sensor data. We leverage Continuous Wavelet Transform (CWT) to convert 1D sensor signals into time-frequency images and train a deep convolutional neural network (DCNN) for robust damage classification.

## Key Features

- **Data Loading & Preprocessing**: Automated loading of accelerometer and force transducer data, label encoding, and signal scaling.
- **Time-Frequency Conversion**: Continuous Wavelet Transform (CWT) to generate 224×224 grayscale images from 1D vibration signals.
- **Deep Convolutional Neural Network**: A custom DCNN architecture inspired by state-of-the-art image classification models.
- **Training & Evaluation**: Model training with early stopping and checkpointing, followed by performance evaluation on a held-out test set.

---

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
   - [Loading Raw Signals](#loading-raw-signals)
   - [Signal Preprocessing](#signal-preprocessing)
   - [Time-Frequency Image Conversion](#time-frequency-image-conversion)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Usage](#usage)
8. [Results](#results)

---

## Directory Structure

```
project-root/
├── data/                    # Root folder containing state sub-folders with raw .txt signals
│   ├── state#01/
│   ├── state#02/
│   └── ...
├── models/                  # Saved model checkpoints and final model
│   └── structural_state_model.keras
├── main.py/                     # Source code
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/username/structural-damage-detection.git
   cd structural-damage-detection
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

**`requirements.txt`** should include:

```
numpy
scipy
scikit-learn
tensorflow
keras
pywt
pillow
```

---

## Data Preparation

### Loading Raw Signals

- **Function**: `load_data(data_root)`
- **Description**: Iterates through sub-folders in `data_root`, matching folder names to predefined `state_labels`. Parses sample numbers from filenames, loads `.txt` files into NumPy arrays, and collects corresponding labels.

### Signal Preprocessing

- **Function**: `preprocess_signal_data(data_samples, data_labels)`
- **Steps**:
  1. **Label Encoding**: Converts textual state labels into integer classes and one-hot encodes them.
  2. **Standardization**: Stacks all 1D signals, applies `StandardScaler` for zero-mean, unit-variance normalization, and restores individual sample shapes.

### Time-Frequency Image Conversion

- **Function**: `compute_cwt_image(signal, scales=None, wavelet_name='morl', output_size=(224,224))`

  - Computes Continuous Wavelet Transform (CWT) coefficients using PyWavelets (`pywt.cwt`).
  - Normalizes coefficients to the 0–255 range and converts to a grayscale image via PIL.
  - Resizes images to 224×224 pixels.

- **Function**: `create_image_dataset(signal_data)`

  - Applies `compute_cwt_image` to each preprocessed signal sample, logging progress every 5 samples.

---

## Model Architecture

Defined in `build_dcnn_model(input_shape, num_classes)`:

1. **Input Layer**: 224×224×1 grayscale image.
2. **Convolution Block 1**:
   - `Conv2D(96, 7×7, strides=2)`, ReLU activation.
   - Batch Normalization.
   - Max Pooling (3×3, strides=2).
3. **Convolution Block 2**:
   - `Conv2D(256, 5×5, strides=2)`, ReLU activation.
   - Batch Normalization.
   - Max Pooling (3×3, strides=2).
4. **Convolution Blocks 3–5**:
   - Three `Conv2D` layers with 384, 256, and 256 filters respectively, all 3×3 kernels, ReLU.
   - Max Pooling (2×2, strides=2).
5. **Fully Connected Layers**:
   - Flatten.
   - Dense(1024), ReLU.
   - Dropout(0.5).
   - Output Dense(num\_classes), Softmax.

**Optimizer**: Adam (learning rate=1e-4, decay=1e-3).
**Loss**: Categorical Crossentropy.
**Metrics**: Accuracy.

---

## Training

- **Entry Point**: `main()` in `src/main.py`.
- **Data Split**: 80% training, 20% testing (random state=42).
- **Validation**: 10% of training data used for validation.
- **Callbacks**:
  - `ModelCheckpoint`: Saves best model based on validation accuracy.
  - `EarlyStopping`: Monitors validation accuracy, patience=10 epochs, restores best weights.
- **Hyperparameters**:
  - Epochs: 50
  - Batch Size: 32

Run training with:

```bash
python src/main.py
```

---

## Evaluation

After training, the model is evaluated on the held-out test set. Key outputs:

- **Test Loss**
- **Test Accuracy**

Example log output:

```
Test Loss: 0.1234   Test Accuracy: 0.9567
```

The trained model is saved to `models/structural_state_model.keras`.

---

## Usage

To use the trained model for inference on new sensor data:

1. **Load the Model**:

   ```python
   from tensorflow.keras.models import load_model
   model = load_model('models/structural_state_model.keras')
   ```

2. **Preprocess New Data**:

   - Scale raw signals using the saved `StandardScaler`.
   - Convert to CWT images (224×224 grayscale).
   - Normalize pixel values to [0,1].

3. **Predict**:

   ```python
   predictions = model.predict(new_images)
   predicted_classes = predictions.argmax(axis=1)
   ```

---

## Results & Discussion

The proposed CWT-DCNN approach achieved high accuracy in distinguishing damaged vs. undamaged structural states. Time-frequency images effectively capture subtle vibration signatures induced by simulated damage conditions.

Future improvements may include:

- Experimenting with alternative time-frequency transforms (e.g., STFT, Wavelet Packet Transform).
- Incorporating data augmentation to improve generalization.
- Exploring transfer learning with pretrained image backbones.

---

##

