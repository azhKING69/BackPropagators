import os
import glob
import logging
import numpy as np
import pywt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -------------------------------
# CONFIGURATION & LOGGING
# -------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
DATA_ROOT = "data"  # Root folder with sub-folders for each state
MODEL_SAVE_DIR = "models"
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "structural_state_model.keras")

# -------------------------------
# STEP 1: Define the label mapping
# -------------------------------
state_labels = {
    "state#01": {"label": "Mass on the 1st floor", "samples": list(range(11, 21))},
    "state#02": {"label": "Mass at the base", "samples": list(range(21, 31))},
    "state#08": {"label": "Gap=0.13mm", "samples": list(range(160, 170))},
    "state#09": {"label": "Gap=0.10mm", "samples": list(range(170, 180))},
    "state#10": {"label": "Gap=0.05mm", "samples": list(range(180, 190))},
    "state#11": {"label": "Gap=0.15mm", "samples": list(range(190, 200))},
    "state#12": {"label": "Gap=0.20mm", "samples": list(range(200, 210))},
    "state#13": {"label": "Baseline condition", "samples": list(range(210, 220))},
    "state#14": {"label": "Gap=0.20mm + mass on the 1st floor", "samples": list(range(220, 230))},
    "state#15": {"label": "Gap=0.10mm + mass on the 1st floor", "samples": list(range(230, 240))},
    "state#16": {"label": "Gap=0.20mm + mass at the base", "samples": list(range(240, 250))},
    "state#17": {"label": "Column: 1BD – 50% stiffness reduction", "samples": list(range(251, 261))},
    "state#18": {"label": "Column: 1AD + 1BD – 50% stiffness reduction", "samples": list(range(261, 271))},
    "state#21": {"label": "Column: 3BD – 50% stiffness reduction", "samples": list(range(291, 301))},
    "state#22": {"label": "Column: 3AD + 3BD – 50% stiffness reduction", "samples": list(range(302, 312))},
    "state#23": {"label": "Column: 2AD + 2BD – 50% stiffness reduction", "samples": list(range(312, 322))},
    "state#24": {"label": "Column: 2BD – 50% stiffness reduction", "samples": list(range(322, 332))}
}

# -------------------------------
# STEP 2: Load and preprocess the dataset (signal level)
# -------------------------------
def load_data(data_root=DATA_ROOT):
    """
    Loads 1D sensor signal text files from folders (states).
    Each folder name must be in state_labels.
    Files are filtered based on sample number parsed from filename.
    """
    data_samples = []
    labels = []
    for state_folder in os.listdir(data_root):
        folder_path = os.path.join(data_root, state_folder)
        if os.path.isdir(folder_path) and state_folder in state_labels:
            mapping = state_labels[state_folder]
            sample_range = mapping["samples"]
            label_str = mapping["label"]
            logging.info(f"Processing folder '{state_folder}' with sample range {sample_range} and label: {label_str}")
            file_pattern = os.path.join(folder_path, "data*.txt")
            for file_path in glob.glob(file_pattern):
                filename = os.path.basename(file_path)
                sample_num_str = "".join(filter(str.isdigit, filename))
                try:
                    sample_num = int(sample_num_str)
                except ValueError:
                    logging.warning(f"Could not parse sample number from {filename}")
                    continue
                if sample_num in sample_range:
                    try:
                        data_array = np.loadtxt(file_path)
                        data_samples.append(data_array)
                        labels.append(label_str)
                        logging.info(f"Loaded {file_path} with label: {label_str}")
                    except Exception as e:
                        logging.error(f"Error loading {file_path}: {e}")
                        continue
    if len(data_samples) == 0 or len(labels) == 0:
        raise ValueError("No data was loaded. Verify your folder structure and file naming.")
    return np.array(data_samples), np.array(labels)

def preprocess_signal_data(data_samples, data_labels):
    """
    Performs label encoding and scales the 1D sensor signals.
    This preprocessing is applied before converting signals to images.
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(data_labels)
    logging.info("Encoded label mapping: %s", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
    onehot_labels = to_categorical(encoded_labels)

    scaler = StandardScaler()
    data_samples_stacked = np.vstack(data_samples)
    data_samples_scaled = scaler.fit_transform(data_samples_stacked)
    # Restore individual samples
    data_samples_processed = []
    idx = 0
    for sample in data_samples:
        length = sample.shape[0]
        data_samples_processed.append(data_samples_scaled[idx:idx+length])
        idx += length
    data_samples_processed = np.array(data_samples_processed)
    return data_samples_processed, onehot_labels, label_encoder, scaler

# -------------------------------
# STEP 3: Convert 1D signals to Time-Frequency Images using CWT
# -------------------------------
def compute_cwt_image(signal, scales=None, wavelet_name='morl', output_size=(224, 224)):
    """
    Computes the Continuous Wavelet Transform (CWT) of a 1D signal and returns a grayscale image.
    
    If 'scales' is not provided, it defaults to np.arange(1, min(129, len(signal)+1)).
    The resulting coefficient matrix is normalized and forced to be 2D.
    """
    # Automatically adjust scales if signal is short
    if scales is None:
        scales = np.arange(1, min(129, len(signal)+1))
    
    coefficients, _ = pywt.cwt(signal, scales, wavelet_name)
    # Normalize coefficients to [0, 255]
    coeff_min, coeff_max = np.min(coefficients), np.max(coefficients)
    if coeff_max != coeff_min:
        coeff_norm = (coefficients - coeff_min) / (coeff_max - coeff_min)
    else:
        coeff_norm = coefficients
    coeff_image = (coeff_norm * 255).astype(np.uint8)
    
    # Squeeze out singleton dimensions
    coeff_image = np.squeeze(coeff_image)
    
    # Force the array to be 2D:
    if coeff_image.ndim == 1:
        coeff_image = np.expand_dims(coeff_image, axis=0)
    elif coeff_image.ndim > 2:
        coeff_image = coeff_image.reshape(coeff_image.shape[0], -1)
    
    try:
        img = Image.fromarray(coeff_image)
    except Exception as e:
        logging.error(f"Error converting coefficient array to image: {e}")
        raise e
    img = img.resize(output_size, resample=Image.BILINEAR)
    img_array = np.array(img)
    # Ensure the image has a channel dimension
    if img_array.ndim == 2:
        img_array = np.expand_dims(img_array, axis=-1)
    return img_array

def create_image_dataset(signal_data):
    """
    Converts all 1D signal samples to 224x224 time-frequency images using CWT.
    Returns a numpy array with shape: (num_samples, 224, 224, 1)
    Logs progress after every image conversion.
    """
    image_list = []
    for idx, sample in enumerate(signal_data):
        img = compute_cwt_image(sample)
        image_list.append(img)
        logging.info(f"Converted {idx+1}/{len(signal_data)} samples to images.")
    return np.array(image_list)

# -------------------------------
# STEP 4: Build the Deep Convolutional Neural Network (DCNN)
# -------------------------------
def build_dcnn_model(input_shape, num_classes):
    """
    Builds the CWT-DCNN model architecture as described in the article.
    The architecture includes several Conv2D layers, BatchNormalization,
    MaxPooling, Dropout and Dense layers with a final softmax.
    """
    inputs = Input(shape=input_shape)  # e.g., (224, 224, 1)
    
    # Convolution Block 1
    x = Conv2D(96, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    
    # Convolution Block 2
    x = Conv2D(256, kernel_size=(5,5), strides=(2,2), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
    
    # Convolution Blocks 3, 4 and 5 using 3x3 kernels
    x = Conv2D(384, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(x)
    x = Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same')(x)
    
    # Fully Connected Layers
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Use updated optimizer parameter names: learning_rate instead of lr, and remove decay.
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary(print_fn=logging.info)
    
    return model

# -------------------------------
# STEP 5: Main Pipeline: Load, Preprocess, Convert Signals to Images, Train & Evaluate
# -------------------------------
def main():
    # Load raw 1D sensor signal data and labels
    logging.info("Loading data...")
    signals, labels = load_data(DATA_ROOT)
    
    # Preprocess the raw 1D signals (scaling and label encoding)
    logging.info("Preprocessing signal data...")
    signals_processed, onehot_labels, label_encoder, scaler = preprocess_signal_data(signals, labels)
    
    # Split data into training and testing sets (80/20 split)
    X_train_signals, X_test_signals, y_train, y_test = train_test_split(
        signals_processed, onehot_labels, test_size=0.2, random_state=42)
    
    # Convert training signals to time-frequency images via CWT
    logging.info("Converting training signals to time-frequency images using CWT...")
    X_train_images = create_image_dataset(X_train_signals)
    logging.info("Converting testing signals to time-frequency images using CWT...")
    X_test_images = create_image_dataset(X_test_signals)
    
    # Normalize image pixel values to [0, 1]
    X_train_images = X_train_images.astype('float32') / 255.0
    X_test_images = X_test_images.astype('float32') / 255.0
    
    # Get input shape and number of classes for the model
    input_shape = X_train_images.shape[1:]  # e.g., (224, 224, 1)
    num_classes = y_train.shape[1]
    
    # Build the DCNN model
    logging.info("Building the DCNN model...")
    model = build_dcnn_model(input_shape, num_classes)
    
    # Define callbacks for saving the best model and early stopping
    checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    earlystop = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True)
    callbacks_list = [checkpoint, earlystop]
    
    # Train the model
    logging.info("Starting model training...")
    history = model.fit(
        X_train_images, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=32,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Evaluate on the test set
    logging.info("Evaluating the model on the test dataset...")
    test_loss, test_accuracy = model.evaluate(X_test_images, y_test, verbose=1)
    logging.info(f"Test Loss: {test_loss:.4f}   Test Accuracy: {test_accuracy:.4f}")
    
    # Save the final model
    model.save(MODEL_SAVE_PATH)
    logging.info(f"Model saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
