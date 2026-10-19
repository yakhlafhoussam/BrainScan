"""
BrainScan AI — Prediction Module

Loads the trained Keras CNN and classifies brain MRI images into
four categories: glioma, meningioma, notumor, pituitary.

Preprocessing reproduces the exact pipeline used during training:
  1. Convert BGR → RGB (when loading from file with OpenCV).
  2. Resize with aspect-ratio-preserving padding to 224×224.
  3. Normalize pixel values to [0, 1].
"""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# Class names in alphabetical order — matches the LabelEncoder used
# during training (sklearn sorts class labels alphabetically).
CLASS_NAMES: list[str] = ["glioma", "meningioma", "notumor", "pituitary"]

# Expected input shape for the CNN.
IMAGE_SIZE: tuple[int, int] = (224, 224)

# Default model path, resolved relative to this file's directory.
_DEFAULT_MODEL_PATH: Path = (
    Path(__file__).resolve().parent / "models" / "brainscan_best.keras"
)


def resize_with_padding(
    image: np.ndarray, size: tuple[int, int] = IMAGE_SIZE
) -> np.ndarray:
    """Resize an image while preserving its aspect ratio, padding with black.

    This is an exact copy of the function used during training (notebook
    cell 16) so that inference preprocessing matches the training pipeline.

    Parameters
    ----------
    image : np.ndarray
        Input image in HWC format (any number of channels).
    size : tuple[int, int]
        Target (width, height).

    Returns
    -------
    np.ndarray
        Resized and padded image with shape (height, width, channels).
    """
    target_width, target_height = size
    height, width = image.shape[:2]

    scale = min(target_width / width, target_height / height)

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    canvas[
        y_offset : y_offset + new_height,
        x_offset : x_offset + new_width,
    ] = resized

    return canvas


def load_model(model_path: Path | str | None = None) -> tf.keras.Model:
    """Load the trained Keras model from disk.

    Parameters
    ----------
    model_path : Path | str | None
        Absolute or relative path to the ``.keras`` file.  When *None*,
        defaults to ``models/brainscan_best.keras`` relative to this
        module's directory.

    Returns
    -------
    tf.keras.Model
        The compiled Keras model ready for inference.
    """
    path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Make sure 'models/brainscan_best.keras' is present in the application directory."
        )
    return tf.keras.models.load_model(str(path))


def preprocess_image(image_rgb: np.ndarray) -> np.ndarray:
    """Apply the training-time preprocessing pipeline to an RGB image.

    Steps (must match training):
      1. Resize with padding to 224×224.
      2. Normalize to [0, 1] float32.
      3. Add batch dimension.

    Parameters
    ----------
    image_rgb : np.ndarray
        Image in RGB, HWC uint8 format.

    Returns
    -------
    np.ndarray
        Batch of shape ``(1, 224, 224, 3)`` with float32 values in [0, 1].
    """
    image = resize_with_padding(image_rgb, IMAGE_SIZE)
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)


def predict_image(
    model: tf.keras.Model,
    image_rgb: np.ndarray,
) -> tuple[str, dict[str, float]]:
    """Classify a single brain MRI image.

    Parameters
    ----------
    model : tf.keras.Model
        The loaded Keras model.
    image_rgb : np.ndarray
        Image in RGB, HWC uint8 format.

    Returns
    -------
    tuple[str, dict[str, float]]
        A tuple of (predicted_class_name, {class_name: probability}).
    """
    batch = preprocess_image(image_rgb)
    probabilities = model.predict(batch, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]

    class_probabilities = {
        name: float(prob) for name, prob in zip(CLASS_NAMES, probabilities)
    }

    return predicted_class, class_probabilities
