"""
utils.py – helper functions used across the project.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(BASE_DIR, "models")
LABELS_PATH = os.path.join(MODEL_DIR, "class_labels.json")


def get_class_labels() -> dict:
    """Return {class_name: index} dict from saved JSON."""
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            "class_labels.json not found. Train the model first."
        )
    with open(LABELS_PATH) as f:
        return json.load(f)


def get_index_to_class() -> dict:
    """Return {index: class_name} dict."""
    labels = get_class_labels()
    return {v: k for k, v in labels.items()}


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot and optionally save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


def evaluate_model(model_path, test_dir):
    """Evaluate saved model on test set and print metrics."""
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import classification_report, confusion_matrix

    model = load_model(model_path)
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode="categorical",
        shuffle=False,
    )

    preds = model.predict(test_data, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_data.classes
    class_names = list(test_data.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, save_path=cm_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        evaluate_model(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python src/utils.py <model.h5> <test_dir>")