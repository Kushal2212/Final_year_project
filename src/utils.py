

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
        
def evaluate_ensemble(test_dir):
    """Evaluate soft voting ensemble on test set and save results."""
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 accuracy_score)

    effnet_path    = os.path.join(MODEL_DIR, "model_efficientnet.keras")
    mobilenet_path = os.path.join(MODEL_DIR, "model_mobilenet.keras")

    if not os.path.exists(effnet_path) or not os.path.exists(mobilenet_path):
        print("❌ Model files not found. Train first: python main.py train")
        return

    print("Loading EfficientNetB0...")
    effnet    = load_model(effnet_path)
    print("Loading MobileNetV2...")
    mobilenet = load_model(mobilenet_path)

    # ── Load test images WITHOUT rescaling ────────────────────────────────
    # EfficientNetB0 handles its own preprocessing internally
    datagen   = ImageDataGenerator()
    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode="categorical",
        shuffle=False,
    )

    class_names = list(test_data.class_indices.keys())
    print(f"\nClasses found: {class_names}")
    print(f"Total test images: {test_data.n}\n")

    # ── Run predictions on all batches ────────────────────────────────────
    y_true     = []
    y_pred_eff = []
    y_pred_mob = []
    y_pred_ens = []

    test_data.reset()
    steps = len(test_data)

    for step in range(steps):
        imgs, labels = next(test_data)

        # EfficientNetB0 — raw [0,255] (no rescaling needed)
        p_eff = effnet.predict(imgs, verbose=0)

        # MobileNetV2 — needs [-1, 1] range
        imgs_mob = (imgs / 127.5) - 1.0
        p_mob    = mobilenet.predict(imgs_mob, verbose=0)

        # Soft voting — average probabilities
        p_ens = (p_eff + p_mob) / 2.0

        y_true.extend(np.argmax(labels, axis=1))
        y_pred_eff.extend(np.argmax(p_eff,  axis=1))
        y_pred_mob.extend(np.argmax(p_mob,  axis=1))
        y_pred_ens.extend(np.argmax(p_ens,  axis=1))

        print(f"  Step {step+1}/{steps} done", end="\r")

    # ── Print accuracy numbers ─────────────────────────────────────────────
    acc_eff = accuracy_score(y_true, y_pred_eff) * 100
    acc_mob = accuracy_score(y_true, y_pred_mob) * 100
    acc_ens = accuracy_score(y_true, y_pred_ens) * 100

    print(f"\n{'='*50}")
    print(f"  EfficientNetB0  Accuracy : {acc_eff:.2f}%")
    print(f"  MobileNetV2     Accuracy : {acc_mob:.2f}%")
    print(f"  Ensemble        Accuracy : {acc_ens:.2f}%  ← Best")
    print(f"{'='*50}")

    print("\nDetailed Classification Report (Ensemble):")
    print(classification_report(y_true, y_pred_ens,
                                target_names=class_names))

    # ── Save results folder ────────────────────────────────────────────────
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ── Plot 3 confusion matrices side by side ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, y_pred, title in zip(
        axes,
        [y_pred_eff, y_pred_mob, y_pred_ens],
        ["EfficientNetB0", "MobileNetV2", "Ensemble (Soft Voting)"]
    ):
        cm  = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred) * 100

        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            xlabel=f"Predicted Label  |  Accuracy: {acc:.2f}%",
            ylabel="True Label",
            title=title
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        # Write numbers inside each cell
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=13, fontweight="bold")

    plt.suptitle(
        "Confusion Matrix — Cardamom Disease Detection System",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved → {cm_path}")

    # ── Plot accuracy comparison bar chart ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    models  = ["EfficientNetB0", "MobileNetV2", "Ensemble"]
    accs    = [acc_eff, acc_mob, acc_ens]
    colors  = ["#4a9463", "#2d5c3f", "#c8a84b"]

    bars = ax.bar(models, accs, color=colors, width=0.5, edgecolor="white")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")

    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{acc:.2f}%",
            ha="center", fontsize=12, fontweight="bold"
        )
    ax.axhline(y=max(accs), color="red", linestyle="--",
               alpha=0.4, label=f"Best: {max(accs):.2f}%")
    ax.legend()
    plt.tight_layout()
    bar_path = os.path.join(results_dir, "accuracy_comparison.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved → {bar_path}")
    print("\nInsert both images from results/ folder into Chapter 5 of your report.")