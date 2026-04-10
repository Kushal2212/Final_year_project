"""
main.py  –  Cardamom Disease Classifier  (3 classes)
Fixed: double-preprocessing bug, auto-detects dataset structure.

Usage:
    python main.py train
    python main.py evaluate
    python main.py predict <image_path>
"""

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "cardamom_disease_model.keras"
LABELS_PATH = MODEL_DIR / "class_labels.json"
MODEL_DIR.mkdir(exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE = 224
BATCH = 32
SEED = 42
CLASSES = ["chhirke", "healthy", "leaf_blight"]


# ════════════════════════════════════════════════════════════════════════════
#  AUTO-DETECT DATASET STRUCTURE
#  Handles both layouts:
#    Layout A (flat):       dataset/chhirke/img.jpg
#    Layout B (split):      dataset/train/chhirke/img.jpg
# ════════════════════════════════════════════════════════════════════════════
def find_dataset():
    """
    Search common locations and return (train_dir, val_dir) or (flat_dir, None).
    Prints what it finds so you can see exactly what path is being used.
    """
    candidates = [
        BASE_DIR / "dataset",
        BASE_DIR / "data",
        BASE_DIR / "Dataset",
        BASE_DIR / "DATA",
    ]

    for base in candidates:
        if not base.exists():
            continue

        # Layout B: base/train/chhirke/  base/validation/chhirke/
        for val_name in ["validation", "val", "valid"]:
            train_dir = base / "train"
            val_dir = base / val_name
            if train_dir.exists() and val_dir.exists():
                # check at least one class folder inside
                if any((train_dir / c).exists() for c in CLASSES):
                    print(f"\n📂 Dataset found (split layout):")
                    print(f"   Train : {train_dir}")
                    print(f"   Val   : {val_dir}")
                    return train_dir, val_dir

        # Layout A: base/chhirke/  base/healthy/  base/leaf_blight/
        if any((base / c).exists() for c in CLASSES):
            print(f"\n📂 Dataset found (flat layout):")
            print(f"   Path  : {base}")
            return base, None

    # Nothing found — print helpful message
    print("\n❌ Dataset not found!")
    print(f"   Looked in: {[str(c) for c in candidates]}")
    print(f"\n   Create this structure in your project folder:")
    print(f"   {BASE_DIR}\\dataset\\")
    for c in CLASSES:
        print(f"      {c}\\   ← put {c} leaf images here")
    sys.exit(1)


def count_class_images(directory):
    """Count images per class in a directory."""
    counts = {}
    for cls in CLASSES:
        folder = directory / cls
        if folder.exists():
            n = len([f for f in folder.iterdir()
                     if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}])
            counts[cls] = n
        else:
            counts[cls] = 0
    return counts


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
#  KEY FIX: NO Rescaling(1./255) — EfficientNetB0 preprocesses internally.
#  Feeding [0,1] instead of [0,255] was the cause of 44% accuracy.
# ════════════════════════════════════════════════════════════════════════════
def load_data():
    train_dir, val_dir = find_dataset()

    def make_ds(directory, shuffle=False, split=None, subset=None):
        kwargs = dict(
            labels="inferred",
            label_mode="categorical",
            class_names=CLASSES,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH,
            seed=SEED,
        )
        if split:
            kwargs.update(validation_split=split, subset=subset)
        if shuffle:
            kwargs["shuffle"] = True
        return tf.keras.utils.image_dataset_from_directory(directory, **kwargs)

    if val_dir:
        # Layout B — already split
        train_ds = make_ds(train_dir, shuffle=True)
        val_ds = make_ds(val_dir,   shuffle=False)
    else:
        # Layout A — flat, use 20% for validation
        train_ds = make_ds(train_dir, shuffle=True,
                           split=0.2, subset="training")
        val_ds = make_ds(train_dir, shuffle=False,
                         split=0.2, subset="validation")

    # Print counts
    counts = count_class_images(train_dir)
    print(f"\n   Class image counts (train dir):")
    total = 0
    for cls, n in counts.items():
        flag = "✅" if n >= 200 else "⚠️  LOW"
        print(f"   {flag}  {cls:20s}: {n} images")
        total += n
    print(f"   Total: {total} images")

    AUTOTUNE = tf.data.AUTOTUNE
    return train_ds.prefetch(AUTOTUNE), val_ds.prefetch(AUTOTUNE)


# ════════════════════════════════════════════════════════════════════════════
#  AUGMENTATION (operates on raw [0,255] values — correct)
# ════════════════════════════════════════════════════════════════════════════
def build_augmentation():
    aug_layers = [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.25),
        layers.RandomTranslation(0.15, 0.15),
        layers.RandomContrast(0.3),
    ]
    try:
        aug_layers.append(layers.RandomBrightness(0.3))   # TF 2.11+
    except AttributeError:
        pass
    return keras.Sequential(aug_layers, name="augmentation")


# ════════════════════════════════════════════════════════════════════════════
#  BUILD MODEL
#  No Rescaling layer — EfficientNetB0 handles preprocessing internally.
# ════════════════════════════════════════════════════════════════════════════
def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = build_augmentation()(inputs)

    base = EfficientNetB0(
        weights="imagenet",      # pretrained weights — critical for small datasets
        include_top=False,
        input_tensor=x,
        # ── no Rescaling(1./255) before this ──
        # EfficientNetB0 preprocesses [0,255] → correct range internally.
        # The old script added Rescaling BEFORE this, feeding [0,1] values,
        # which caused the model to receive near-zero inputs → 44% accuracy.
    )
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(len(CLASSES), activation="softmax")(x)

    return Model(inputs=base.input, outputs=out, name="cardamom_efficientnet"), base


# ════════════════════════════════════════════════════════════════════════════
#  CLASS WEIGHTS
# ════════════════════════════════════════════════════════════════════════════
def compute_class_weights(train_dir):
    counts = count_class_images(train_dir)
    total = sum(counts.values())
    n_cls = len(CLASSES)
    weights = {}
    print("\n⚖️  Class weights:")
    for i, cls in enumerate(CLASSES):
        n = counts[cls] or 1
        w = total / (n_cls * n)
        weights[i] = w
        print(f"   {cls:20s}: {w:.3f}")
    return weights


# ════════════════════════════════════════════════════════════════════════════
#  TRAIN
# ════════════════════════════════════════════════════════════════════════════
def train():
    # Save labels
    labels_map = {str(i): cls for i, cls in enumerate(CLASSES)}
    with open(LABELS_PATH, "w") as f:
        json.dump(labels_map, f, indent=2)
    print(f"   Labels saved → {LABELS_PATH}")

    train_ds, val_ds = load_data()
    train_dir, _ = find_dataset()
    class_weights = compute_class_weights(train_dir)

    print("\n🧠 Building model …")
    model, base = build_model()
    model.summary()

    # ── Phase 1: train head only ──────────────────────────────────────────
    print("\n🚀 Phase 1 — training head (20 epochs) …")
    model.compile(
        optimizer=keras.optimizers.Adam(5e-5),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.15),
        metrics=["accuracy"],
)
    cb1 = [
        EarlyStopping(monitor="val_accuracy", patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy",
                        save_best_only=True, verbose=1),
    ]
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=20,
                   class_weight=class_weights, callbacks=cb1)
    best1 = max(h1.history["val_accuracy"])
    print(f"\n   Phase 1 best val_accuracy: {best1*100:.2f}%")

    # ── Phase 2: fine-tune top 40 layers ─────────────────────────────────
    print("\n🔥 Phase 2 — fine-tuning top 40 layers (40 epochs) …")
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.15),
        metrics=["accuracy"],

    )
    cb2 = [
        EarlyStopping(monitor="val_accuracy", patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, min_lr=1e-8, verbose=1),
        ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy",
                        save_best_only=True, verbose=1),
    ]
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=40,
                   class_weight=class_weights, callbacks=cb2)
    best2 = max(h2.history["val_accuracy"])
    print(f"\n   Phase 2 best val_accuracy: {best2*100:.2f}%")

    # ── Save plot ─────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        all_val = h1.history["val_accuracy"] + h2.history["val_accuracy"]
        all_trn = h1.history["accuracy"] + h2.history["accuracy"]
        split = len(h1.history["accuracy"])
        plt.figure(figsize=(10, 4))
        plt.plot(all_val, label="val_accuracy")
        plt.plot(all_trn, label="train_accuracy")
        plt.axvline(split - 1, color="gray",
                    linestyle="--", label="phase 2 start")
        plt.legend()
        plt.title("Training History")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        out = MODEL_DIR / "training_history.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"   Chart saved → {out}")
    except Exception:
        pass

    final = max(best1, best2)
    print(f"\n✅ Final best val_accuracy : {final*100:.2f}%")
    print(f"   Model saved → {MODEL_PATH}")

    if final < 0.70:
        print("\n⚠️  Accuracy below 70%. Tips to improve:")
        print("   1. Aim for 400+ images per class (currently have less)")
        print("   2. Make sure images are clear, well-lit cardamom leaves")
        print("   3. Remove blurry, duplicate, or non-leaf images")
        print("   4. Try running train again — results vary slightly each run")


# ════════════════════════════════════════════════════════════════════════════
#  EVALUATE
# ════════════════════════════════════════════════════════════════════════════
def evaluate():
    if not MODEL_PATH.exists():
        print("No model found. Run: python main.py train")
        return

    _, val_ds = load_data()
    model = keras.models.load_model(str(MODEL_PATH))
    loss, acc = model.evaluate(val_ds)
    print(f"\n   Loss     : {loss:.4f}")
    print(f"   Accuracy : {acc*100:.2f}%")

    print("\n   Per-class accuracy:")
    correct = {c: 0 for c in CLASSES}
    total = {c: 0 for c in CLASSES}
    for images, labels_b in val_ds:
        preds = model.predict(images, verbose=0)
        pred_idx = np.argmax(preds, axis=1)
        true_idx = np.argmax(labels_b.numpy(), axis=1)
        for p, t in zip(pred_idx, true_idx):
            total[CLASSES[t]] += 1
            correct[CLASSES[t]] += (p == t)
    for cls in CLASSES:
        if total[cls]:
            pct = correct[cls] / total[cls] * 100
            print(f"   {cls:20s}: {pct:5.1f}%  {'█' * int(pct/5)}")


# ════════════════════════════════════════════════════════════════════════════
#  PREDICT SINGLE IMAGE
# ════════════════════════════════════════════════════════════════════════════
def predict_image(path):
    if not MODEL_PATH.exists():
        print("No model found. Run: python main.py train")
        return

    # Load raw [0,255] — no rescaling, EfficientNet handles it
    img = tf.keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = tf.keras.utils.img_to_array(img)     # [0, 255]
    arr = np.expand_dims(arr, 0)

    model = keras.models.load_model(str(MODEL_PATH))
    preds = model.predict(arr, verbose=0)[0]

    print(f"\n🔬 Prediction for: {path}")
    for i, p in enumerate(preds):
        print(f"   {CLASSES[i]:20s}: {p*100:5.1f}%  {'█' * int(p*40)}")
    top = int(np.argmax(preds))
    print(f"\n   ✅ Predicted: {CLASSES[top]}  ({preds[top]*100:.1f}%)")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=[
                        "train", "evaluate", "predict", "web"])
    parser.add_argument("image",   nargs="?", help="Image path for predict")
    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "predict":
        if not args.image:
            print("Usage: python main.py predict <image_path>")
        else:
            predict_image(args.image)

    elif args.command == "web":
        from webapp.app import create_app
        print("Starting web server...")
        print("Open http://127.0.0.1:5000")
        app = create_app()
        app.run(debug=True, host="0.0.0.0", port=5000)
