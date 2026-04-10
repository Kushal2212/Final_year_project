"""
train_model.py
──────────────
Train EfficientNetB0 on cardamom disease images.
Fixes overconfidence using label smoothing and stronger regularisation.

Run:
    python main.py train
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR  = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR    = os.path.join(BASE_DIR, "dataset", "validation")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Settings ───────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS_P1  = 15
EPOCHS_P2  = 20
SEED       = 42

# ── Label smoothing ────────────────────────────────────────────────────────
# This is the KEY fix for overconfidence.
# Instead of training with hard labels [0, 0, 1, 0]
# it uses soft labels [0.05, 0.05, 0.85, 0.05]
# This prevents model from being 100% confident on anything
LABEL_SMOOTHING = 0.1


# ─────────────────────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────
def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        seed=SEED,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        seed=SEED,
        shuffle=False,
    )
    class_names = train_ds.class_names
    num_classes = len(class_names)

    print(f"\n✅ Classes found ({num_classes}): {class_names}")
    for cls in class_names:
        path  = os.path.join(TRAIN_DIR, cls)
        count = len(os.listdir(path))
        print(f"   {cls:20s}: {count} images")

    labels_path = os.path.join(MODEL_DIR, "class_labels.json")
    with open(labels_path, "w") as f:
        json.dump({name: i for i, name in enumerate(class_names)}, f, indent=2)
    print(f"   Labels saved → {labels_path}")

    return train_ds, val_ds, class_names


# ─────────────────────────────────────────────────────────────────────────
# 2. CLASS WEIGHTS — fix imbalanced dataset
# ─────────────────────────────────────────────────────────────────────────
def get_class_weights(class_names):
    counts = []
    for cls in class_names:
        path  = os.path.join(TRAIN_DIR, cls)
        count = len([
            f for f in os.listdir(path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ])
        counts.append(count)

    total   = sum(counts)
    weights = {}
    for i, count in enumerate(counts):
        weights[i] = total / (len(counts) * count)

    print("\n⚖️  Class weights:")
    for i, cls in enumerate(class_names):
        print(f"   {cls:20s}: {weights[i]:.3f}")
    return weights


# ─────────────────────────────────────────────────────────────────────────
# 3. DATA AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────
def make_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.25),
        layers.RandomContrast(0.2),
        layers.RandomTranslation(0.1, 0.1),
    ], name="augmentation")


# ─────────────────────────────────────────────────────────────────────────
# 4. BUILD MODEL
#    Key additions vs previous version:
#    - Stronger dropout (0.5 and 0.4) to prevent overfitting
#    - BatchNormalization to stabilise training
#    - L2 regularisation on Dense layers
# ─────────────────────────────────────────────────────────────────────────
def build_model(num_classes):
    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    base.trainable = False

    reg = tf.keras.regularizers.l2(1e-4)

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = make_augmentation()(inputs, training=None)
    x = layers.Rescaling(1.0 / 255)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="cardamom_efficientnet")
    return model, base


# ─────────────────────────────────────────────────────────────────────────
# 5. TRAIN
# ─────────────────────────────────────────────────────────────────────────
def train():
    print("\n📂 Loading datasets …")
    train_ds, val_ds, class_names = load_datasets()
    num_classes   = len(class_names)
    class_weights = get_class_weights(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)

    print("\n🧠 Building model …")
    model, base = build_model(num_classes)
    model.summary(line_length=80)

    MODEL_SAVE = os.path.join(MODEL_DIR, "cardamom_disease_model.keras")

    # ── Phase 1: train head only ─────────────────────────────────────────
    print(f"\n🚀 Phase 1 — training head ({EPOCHS_P1} epochs) …")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        # label_smoothing is the KEY fix for 100% confidence problem
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_P1,
        class_weight=class_weights,
        callbacks=[
            ModelCheckpoint(MODEL_SAVE, save_best_only=True,
                            monitor="val_accuracy", verbose=1),
            EarlyStopping(monitor="val_loss", patience=5,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                              patience=3, verbose=1, min_lr=1e-7),
        ],
    )

    # ── Phase 2: fine-tune top layers ────────────────────────────────────
    print(f"\n🔥 Phase 2 — fine-tuning top 50 layers ({EPOCHS_P2} epochs) …")
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        metrics=["accuracy"],
    )
    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_P2,
        class_weight=class_weights,
        callbacks=[
            ModelCheckpoint(MODEL_SAVE, save_best_only=True,
                            monitor="val_accuracy", verbose=1),
            EarlyStopping(monitor="val_loss", patience=6,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                              patience=3, verbose=1, min_lr=1e-8),
        ],
    )

    _plot_history(hist1, hist2)

    final_loss, final_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n✅ Final validation accuracy : {final_acc * 100:.2f}%")
    print(f"   Model saved → {MODEL_SAVE}\n")


def _plot_history(h1, h2):
    acc   = h1.history["accuracy"]     + h2.history["accuracy"]
    vacc  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
    loss  = h1.history["loss"]         + h2.history["loss"]
    vloss = h1.history["val_loss"]     + h2.history["val_loss"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(acc,  label="Train"); ax1.plot(vacc, label="Val")
    ax1.axvline(len(h1.history["accuracy"]) - 1,
                color="gray", ls="--", label="Fine-tune start")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(loss,  label="Train"); ax2.plot(vloss, label="Val")
    ax2.axvline(len(h1.history["loss"]) - 1, color="gray", ls="--")
    ax2.set_title("Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(MODEL_DIR, "training_history.png")
    plt.savefig(out)
    print(f"   Training chart saved → {out}")


if __name__ == "__main__":
    train()