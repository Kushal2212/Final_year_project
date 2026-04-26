import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODEL_DIR  = BASE_DIR / "models"
TRAIN_DIR  = BASE_DIR / "dataset" / "train"
VAL_DIR    = BASE_DIR / "dataset" / "validation"
MODEL_DIR.mkdir(exist_ok=True)

# Separate model save paths
EFFNET_PATH  = MODEL_DIR / "model_efficientnet.keras"
MOBILENET_PATH = MODEL_DIR / "model_mobilenet.keras"
LABELS_PATH  = MODEL_DIR / "class_labels.json"

# ── Settings ───────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
SEED       = 42
CLASSES    = ["chhirke", "healthy", "leaf_blight"]


# ─────────────────────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────
def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(TRAIN_DIR),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASSES,
        seed=SEED,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(VAL_DIR),
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASSES,
        seed=SEED,
        shuffle=False,
    )
    print(f"\n✅ Classes: {train_ds.class_names}")
    for cls in CLASSES:
        n = len(list((TRAIN_DIR / cls).glob("*.*")))
        print(f"   {cls:20s}: {n} images")

    # Save labels
    with open(LABELS_PATH, "w") as f:
        json.dump({str(i): cls for i, cls in enumerate(CLASSES)}, f, indent=2)
    print(f"   Labels saved → {LABELS_PATH}")

    AUTOTUNE = tf.data.AUTOTUNE
    return (train_ds.cache().shuffle(1000).prefetch(AUTOTUNE),
            val_ds.cache().prefetch(AUTOTUNE))


# ─────────────────────────────────────────────────────────────────────────
# 2. CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────
def get_class_weights():
    counts = [len(list((TRAIN_DIR / cls).glob("*.*"))) for cls in CLASSES]
    total  = sum(counts)
    weights = {i: total / (len(CLASSES) * c) for i, c in enumerate(counts)}
    print("\n⚖️  Class weights:")
    for i, cls in enumerate(CLASSES):
        print(f"   {cls:20s}: {weights[i]:.3f}")
    return weights


# ─────────────────────────────────────────────────────────────────────────
# 3. AUGMENTATION (same for both models)
# ─────────────────────────────────────────────────────────────────────────
def make_augmentation():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.2),
    ], name="augmentation")


# ─────────────────────────────────────────────────────────────────────────
# 4. BUILD EFFICIENTNETB0
#    Input: raw [0, 255] pixels
#    EfficientNetB0 has internal preprocessing — no Rescaling needed
# ─────────────────────────────────────────────────────────────────────────
def build_efficientnet(num_classes):
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="effnet_input")
    x      = make_augmentation()(inputs)

    base = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_tensor=x,
    )
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax", name="effnet_output")(x)

    model = Model(inputs=base.input, outputs=out, name="efficientnetb0_model")
    return model, base


# ─────────────────────────────────────────────────────────────────────────
# 5. BUILD MOBILENETV2
#    Input: raw [0, 255] pixels
#    MobileNetV2 needs Rescaling to [-1, 1]
# ─────────────────────────────────────────────────────────────────────────
def build_mobilenet(num_classes):
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="mobilenet_input")
    x      = make_augmentation()(inputs)
    # MobileNetV2 expects [-1, 1] range
    x      = layers.Rescaling(1.0 / 127.5, offset=-1.0)(x)

    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=x,
    )
    base.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax", name="mobilenet_output")(x)

    model = Model(inputs=base.input, outputs=out, name="mobilenetv2_model")
    return model, base


# ─────────────────────────────────────────────────────────────────────────
# 6. TRAIN ONE MODEL (reused for both)
# ─────────────────────────────────────────────────────────────────────────
def train_single_model(model, base, model_name, save_path,
                       train_ds, val_ds, class_weights):
    print(f"\n{'='*55}")
    print(f"  Training {model_name}")
    print(f"{'='*55}")

    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    # ── Phase 1: frozen base ──────────────────────────────────────────────
    print(f"\n🚀 Phase 1 — training head (base frozen) …")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    h1 = model.fit(
        train_ds, validation_data=val_ds, epochs=20,
        class_weight=class_weights,
        callbacks=[
            ModelCheckpoint(str(save_path), save_best_only=True,
                            monitor="val_accuracy", verbose=1),
            EarlyStopping(monitor="val_loss", patience=6,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                              patience=3, verbose=1, min_lr=1e-7),
        ],
    )
    best1 = max(h1.history["val_accuracy"])
    print(f"   Phase 1 best val_accuracy: {best1*100:.2f}%")

    # ── Phase 2: fine-tune top 40 layers ─────────────────────────────────
    print(f"\n🔥 Phase 2 — fine-tuning top 40 layers …")
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    h2 = model.fit(
        train_ds, validation_data=val_ds, epochs=40,
        class_weight=class_weights,
        callbacks=[
            ModelCheckpoint(str(save_path), save_best_only=True,
                            monitor="val_accuracy", verbose=1),
            EarlyStopping(monitor="val_loss", patience=8,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                              patience=4, verbose=1, min_lr=1e-8),
        ],
    )
    best2 = max(h2.history["val_accuracy"])
    print(f"   Phase 2 best val_accuracy: {best2*100:.2f}%")

    final = max(best1, best2)
    print(f"\n✅ {model_name} final accuracy: {final*100:.2f}%")
    print(f"   Saved → {save_path}")
    return h1, h2, final


# ─────────────────────────────────────────────────────────────────────────
# 7. EVALUATE SOFT VOTING ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────
def evaluate_ensemble(val_ds):
    print(f"\n{'='*55}")
    print(f"  Evaluating Soft Voting Ensemble")
    print(f"{'='*55}")

    effnet_model   = tf.keras.models.load_model(str(EFFNET_PATH))
    mobilenet_model = tf.keras.models.load_model(str(MOBILENET_PATH))

    correct_eff  = 0
    correct_mob  = 0
    correct_ens  = 0
    total        = 0

    for images, labels in val_ds:
        true_idx  = np.argmax(labels.numpy(), axis=1)

        preds_eff = effnet_model.predict(images, verbose=0)
        preds_mob = mobilenet_model.predict(images, verbose=0)

        # Soft voting: average probabilities
        preds_ens = (preds_eff + preds_mob) / 2.0

        correct_eff += np.sum(np.argmax(preds_eff, axis=1) == true_idx)
        correct_mob += np.sum(np.argmax(preds_mob, axis=1) == true_idx)
        correct_ens += np.sum(np.argmax(preds_ens, axis=1) == true_idx)
        total       += len(true_idx)

    print(f"\n   EfficientNetB0 alone : {correct_eff/total*100:.2f}%")
    print(f"   MobileNetV2 alone    : {correct_mob/total*100:.2f}%")
    print(f"   Soft Voting Ensemble : {correct_ens/total*100:.2f}%  ← final model")


# ─────────────────────────────────────────────────────────────────────────
# 8. PLOT HISTORY
# ─────────────────────────────────────────────────────────────────────────
def plot_history(h1, h2, title, out_path):
    try:
        acc   = h1.history["accuracy"]     + h2.history["accuracy"]
        vacc  = h1.history["val_accuracy"] + h2.history["val_accuracy"]
        split = len(h1.history["accuracy"])
        plt.figure(figsize=(9, 4))
        plt.plot(vacc, label="val_accuracy")
        plt.plot(acc,  label="train_accuracy", alpha=0.6)
        plt.axvline(split - 1, color="gray", ls="--", label="fine-tune start")
        plt.title(title); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"   Chart saved → {out_path}")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────
# 9. MAIN TRAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────
def train():
    print("\n📂 Loading datasets …")
    train_ds, val_ds = load_datasets()
    class_weights    = get_class_weights()
    num_classes      = len(CLASSES)

    # ── Train EfficientNetB0 ──────────────────────────────────────────────
    eff_model, eff_base = build_efficientnet(num_classes)
    h1e, h2e, acc_eff  = train_single_model(
        eff_model, eff_base, "EfficientNetB0",
        EFFNET_PATH, train_ds, val_ds, class_weights
    )
    plot_history(h1e, h2e, "EfficientNetB0 Training",
                 str(MODEL_DIR / "history_efficientnet.png"))

    # ── Train MobileNetV2 ─────────────────────────────────────────────────
    mob_model, mob_base = build_mobilenet(num_classes)
    h1m, h2m, acc_mob  = train_single_model(
        mob_model, mob_base, "MobileNetV2",
        MOBILENET_PATH, train_ds, val_ds, class_weights
    )
    plot_history(h1m, h2m, "MobileNetV2 Training",
                 str(MODEL_DIR / "history_mobilenet.png"))

    # ── Evaluate ensemble ─────────────────────────────────────────────────
    evaluate_ensemble(val_ds)

    print(f"\n{'='*55}")
    print(f"  Training Complete!")
    print(f"{'='*55}")
    print(f"  EfficientNetB0 : {acc_eff*100:.2f}%")
    print(f"  MobileNetV2    : {acc_mob*100:.2f}%")
    print(f"  Models saved   : models/")
    print(f"    model_efficientnet.keras")
    print(f"    model_mobilenet.keras")


if __name__ == "__main__":
    train()


