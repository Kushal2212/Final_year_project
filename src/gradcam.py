import os
import numpy as np
import tensorflow as tf
import cv2 

IMG_SIZE = 224


def generate_gradcam(img_path: str, model, class_idx: int, output_path: str) -> str:
    """
    Generate Grad-CAM heatmap overlaid on original image.

    Args:
        img_path    : path to original leaf image
        model       : loaded Keras model
        class_idx   : predicted class index (0, 1, 2)
        output_path : where to save the heatmap image

    Returns:
        output_path if successful, None if failed
    """
    try:
        # Load image — NO division by 255, EfficientNet handles it
        img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        arr = tf.keras.utils.img_to_array(img)      # [0, 255]
        arr_batch = np.expand_dims(arr, axis=0).astype("float32")

        # Find last conv layer inside EfficientNetB0 sub-model
        last_conv_name = "top_conv"
        try:
            effnet     = model.get_layer("efficientnetb0")
            conv_layer = effnet.get_layer(last_conv_name)
            grad_model = tf.keras.Model(
                inputs  = model.inputs,
                outputs = [conv_layer.output, model.output]
            )

            with tf.GradientTape() as tape:
                inputs = tf.cast(arr_batch, tf.float32)
                conv_outputs, predictions = grad_model(inputs)
                loss  = predictions[:, class_idx]
            grads = tape.gradient(loss, conv_outputs)

            pooled_grads  = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs  = conv_outputs[0]
            heatmap       = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap       = tf.squeeze(heatmap)
            heatmap       = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
            heatmap       = heatmap.numpy()

        except Exception:
            # Fallback: simple input gradient saliency
            inputs = tf.Variable(arr_batch)
            with tf.GradientTape() as tape:
                preds = model(inputs)
                loss  = preds[:, class_idx]
            grads   = tape.gradient(loss, inputs)
            heatmap = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()

        # Resize and colorize heatmap
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_norm    = np.uint8(255 * (heatmap_resized - heatmap_resized.min()) /
                                   (heatmap_resized.max() - heatmap_resized.min() + 1e-8))
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        # Overlay on original
        original = cv2.imread(img_path)
        original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))
        overlay  = cv2.addWeighted(original, 0.55, heatmap_colored, 0.45, 0)

        # Add legend bar at bottom
        bar_h  = 18
        canvas = np.zeros((IMG_SIZE + bar_h + 22, IMG_SIZE, 3), dtype=np.uint8)
        canvas[:IMG_SIZE] = overlay
        for x in range(IMG_SIZE):
            val   = int(x / IMG_SIZE * 255)
            color = cv2.applyColorMap(
                np.array([[val]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]
            canvas[IMG_SIZE + 4: IMG_SIZE + 4 + bar_h, x] = color

        cv2.putText(canvas, "Low",  (2, IMG_SIZE + bar_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
        cv2.putText(canvas, "High", (IMG_SIZE - 28, IMG_SIZE + bar_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        cv2.imwrite(output_path, canvas)
        return output_path

    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

