
import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

IMG_SIZE = 96


def load_dataset(data_dir):
    labels_path = os.path.join(data_dir, "labels.csv")
    df = pd.read_csv(labels_path)

    images = []
    extra_features = []
    targets = []

    for _, row in df.iterrows():
        img_path = os.path.join(data_dir, "images", row["filename"])
        img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_arr = tf.keras.utils.img_to_array(img) / 255.0
        images.append(img_arr)
        extra_features.append([row["width_m"], row["height_m"]])
        targets.append(row["force_n"])

    return (np.array(images, dtype=np.float32),
            np.array(extra_features, dtype=np.float32),
            np.array(targets, dtype=np.float32))


def build_model():
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    image_input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(image_input, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    size_input = tf.keras.Input(shape=(2,))

    combined = tf.keras.layers.Concatenate()([x, size_input])
    combined = tf.keras.layers.Dense(64, activation="relu")(combined)
    combined = tf.keras.layers.Dropout(0.3)(combined)
    combined = tf.keras.layers.Dense(16, activation="relu")(combined)
    output = tf.keras.layers.Dense(1, activation="linear", name="force_n")(combined)

    model = tf.keras.Model(inputs=[image_input, size_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae", metrics=["mae"])
    return model


def train(data_dir, export_path, epochs, batch_size):
    images, extra, targets = load_dataset(data_dir)

    (img_train, img_val,
     extra_train, extra_val,
     y_train, y_val) = train_test_split(images, extra, targets, test_size=0.2, random_state=42)

    model = build_model()
    model.fit(
        [img_train, extra_train], y_train,
        validation_data=([img_val, extra_val], y_val),
        epochs=epochs,
        batch_size=batch_size,
    )

    val_mae = model.evaluate([img_val, extra_val], y_val, verbose=0)
    print(f"Validation MAE (N): {val_mae}")

    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    model.save(export_path)
    print(f"Saved model to {export_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--export_path", default="force/exported_model/force_regressor.h5")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train(args.data_dir, args.export_path, args.epochs, args.batch_size)