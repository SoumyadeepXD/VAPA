
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from mediapipe_model_maker import object_detector


def build_dataset(images_and_labels_dir):
    return object_detector.Dataset.from_coco_folder(
        images_and_labels_dir, cache_dir="/tmp/mp_dataset_cache"
    )


def train(train_dir, val_dir, export_dir, epochs, batch_size, learning_rate):
    train_data = build_dataset(train_dir)
    validation_data = build_dataset(val_dir)

    spec = object_detector.SupportedModels.MOBILENET_V2

    hparams = object_detector.HParams(
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        export_dir=export_dir,
    )
    options = object_detector.ObjectDetectorOptions(
        supported_model=spec,
        hparams=hparams,
    )

    model = object_detector.ObjectDetector.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options,
    )

    metrics = model.evaluate(validation_data, batch_size=batch_size)
    print("Validation metrics:", metrics)

    os.makedirs(export_dir, exist_ok=True)
    model.export_model(model_name="detector.tflite")
    print(f"Exported model to {os.path.join(export_dir, 'detector.tflite')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--export_dir", default=os.path.dirname(config.DETECTOR_MODEL_PATH))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.3)
    args = parser.parse_args()

    train(args.train_dir, args.val_dir, args.export_dir,
          args.epochs, args.batch_size, args.learning_rate)