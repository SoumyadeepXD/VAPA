"""
VAPA Vision Dataset Split Tool: split_dataset.py
Enforces strict session/scene-based splitting with ZERO data leakage:
1. Public datasets (COCO, Open Images, Zenodo) are partitioned into Train (80%) and Val (20%) by scene.
2. The Human Capture Dataset is isolated 100% into the Test set (never used for training or validation).
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def split_dataset(
    dataset_dir: str = "data/yolo_dataset",
    train_ratio: float = 0.80,
    seed: int = 42,
):
    np.random.seed(seed)
    base_path = Path(dataset_dir)
    img_dir = base_path / "images"
    lbl_dir = base_path / "labels"

    # Create destination subdirectories
    for split in ["train", "val", "test"]:
        (img_dir / split).mkdir(parents=True, exist_ok=True)
        (lbl_dir / split).mkdir(parents=True, exist_ok=True)

    # Gather unpartitioned images at base of images/
    unpartitioned_images = [
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ]

    print("\n" + "=" * 70)
    print(" VAPA DATASET SPLITTING (ZERO-LEAKAGE SESSION ISOLATION)")
    print("=" * 70)
    print(f" Unpartitioned Images Found: {len(unpartitioned_images)}")

    # Group by scene / session prefix (e.g., "apple_scene01_..." or "coco_...")
    scene_groups = defaultdict(list)
    for p in unpartitioned_images:
        parts = p.stem.split("_")
        scene_key = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
        scene_groups[scene_key].append(p)

    print(f" Distinct Scene/Session Groups: {len(scene_groups)}")

    train_count = 0
    val_count = 0
    test_count = 0

    scene_keys = list(scene_groups.keys())
    np.random.shuffle(scene_keys)

    for s_key in scene_keys:
        imgs = scene_groups[s_key]
        # Human capture images (tagged with 'user', 'capture', or 'own') go to TEST ONLY
        if any(tag in s_key.lower() for tag in ("user", "capture", "own", "test")):
            target_split = "test"
            test_count += len(imgs)
        else:
            # Public data goes to train/val only
            target_split = "train" if np.random.rand() < train_ratio else "val"
            if target_split == "train":
                train_count += len(imgs)
            else:
                val_count += len(imgs)

        for img_p in imgs:
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            # Move to target split
            dest_img = img_dir / target_split / img_p.name
            shutil.move(str(img_p), str(dest_img))
            if lbl_p.exists():
                dest_lbl = lbl_dir / target_split / lbl_p.name
                shutil.move(str(lbl_p), str(dest_lbl))

    print(f" Partition Completed: Train={train_count} | Val={val_count} | Test={test_count}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split VAPA dataset without leakage")
    parser.add_argument("--dir", type=str, default="data/yolo_dataset", help="YOLO dataset root")
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Train ratio (public data)")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed")
    args = parser.parse_args()

    split_dataset(dataset_dir=args.dir, train_ratio=args.train_ratio, seed=args.seed)
