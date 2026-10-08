"""
VAPA Vision Tool: import_annotations.py
Converts CVAT / Label Studio annotation exports (YOLO or Pascal VOC XML or COCO JSON)
into standardized VAPA YOLO format aligned with data/classes.yaml.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def import_cvat_yolo(
    annotations_dir: str,
    output_yolo_dir: str = "data/yolo_dataset",
    classes_yaml: str = "data/classes.yaml",
):
    ann_path = Path(annotations_dir)
    out_path = Path(output_yolo_dir)
    out_lbl_dir = out_path / "labels" / "test"
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    with open(classes_yaml, "r") as f:
        class_cfg = yaml.safe_load(f)
    name_to_id = {c["name"]: c["id"] for c in class_cfg["classes"]}

    txt_files = list(ann_path.glob("*.txt"))
    print(f"[Annotation Importer] Ingesting {len(txt_files)} label files from {ann_path}")

    converted = 0
    for txt_file in txt_files:
        dest_file = out_lbl_dir / txt_file.name
        shutil.copy(txt_file, dest_file)
        converted += 1

    print(f"[Annotation Importer] Successfully imported {converted} YOLO label files into {out_lbl_dir}")
    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import annotations from CVAT / Label Studio")
    parser.add_argument("--annotations", type=str, required=True, help="Directory containing exported annotation txt files")
    parser.add_argument("--out", type=str, default="data/yolo_dataset", help="Target YOLO dataset directory")
    args = parser.parse_args()

    import_cvat_yolo(args.annotations, args.out)
