"""ZeroWaste-f COCO to YOLO11-seg instance segmentation format converter.

Dataset: ZeroWaste-f (CVPR 2022)
Official Splits: train (3002), val (572), test (929)
Classes:
  COCO ID 1: rigid_plastic -> YOLO ID 0
  COCO ID 2: cardboard     -> YOLO ID 1
  COCO ID 3: metal         -> YOLO ID 2
  COCO ID 4: soft_plastic  -> YOLO ID 3
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


RAW_DATASET_ROOT = Path(r"E:\PML\zerowaste-f-final\splits_final_deblurred")
PROCESSED_ROOT = Path(r"E:\PML\data\processed\zerowaste_yolo")

# Verified class mapping: COCO category_id -> YOLO class_id (0-indexed)
COCO_TO_YOLO_CLASS = {
    1: 0,  # rigid_plastic
    2: 1,  # cardboard
    3: 2,  # metal
    4: 3,  # soft_plastic
}

CLASS_NAMES = {
    0: "rigid_plastic",
    1: "cardboard",
    2: "metal",
    3: "soft_plastic",
}

CLASS_COLORS = {
    0: (255, 0, 128),   # rigid_plastic: purple/magenta (BGR)
    1: (0, 215, 255),   # cardboard: yellow/gold (BGR)
    2: (0, 140, 255),   # metal: orange (BGR)
    3: (255, 255, 0),   # soft_plastic: cyan (BGR)
}


def setup_image_junctions(splits: List[str] = ("train", "val", "test")) -> None:
    """Create directory junctions for images from raw dataset to processed dataset.
    This avoids duplicating ~7GB of image files while conforming to YOLO layout."""
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    images_base = PROCESSED_ROOT / "images"
    images_base.mkdir(parents=True, exist_ok=True)

    for split in splits:
        junction_path = images_base / split
        raw_images_dir = RAW_DATASET_ROOT / split / "images"
        if not raw_images_dir.exists():
            raw_images_dir = RAW_DATASET_ROOT / split / "data"

        if not junction_path.exists():
            cmd = f'cmd /c mklink /J "{junction_path}" "{raw_images_dir}"'
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"Warning: Junction creation returned {res.returncode}: {res.stderr}")
            else:
                print(f"Created image junction: {junction_path} -> {raw_images_dir}")


def convert_split(
    split: str,
    limit: Optional[int] = None,
    output_label_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """Convert COCO annotations to YOLO segmentation format for a single split.

    Args:
        split: 'train', 'val', or 'test'
        limit: If set, only convert the first `limit` images (for Phase 3 testing)
        output_label_dir: Directory to save .txt label files. Defaults to processed labels/<split>

    Returns:
        Statistics dictionary
    """
    json_path = RAW_DATASET_ROOT / split / "labels.json"
    if not json_path.exists():
        raise FileNotFoundError(f"labels.json not found at {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    if output_label_dir is None:
        output_label_dir = PROCESSED_ROOT / "labels" / split
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # Index annotations by image_id
    ann_by_image: Dict[int, List[dict]] = {}
    for ann in coco_data.get("annotations", []):
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    images = coco_data.get("images", [])
    if limit is not None:
        images = images[:limit]

    stats = {
        "images_converted": 0,
        "labels_created": 0,
        "skipped_annotations": 0,
        "invalid_polygons": 0,
        "missing_images": 0,
        "empty_label_files": 0,
    }

    raw_images_dir = RAW_DATASET_ROOT / split / "images"
    if not raw_images_dir.exists():
        raw_images_dir = RAW_DATASET_ROOT / split / "data"

    for img_info in images:
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        width = float(img_info["width"])
        height = float(img_info["height"])

        img_file = raw_images_dir / file_name
        if not img_file.exists():
            stats["missing_images"] += 1
            continue

        anns = ann_by_image.get(img_id, [])
        label_lines: List[str] = []

        for ann in anns:
            cat_id = ann.get("category_id")
            if cat_id not in COCO_TO_YOLO_CLASS:
                stats["skipped_annotations"] += 1
                continue

            yolo_cls = COCO_TO_YOLO_CLASS[cat_id]
            segm = ann.get("segmentation", [])

            # ZeroWaste uses polygon segmentation: list of coordinate lists
            if not isinstance(segm, list) or len(segm) == 0:
                stats["skipped_annotations"] += 1
                continue

            for poly in segm:
                # Need at least 3 points = 6 coordinates
                if len(poly) < 6 or len(poly) % 2 != 0:
                    stats["invalid_polygons"] += 1
                    continue

                # Normalize coordinates to [0, 1]
                coords_norm: List[float] = []
                for i in range(0, len(poly), 2):
                    x = poly[i]
                    y = poly[i + 1]
                    xn = max(0.0, min(1.0, x / width))
                    yn = max(0.0, min(1.0, y / height))
                    coords_norm.extend([xn, yn])

                coords_str = " ".join(f"{c:.6f}" for c in coords_norm)
                label_lines.append(f"{yolo_cls} {coords_str}")

        # Write YOLO label file
        label_filename = Path(file_name).stem + ".txt"
        label_path = output_label_dir / label_filename
        with open(label_path, "w", encoding="utf-8") as lf:
            for line in label_lines:
                lf.write(line + "\n")

        stats["images_converted"] += 1
        stats["labels_created"] += 1
        if len(label_lines) == 0:
            stats["empty_label_files"] += 1

    return stats


def visualize_conversion(
    image_path: Path,
    label_path: Path,
    save_path: Path,
    coco_ann: Optional[List[dict]] = None,
) -> None:
    """Render converted YOLO polygon masks over the original image for visual audit."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    overlay = img.copy()

    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]

            # Denormalize to pixel coordinates
            pts = []
            for i in range(0, len(coords), 2):
                px = int(round(coords[i] * w))
                py = int(round(coords[i + 1] * h))
                pts.append([px, py])

            pts_arr = np.array([pts], dtype=np.int32)
            color = CLASS_COLORS.get(cls_id, (0, 255, 0))

            # Draw filled semi-transparent polygon
            cv2.fillPoly(overlay, pts_arr, color)
            # Draw polygon boundary
            cv2.polylines(img, pts_arr, isClosed=True, color=color, thickness=2)

            # Class label at centroid
            if len(pts) > 0:
                cx = int(np.mean([p[0] for p in pts]))
                cy = int(np.mean([p[1] for p in pts]))
                label_text = CLASS_NAMES.get(cls_id, str(cls_id))
                cv2.putText(
                    img,
                    label_text,
                    (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

    alpha = 0.4
    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), blended)


if __name__ == "__main__":
    setup_image_junctions()
    print("Testing small conversion (10 train images)...")
    stats = convert_split("train", limit=10)
    print("Phase 3 Stats:", stats)
