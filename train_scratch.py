"""Train YOLO11s-seg from scratch (random initialization) for 30 epochs.

This trains without pre-trained COCO weights (Model 2) to compare against
the fine-tuned baseline (Model 1).
"""

from pathlib import Path
from ultralytics import YOLO


def train_model_2_scratch():
    # Load model architecture definition without loading pretrained weights
    print("Initializing YOLO11s-seg from scratch (untrained architecture)...")
    model = YOLO("yolo11s-seg.yaml")

    # Start training for 30 epochs
    results = model.train(
        data=r"E:\PML\data.yaml",
        epochs=30,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        project=r"E:\PML\model\runs",
        name="model_2_scratch",
        save=True,
        exist_ok=True,
        pretrained=False,
    )
    print("Training finished! Results saved to E:/PML/model/runs/model_2_scratch")
    return results


if __name__ == "__main__":
    train_model_2_scratch()
