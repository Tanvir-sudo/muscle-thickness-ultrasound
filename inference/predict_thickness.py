"""Standalone inference: segment aponeuroses in an ultrasound image and measure muscle thickness.

Usage:
    python inference/predict_thickness.py samples/sample_01.jpg
    python inference/predict_thickness.py samples/sample_01.jpg --mm-per-px 0.058

Adapted from training/AI_Muscle_Segmentation_Model.ipynb (training and evaluation code removed).
"""

import argparse
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

IMG_SIZE = 640
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "apo_model.onnx")


def measure_thickness(seg_mask, min_points=100):
    """Median vertical distance between superficial (1) and deep (2) aponeurosis, per column."""
    sup = seg_mask == 1
    deep = seg_mask == 2
    if sup.sum() < min_points or deep.sum() < min_points:
        return None

    thickness = []
    for x in range(seg_mask.shape[1]):
        sup_y = np.where(sup[:, x])[0]
        deep_y = np.where(deep[:, x])[0]
        if len(sup_y) > 0 and len(deep_y) > 0:
            thickness.append(abs(np.median(deep_y) - np.median(sup_y)))

    return float(np.median(thickness)) if thickness else None


def run_inference(image_path, mm_per_px=None, out_json="result.json", out_mask="seg_result.png"):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape

    x = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    x = x[None, None, :, :]

    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    logits = session.run(None, {"input": x})[0]
    pred = logits.argmax(1)[0].astype(np.uint8)
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    thickness_px = measure_thickness(pred)
    cv2.imwrite(out_mask, (pred * 80).astype(np.uint8))

    result = {
        "image": os.path.basename(image_path),
        "thickness_px": None if thickness_px is None else round(thickness_px, 2),
        "thickness_mm": None if (thickness_px is None or mm_per_px is None) else round(thickness_px * mm_per_px, 2),
        "status": "ok" if thickness_px is not None else "failed",
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="Path to an ultrasound image")
    parser.add_argument("--mm-per-px", type=float, default=None,
                        help="Pixel spacing in mm (depends on the probe; not calibrated by default)")
    args = parser.parse_args()
    print(json.dumps(run_inference(args.image, args.mm_per_px), indent=2))
