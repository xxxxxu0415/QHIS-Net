import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from datetime import datetime

from m2fp import add_m2fp_config


def encode_mask(mask):
    mask = np.asfortranarray(mask.astype(np.uint8))
    rle = maskUtils.encode(mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def mask_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return inter / union


def mask_nms(preds, iou_thr=0.6):
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    keep = []

    for p in preds:
        duplicate = False
        for k in keep:
            if mask_iou(p["mask"], k["mask"]) > iou_thr:
                duplicate = True
                break
        if not duplicate:
            keep.append(p)

    return keep


def build_predictor(config, weights):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_m2fp_config(cfg)
    cfg.merge_from_file(config)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    return DefaultPredictor(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--image-root", required=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument(
    "--out",
    default=f"/home/M2FP-mix/outputs/cihp_person_coco_eval_{timestamp}"
)
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument("--mask-thresh", type=float, default=0.5)
    parser.add_argument("--nms-thresh", type=float, default=0.6)
    parser.add_argument("--min-area", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = build_predictor(args.config, args.weights)

    coco_gt = COCO(args.gt_json)
    image_root = Path(args.image_root)

    results = []
    img_ids = coco_gt.getImgIds()

    total_before_nms = 0
    total_after_nms = 0

    for img_id in tqdm(img_ids, desc="Running M2FP inference"):
        info = coco_gt.loadImgs([img_id])[0]

        img_path = image_root / info["file_name"]
        if not img_path.exists():
            img_path = image_root.parent / info["file_name"]

        img = cv2.imread(str(img_path))
        if img is None:
            print("[WARN] failed to read:", img_path)
            continue

        h, w = img.shape[:2]
        outputs = predictor(img)

        if "parsing" not in outputs or "human_outputs" not in outputs["parsing"]:
            print("[WARN] no human_outputs:", img_path)
            continue

        candidates = []

        for pred in outputs["parsing"]["human_outputs"]:
            score = float(pred.get("score", 1.0))
            if score < args.score_thresh:
                continue

            logits = pred["mask"].detach().float().cpu()

            prob = torch.sigmoid(logits).numpy()
            mask = (prob > args.mask_thresh).astype(np.uint8)

            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            area = int(mask.sum())
            if area < args.min_area:
                continue

            candidates.append({
                "score": score,
                "mask": mask.astype(bool)
            })

        total_before_nms += len(candidates)

        kept = mask_nms(candidates, iou_thr=args.nms_thresh)
        total_after_nms += len(kept)

        for p in kept:
            mask = p["mask"].astype(np.uint8)
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue

            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

            results.append({
                "image_id": img_id,
                "category_id": 1,
                "segmentation": encode_mask(mask),
                "score": float(p["score"]),
                "bbox": [
                    x1,
                    y1,
                    x2 - x1 + 1,
                    y2 - y1 + 1
                ]
            })

    pred_json = out_dir / "predictions.json"
    with open(pred_json, "w") as f:
        json.dump(results, f)

    print("\nSaved predictions to:", pred_json)
    print("Prediction count:", len(results))
    print("Candidates before NMS:", total_before_nms)
    print("Candidates after NMS:", total_after_nms)

    if len(results) == 0:
        print("No predictions.")
        return

    coco_dt = coco_gt.loadRes(str(pred_json))
    evaluator = COCOeval(coco_gt, coco_dt, "segm")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = {
        "AP_50_95": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AP_small": float(evaluator.stats[3]),
        "AP_medium": float(evaluator.stats[4]),
        "AP_large": float(evaluator.stats[5]),
        "AR_1": float(evaluator.stats[6]),
        "AR_10": float(evaluator.stats[7]),
        "AR_100": float(evaluator.stats[8]),
        "score_thresh": args.score_thresh,
        "mask_thresh": args.mask_thresh,
        "nms_thresh": args.nms_thresh,
        "min_area": args.min_area,
        "prediction_count": len(results),
        "candidates_before_nms": total_before_nms,
        "candidates_after_nms": total_after_nms
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\nSaved metrics to:", out_dir / "metrics.json")


if __name__ == "__main__":
    main()
