import os
import json
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as maskUtils


def encode_binary_mask(mask):
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def convert_split(image_dir, inst_dir, out_img_dir, out_json, start_img_id=1, start_ann_id=1, copy_images=True):
    image_dir = Path(image_dir)
    inst_dir = Path(inst_dir)
    out_img_dir = Path(out_img_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []

    img_id = start_img_id
    ann_id = start_ann_id

    img_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])

    for img_path in img_files:
        stem = img_path.stem

        inst_path = None
        for ext in [".png", ".jpg"]:
            cand = inst_dir / f"{stem}{ext}"
            if cand.exists():
                inst_path = cand
                break

        if inst_path is None:
            print(f"[WARN] no instance mask for {img_path.name}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] cannot read image {img_path}")
            continue

        h, w = img.shape[:2]
        inst = cv2.imread(str(inst_path), cv2.IMREAD_UNCHANGED)
        if inst is None:
            print(f"[WARN] cannot read instance {inst_path}")
            continue

        if inst.ndim == 3:
            inst = inst[:, :, 0]

        file_name = img_path.name
        dst_img_path = out_img_dir / file_name
        if copy_images and not dst_img_path.exists():
            shutil.copy2(img_path, dst_img_path)

        images.append({
            "id": img_id,
            "file_name": file_name,
            "height": h,
            "width": w,
        })

        ids = np.unique(inst)
        ids = ids[ids > 0]

        for pid in ids:
            binary = (inst == pid).astype(np.uint8)
            area = int(binary.sum())
            if area < 10:
                continue

            rle = encode_binary_mask(binary)
            bbox = maskUtils.toBbox(rle).tolist()

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "area": float(area),
                "bbox": [float(x) for x in bbox],
                "iscrowd": 0,
            })
            ann_id += 1

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "person", "supercategory": "person"}
        ],
    }

    with open(out_json, "w") as f:
        json.dump(coco, f)

    print(f"saved: {out_json}")
    print(f"images: {len(images)}")
    print(f"annotations: {len(annotations)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--inst-dir", required=True)
    parser.add_argument("--out-img-dir", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    convert_split(
        args.image_dir,
        args.inst_dir,
        args.out_img_dir,
        args.out_json,
    )


if __name__ == "__main__":
    main()