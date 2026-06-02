import json
import os
import argparse
import numpy as np
from pycocotools import mask as maskUtils

def encode_mask(binary):
    rle = maskUtils.encode(np.asfortranarray(binary.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def seg_to_rle(seg, h, w):
    # 已经是 COCO RLE
    if isinstance(seg, dict) and "counts" in seg and "size" in seg:
        return seg

    # polygon list
    if isinstance(seg, list):
        rles = maskUtils.frPyObjects(seg, h, w)
        rle = maskUtils.merge(rles)
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    # OCHuman segms: dict，但是不是 counts 格式
    if isinstance(seg, dict):
        # 尝试 value 里找 polygon / rle
        for v in seg.values():
            if isinstance(v, dict) and "counts" in v and "size" in v:
                return v
            if isinstance(v, list):
                rles = maskUtils.frPyObjects(v, h, w)
                rle = maskUtils.merge(rles)
                if isinstance(rle["counts"], bytes):
                    rle["counts"] = rle["counts"].decode("utf-8")
                return rle

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-json", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    data = json.load(open(args.in_json, "r", encoding="utf-8"))

    images = []
    annotations = []
    ann_id = 1

    total_raw = 0
    ok = 0
    bad_seg = 0
    missing_bbox = 0

    for idx, img in enumerate(data["images"], start=1):
        image_id = img.get("image_id", idx)
        h = int(img["height"])
        w = int(img["width"])

        images.append({
            "id": image_id,
            "file_name": img["file_name"],
            "width": w,
            "height": h,
        })

        for a in img.get("annotations", []):
            total_raw += 1
            bbox = a.get("bbox", None)
            seg = a.get("segmentation", a.get("segms", None))

            if bbox is None:
                missing_bbox += 1
                continue

            rle = seg_to_rle(seg, h, w)
            if rle is None or "counts" not in rle:
                bad_seg += 1
                continue

            area = float(maskUtils.area(rle))
            bbox2 = maskUtils.toBbox(rle).tolist()

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": rle,
                "bbox": [float(x) for x in bbox2],
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1
            ok += 1

    out = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"))

    print("saved:", args.out_json)
    print("images:", len(images))
    print("raw annotations:", total_raw)
    print("converted ok:", ok)
    print("bad seg:", bad_seg)
    print("missing bbox:", missing_bbox)
    print("avg persons/image:", ok / max(len(images), 1))

if __name__ == "__main__":
    main()
