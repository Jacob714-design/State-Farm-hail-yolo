import json
import os

# ── CONFIG ──────────────────────────────────────────────────────────────────
# Each split maps:  coco json path  to  output labels folder
SPLITS = {
    "train": {
        "json":   "annotations/train_annotations.coco.json",
        "labels": "datasets/train/labels"
    },
    "valid": {
        "json":   "annotations/valid_annotations.coco.json",
        "labels": "datasets/valid/labels"
    },
    "test": {
        "json":   "annotations/test_annotations.coco.json",
        "labels": "datasets/test/labels"
    }
}


def coco_to_yolo(json_path, labels_dir):
    # Make the output folder if it doesn't exist yet
    os.makedirs(labels_dir, exist_ok=True)

    with open(json_path, "r") as f:
        coco = json.load(f)

    # ── Build lookup: image_id → {file_name, width, height} ─────────────────
    # We need width/height to normalize coords, and file_name to name the .txt
    image_info = {}
    for img in coco["images"]:
        image_info[img["id"]] = {
            "file_name": img["file_name"],
            "width":     img["width"],
            "height":    img["height"]
        }

    # ── Build lookup: category_id → class_index (0-based) ───────────────────
    # COCO category IDs can be arbitrary (1, 2, 5...), YOLO needs 0, 1, 2...
    category_map = {}
    for idx, cat in enumerate(coco["categories"]):
        category_map[cat["id"]] = idx

    print(f"\nClasses found: { {cat['name']: category_map[cat['id']] for cat in coco['categories']} }")

    # ── Group annotations by image_id ────────────────────────────────────────
    # One image can have many bounding boxes, so we group them first
    annotations_by_image = {}
    for ann in coco["annotations"]:
        iid = ann["image_id"]
        if iid not in annotations_by_image:
            annotations_by_image[iid] = []
        annotations_by_image[iid].append(ann)

    # ── Convert & write .txt files ───────────────────────────────────────────
    written = 0
    skipped = 0

    for img_id, img_data in image_info.items():
        W = img_data["width"]
        H = img_data["height"]

        # Derive label filename from image filename
        # e.g.  "datasets/hail_1_cropped/train/img1.jpg"  →  "img1.txt"
        base_name = os.path.splitext(os.path.basename(img_data["file_name"]))[0]
        txt_path  = os.path.join(labels_dir, base_name + ".txt")

        # If no annotations for this image, write empty .txt (YOLO expects it)
        if img_id not in annotations_by_image:
            open(txt_path, "w").close()
            skipped += 1
            continue

        lines = []
        for ann in annotations_by_image[img_id]:
            x_min, y_min, bw, bh = ann["bbox"]   # COCO: top-left x,y + w,h in pixels

            # ── THE CORE MATH ────────────────────────────────────────────────
            x_center = (x_min + bw / 2) / W
            y_center = (y_min + bh / 2) / H
            w_norm   = bw / W
            h_norm   = bh / H

            # Clamp to [0, 1] just in case of any floating point edge cases
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            w_norm   = max(0.0, min(1.0, w_norm))
            h_norm   = max(0.0, min(1.0, h_norm))

            class_idx = category_map[ann["category_id"]]
            lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

        written += 1

    print(f"  ✓ Written: {written} label files  |  Empty (no annotations): {skipped}")


# ── RUN ALL SPLITS ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    for split_name, paths in SPLITS.items():
        print(f"\nProcessing [{split_name}]...")
        coco_to_yolo(paths["json"], paths["labels"])

    print("\n✅ All splits converted. Labels ready for YOLO.")
