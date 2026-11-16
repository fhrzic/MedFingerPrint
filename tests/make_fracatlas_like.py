"""Synthetic FracAtlas-like dataset (images/Fractured, images/Non_fractured, dataset.csv,
COCO/YOLO/VOC annotations) with deliberate problems:
  3 degenerate polygon, 4 tiny object, 6 duplicated object, 7 annotated size mismatch,
  8 fracture_count mismatch, 26 = copy of 5 (other folder), 53 = copy of 30,
  50 blank, 51 inverted, 52 colour, plus an annotation for a missing image.

    python tests/make_fracatlas_like.py OUT_DIR
"""
import csv, json, os, sys
import numpy as np
from PIL import Image
out = sys.argv[1]
rng = np.random.default_rng(0)
for d in ("images/Fractured", "images/Non_fractured", "Annotations/COCO JSON", "Annotations/YOLO", "Annotations/PASCAL VOC"):
    os.makedirs(os.path.join(out, d), exist_ok=True)
coco = {"info": {}, "images": [], "annotations": [], "categories": [{"id": 0, "name": "fractured"}]}
rows = []
aid = 0
for i in range(60):
    frac = i < 24
    name = f"IMG{i:07d}.jpg"
    h, w = (300, 240) if i % 3 else (240, 300)
    yy, xx = np.mgrid[:h, :w]
    img = 30 + 10 * rng.standard_normal((h, w))
    bx, bw_ = w * rng.uniform(0.35, 0.65), w * rng.uniform(0.08, 0.16)
    bone = (np.abs(xx - bx) < bw_) & (yy > h * rng.uniform(0.05, 0.3)) & (yy < h * rng.uniform(0.7, 0.95))
    img[bone] += 140
    ex, ey, er = w * rng.uniform(0.2, 0.8), h * rng.uniform(0.2, 0.8), rng.uniform(10, 40)
    img[(xx - ex) ** 2 + (yy - ey) ** 2 < er ** 2] += 70
    if i == 50:
        img[:] = 250  # blank white
    if i == 51:
        img = 255 - img  # inverted
    img = np.clip(img, 0, 255).astype(np.uint8)
    sub = "Fractured" if frac else "Non_fractured"
    if i == 52:
        Image.fromarray(np.stack([img, img // 2, img], -1)).save(os.path.join(out, "images", sub, name))
    else:
        Image.fromarray(img).save(os.path.join(out, "images", sub, name), quality=92)
    if i == 53:  # exact duplicate of image 30
        import shutil; shutil.copy(os.path.join(out, "images/Non_fractured/IMG0000030.jpg"), os.path.join(out, "images", sub, name))
    if i == 26:  # duplicate image placed in the other folder (leak)
        import shutil; shutil.copy(os.path.join(out, "images/Fractured/IMG0000005.jpg"), os.path.join(out, "images", sub, name))
    nobj = 0
    if frac:
        coco["images"].append({"id": i + 1, "width": w, "height": h, "file_name": name})
        k = 1 + (i % 3 == 0)
        yolo = []
        for j in range(k):
            cx, cy = bx + rng.uniform(-5, 5), h * rng.uniform(0.35, 0.65)
            r = rng.uniform(6, 20)
            ang = np.linspace(0, 2 * np.pi, 7)[:-1]
            poly = np.stack([cx + r * np.cos(ang) * 1.5, cy + r * np.sin(ang)], 1)
            if i == 3 and j == 0:
                poly = poly[:2]  # degenerate
            if i == 4:
                poly = np.array([[cx, cy], [cx + 3, cy], [cx + 3, cy + 3], [cx, cy + 3]])  # tiny
            seg = poly.ravel().tolist()
            x0, y0 = poly.min(0); x1, y1 = poly.max(0)
            coco["annotations"].append({"id": aid, "iscrowd": 0, "image_id": i + 1, "category_id": 0,
                                        "segmentation": [seg], "bbox": [x0, y0, x1 - x0, y1 - y0], "area": 1.0})
            aid += 1
            yolo.append(f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
            nobj += 1
            if i == 6:  # duplicated object
                coco["annotations"].append(dict(coco["annotations"][-1], id=aid)); aid += 1; nobj += 1
        open(os.path.join(out, "Annotations/YOLO", name[:-4] + ".txt"), "w").write("\n".join(yolo) + "\n")
        objs = "".join(f"<object><name>fractured</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>9</xmax><ymax>9</ymax></bndbox></object>" for _ in yolo)
        open(os.path.join(out, "Annotations/PASCAL VOC", name[:-4] + ".xml"), "w").write(
            f"<annotation><filename>{name}</filename><size><width>{w}</width><height>{h}</height></size>{objs}</annotation>")
    if i == 7:
        coco["images"][-1]["width"] += 10  # size mismatch
    count = nobj + (1 if i == 8 else 0)  # count mismatch for image 8
    rows.append({"image_id": name, "hand": int(i % 2 == 0), "leg": int(i % 2 == 1), "hip": 0, "shoulder": 0,
                 "fractured": int(frac), "fracture_count": count, "frontal": int(i % 4 != 0), "lateral": int(i % 4 == 0)})
coco["images"].append({"id": 999, "width": 100, "height": 100, "file_name": "IMG9999999.jpg"})
coco["annotations"].append({"id": aid, "image_id": 999, "category_id": 0, "segmentation": [[1, 1, 5, 1, 5, 5]], "bbox": [1, 1, 4, 4]})
json.dump(coco, open(os.path.join(out, "Annotations/COCO JSON/COCO_fracture_masks.json"), "w"))
with open(os.path.join(out, "dataset.csv"), "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print(out)
