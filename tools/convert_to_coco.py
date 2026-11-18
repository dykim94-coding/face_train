"""Convert the raw Kaggle (Open Images faces) and SoloFace archives to one layout:

    <out>/<dataset>/images/<split>/*.jpg
    <out>/<dataset>/annotations/<split>.json      COCO, one category: face (id 1)

Images are copied byte-for-byte out of the zips (no re-encoding).
Run:  python -I tools/convert_to_coco.py --kaggle-zip ... --soloface-zip ... --out data
"""
import argparse
import collections
import hashlib
import io
import json
import zipfile
from pathlib import Path

from PIL import Image

CATEGORIES = [{'id': 1, 'name': 'face', 'supercategory': 'face'}]
MIN_SIDE = 8        # px; smaller boxes are slivers left by crop augmentation
MAX_ASPECT = 3.0    # long side / short side; faces are never this elongated


class Writer:
    """Collects one split and writes images + COCO json."""

    def __init__(self, out, dataset, split, seen_hashes):
        self.image_dir = out / dataset / 'images' / split
        self.json_path = out / dataset / 'annotations' / f'{split}.json'
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.images, self.annotations = [], []
        self.seen = seen_hashes
        self.stats = collections.Counter()
        self.dropped = []

    def add(self, name, data, boxes):
        """boxes: list of [x1, y1, x2, y2] in pixels. Returns False if skipped."""
        digest = hashlib.md5(data).hexdigest()
        if digest in self.seen:
            # Identical bytes under two names breaks ClearML dataset upload.
            self.stats['duplicate_image_skipped'] += 1
            self.dropped.append((name, f'duplicate of {self.seen[digest]}'))
            return False
        self.seen[digest] = name
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
        (self.image_dir / name).write_bytes(data)
        image_id = len(self.images) + 1
        self.images.append({'id': image_id, 'file_name': name, 'width': width, 'height': height})
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            w, h = x2 - x1, y2 - y1
            if w < 1 or h < 1:
                self.stats['box_dropped_empty_after_clip'] += 1
                continue
            self.annotations.append({
                'id': len(self.annotations) + 1, 'image_id': image_id, 'category_id': 1,
                'bbox': [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                'area': round(w * h, 2), 'iscrowd': 0})
        self.stats['images'] += 1
        self.stats['images_without_face'] += not boxes
        return True

    def close(self):
        self.stats['boxes'] = len(self.annotations)
        self.json_path.write_text(json.dumps({
            'images': self.images, 'annotations': self.annotations, 'categories': CATEGORIES}))
        return dict(self.stats)


def convert_kaggle(path, out, seen):
    archive = zipfile.ZipFile(path)
    names = set(archive.namelist())
    report = {}
    for split in ('train', 'val'):
        writer = Writer(out, 'kaggle', split, seen)
        worst = 0.0
        for member in sorted(n for n in names if n.startswith(f'images/{split}/') and n.endswith('.jpg')):
            stem = Path(member).stem
            boxes = [[float(v) for v in line.split()[-4:]]
                     for line in archive.read(f'labels2/{stem}.txt').decode().splitlines() if line.strip()]
            if not writer.add(f'{stem}.jpg', archive.read(member), boxes):
                continue
            # Cross-check pixel labels against the normalised YOLO labels.
            info = writer.images[-1]
            yolo = [[float(v) for v in line.split()[1:]]
                    for line in archive.read(f'labels/{split}/{stem}.txt').decode().splitlines() if line.strip()]
            if len(yolo) != len(boxes):
                writer.stats['yolo_count_mismatch'] += 1
                continue
            for (x1, y1, x2, y2), (cx, cy, w, h) in zip(boxes, yolo):
                worst = max(worst, abs((x1 + x2) / 2 - cx * info['width']), abs((y1 + y2) / 2 - cy * info['height']),
                            abs((x2 - x1) - w * info['width']), abs((y2 - y1) - h * info['height']))
        report[split] = writer.close()
        report[split]['max_px_diff_vs_yolo_labels'] = round(worst, 2)
        report[split]['dropped'] = writer.dropped
    return report


def convert_soloface(path, out, seen):
    archive = zipfile.ZipFile(path)
    prefix = 'soloface-detection-dataset/'
    names = set(archive.namelist())
    report = {}
    for split in ('train', 'val', 'test'):
        writer = Writer(out, 'soloface', split, seen)
        for member in sorted(n for n in names if n.startswith(f'{prefix}{split}/images/') and n.endswith('.jpg')):
            name = Path(member).name
            label = json.loads(archive.read(f'{prefix}{split}/labels/{name[:-4]}.json'))
            data = archive.read(member)
            boxes = []
            if label['class'] == 1:
                with Image.open(io.BytesIO(data)) as image:
                    width, height = image.size
                x1, y1, x2, y2 = label['bbox']
                x1, x2 = max(0.0, x1) * width, min(1.0, x2) * width
                y1, y2 = max(0.0, y1) * height, min(1.0, y2) * height
                w, h = x2 - x1, y2 - y1
                # A bad box on a "face" image means the face is cut off or mislabelled:
                # drop the whole image rather than train on it as background.
                if min(w, h) < MIN_SIDE:
                    writer.stats['image_dropped_tiny_box'] += 1
                    writer.dropped.append((name, f'tiny box {w:.1f}x{h:.1f}'))
                    continue
                if max(w, h) / min(w, h) > MAX_ASPECT:
                    writer.stats['image_dropped_elongated_box'] += 1
                    writer.dropped.append((name, f'elongated box {w:.1f}x{h:.1f}'))
                    continue
                boxes = [[x1, y1, x2, y2]]
            writer.add(name, data, boxes)
        report[split] = writer.close()
        report[split]['dropped'] = writer.dropped
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle-zip', required=True)
    parser.add_argument('--soloface-zip', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    seen = {}   # md5 -> first file name, shared so duplicates across datasets are caught too
    report = {'kaggle': convert_kaggle(args.kaggle_zip, out, seen),
              'soloface': convert_soloface(args.soloface_zip, out, seen)}
    (out / 'conversion_report.json').write_text(json.dumps(report, indent=1))
    for dataset, splits in report.items():
        for split, stats in splits.items():
            print(dataset, split, {k: v for k, v in stats.items() if k != 'dropped'})


if __name__ == '__main__':
    main()
