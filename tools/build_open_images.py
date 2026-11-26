"""Turn the downloaded Open Images face / hand / head subset into the training layout (3 classes).

    python tools/build_open_images.py --src ../datasets/open_images --out data/open_images_fhh

Source: <src>/images/<split>/<ImageID>.jpg and <src>/labels/fhh_<split>.csv (see datasets/open_images/*.py in the demo project).
Only images that carry a real face, hand and head box were downloaded. Open Images 'validation' becomes our 'val'.
Boxes marked IsGroupOf or IsDepiction are kept as iscrowd=1: not trained on, and ignored when scoring.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

CATEGORIES = [{'id': 1, 'name': 'face', 'supercategory': 'person'}, {'id': 2, 'name': 'hand', 'supercategory': 'person'},
              {'id': 3, 'name': 'head', 'supercategory': 'person'}]
SPLITS = {'train': 'train', 'validation': 'val', 'test': 'test'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    src, out = Path(args.src), Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    (out / 'annotations').mkdir(parents=True)
    ids = {c['name']: c['id'] for c in CATEGORIES}
    report = {}
    for source, split in SPLITS.items():
        (out / 'images' / split).mkdir(parents=True)
        df = pd.read_csv(src / 'labels' / f'fhh_{source}.csv')
        chosen = (src / 'labels' / f'selected_{source}.txt').read_text().split()
        rows = {k: g for k, g in df[df.ImageID.isin(set(chosen))].groupby('ImageID')}
        images, annotations, counts = [], [], {c: 0 for c in ids}
        for name in chosen:
            path = src / 'images' / source / f'{name}.jpg'
            with Image.open(path) as im:
                width, height = im.size             # stored pixels; the boxes refer to these (EXIF is not applied)
            target = out / 'images' / split / path.name
            try:
                os.link(path, target)               # same disk: no second copy
            except OSError:
                shutil.copyfile(path, target)
            images.append({'id': len(images) + 1, 'file_name': path.name, 'width': width, 'height': height})
            for r in rows[name].itertuples():
                x, y = r.XMin * width, r.YMin * height
                w, h = (r.XMax - r.XMin) * width, (r.YMax - r.YMin) * height
                if w < 1 or h < 1:
                    continue
                crowd = int(r.IsGroupOf == 1 or r.IsDepiction == 1)
                counts[r.cls] += not crowd
                annotations.append({'id': len(annotations) + 1, 'image_id': len(images), 'category_id': ids[r.cls],
                                    'bbox': [round(x, 2), round(y, 2), round(w, 2), round(h, 2)], 'area': round(w * h, 2), 'iscrowd': crowd})
        (out / 'annotations' / f'{split}.json').write_text(json.dumps({'images': images, 'annotations': annotations, 'categories': CATEGORIES}))
        report[split] = {'images': len(images), 'boxes': counts, 'ignored_group_or_depiction': sum(a['iscrowd'] for a in annotations)}
        print(split, report[split])
    (out / 'build_report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
