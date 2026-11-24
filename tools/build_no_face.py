"""Turn frames collected with collect_frames.py (no face in view: hands, objects) into a dataset of images without boxes.

    python tools/build_no_face.py --src ../collected --out data/ir_no_face

    data/ir_no_face/images/train/<kind>_<session>_<frame>.png     copies of <src>/<kind>/<session>/gray/<frame>.png
                    annotations/train.json                        COCO json with an empty annotation list
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

KINDS = ('hand_only', 'object_only', 'empty')      # folders whose frames hold no visible face


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    src, out = Path(args.src), Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    (out / 'images' / 'train').mkdir(parents=True)
    (out / 'annotations').mkdir()
    images, counts = [], {}
    for path in sorted(f for k in KINDS for f in (src / k).glob('*/gray/*.png')):
        kind, session = path.parts[-4], path.parts[-3]
        name = f'{kind}_{session}_{path.name}'
        height, width = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE).shape
        shutil.copyfile(path, out / 'images' / 'train' / name)
        images.append({'id': len(images) + 1, 'file_name': name, 'width': width, 'height': height})
        counts[f'{kind}/{session}'] = counts.get(f'{kind}/{session}', 0) + 1
    (out / 'annotations' / 'train.json').write_text(json.dumps(
        {'images': images, 'annotations': [], 'categories': [{'id': 1, 'name': 'face', 'supercategory': 'face'}]}))
    (out / 'build_report.json').write_text(json.dumps({'images': len(images), 'per_session': counts}, indent=1))
    print(len(images), 'images', counts)


if __name__ == '__main__':
    main()
