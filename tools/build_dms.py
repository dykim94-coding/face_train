"""Build a diverse subset of AI-Hub DMS (BBox) in the common layout (see convert_to_coco.py):

    <out>/ai_hub_dms_sampling/images/<split>/*.jpg            copied byte-for-byte
    <out>/ai_hub_dms_sampling/annotations/<split>.json        COCO, one category: face (id 1)
    <out>/ai_hub_dms_sampling/manifest_<split>.csv            every picked frame with its states
    <out>/ai_hub_dms_sampling/build_report.json

A person folder holds ~20 clips of 20 near-identical consecutive frames. Per clip we keep one frame for every
distinct state (phone / cigar / eyes / mouth) seen in it, at least 2 and at most 4, and 4 when a phone or cigar
appears. With a single state the 2 frames are the ones whose face boxes differ most. Read-only on the source.

Run:  python tools/build_dms.py --src Y:/ --out data
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

CATEGORIES = [{'id': 1, 'name': 'face', 'supercategory': 'face'}]
BAD_FOLDERS = {'R_094_50_M', 'R_106_40_M'}     # labels belong to other images (boxes off the face)
SUSPECT_FOLDERS = {'R_031_50_M', 'R_032_50_M'}  # here only labels whose inner FileName differs are dropped
MIN_FACE, MAX_ASPECT = 120, 3.0                 # px / long:short; smaller "faces" are eye boxes filed as Face
BASE, CAP = 2, 4


def parse(job):
    split, folder, name, path = job
    try:
        label = json.loads(Path(path).read_bytes().decode('utf-8-sig'))
        boxes = label['ObjectInfo']['BoundingBox']
    except Exception as exc:
        return dict(split=split, folder=folder, name=name, drop=f'unreadable label ({type(exc).__name__})')
    row = dict(split=split, folder=folder, name=name, drop=None)
    get = lambda part: boxes.get(part) or {}
    vis = lambda part: bool(get(part).get('isVisible'))
    eyes = [get(p).get('Opened') for p in ('Leye', 'Reye') if vis(p)]
    row['phone'], row['cigar'] = vis('Phone'), vis('Cigar')
    row['eyes'] = 'hidden' if not eyes else 'open' if all(eyes) else 'closed' if not any(eyes) else 'one_closed'
    row['mouth'] = 'hidden' if not vis('Mouth') else 'open' if get('Mouth').get('Opened') else 'closed'
    acc = label.get('Accessory') or {}
    row['mask'], row['glasses'], row['cap'] = (bool(acc.get(k)) for k in ('Mask', 'Glasses', 'Cap'))
    inner = os.path.splitext((label.get('FileInfo') or {}).get('FileName') or '')[0]
    if folder in BAD_FOLDERS:
        row['drop'] = 'folder with misplaced labels'
    elif folder in SUSPECT_FOLDERS and inner != name:
        row['drop'] = 'inner FileName differs (suspect folder)'
    elif not vis('Face'):
        row['drop'] = 'no face box'
    else:
        x1, y1, x2, y2 = (float(v) for v in get('Face')['Position'])
        w, h = x2 - x1, y2 - y1
        if min(w, h) < MIN_FACE:
            row['drop'] = 'tiny face box'
        elif max(w, h) / min(w, h) > MAX_ASPECT:
            row['drop'] = 'elongated face box'
        row['box'] = [x1, y1, x2, y2]
    return row


def spread(frames, k):
    """k frames whose face boxes differ most (farthest-point picking on box centre + size)."""
    key = lambda r: ((r['box'][0] + r['box'][2]) / 2, (r['box'][1] + r['box'][3]) / 2, r['box'][2] - r['box'][0])
    mean = [sum(v) / len(frames) for v in zip(*map(key, frames))]
    picked = [max(frames, key=lambda r: math.dist(key(r), mean))]
    while len(picked) < min(k, len(frames)):
        picked.append(max((r for r in frames if r not in picked),
                          key=lambda r: min(math.dist(key(r), key(p)) for p in picked)))
    return picked


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='Y:/')
    p.add_argument('--out', required=True)
    p.add_argument('--splits', nargs='+', default=['Training=train', 'Validation=val'])
    args = p.parse_args()
    src, out = Path(args.src), Path(args.out) / 'ai_hub_dms_sampling'
    report, seen = {}, {}
    for item in args.splits:
        source, split = item.split('=')
        label_dir = src / source / 'BBox' / 'label'
        jobs = [(split, d.name, f[:-5], os.path.join(d.path, f)) for d in os.scandir(label_dir) if d.is_dir()
                for f in os.listdir(d.path) if f.endswith('.json')]
        with ThreadPoolExecutor(48) as pool:
            rows = list(pool.map(parse, jobs, chunksize=256))
        dropped = collections.Counter(r['drop'] for r in rows if r['drop'])
        good = [r for r in rows if not r['drop']]
        print(split, 'labels', len(rows), 'usable', len(good), dict(dropped), flush=True)

        sig = lambda r: (r['phone'], r['cigar'], r['eyes'], r['mouth'])
        freq = collections.Counter(sig(r) for r in good)
        clips = collections.defaultdict(list)
        for r in good:      # name = <person>_<clip>_<M_G_C>_<frame>; everything but the frame identifies the clip
            clips[(r['folder'], r['name'].replace(' ', '_').rsplit('_', 1)[0])].append(r)
        picked = []
        for frames in clips.values():
            frames.sort(key=lambda r: r['name'])
            groups = collections.defaultdict(list)
            for r in frames:
                groups[sig(r)].append(r)
            want = CAP if any(r['phone'] or r['cigar'] for r in frames) else BASE
            chosen = [g[len(g) // 2] for _, g in sorted(groups.items(), key=lambda kv: freq[kv[0]])[:CAP]]  # rarest first
            if len(chosen) < want:
                rest = [r for r in frames if r not in chosen]
                chosen += [r for r in spread(rest + chosen[:1], want - len(chosen) + 1) if r not in chosen][:want - len(chosen)]
            picked += chosen
        picked.sort(key=lambda r: r['name'])

        image_dir = out / 'images' / split
        image_dir.mkdir(parents=True, exist_ok=True)
        (out / 'annotations').mkdir(parents=True, exist_ok=True)

        def fetch(r):
            path = src / source / 'BBox' / 'image' / r['folder'] / (r['name'] + '.jpg')
            return r, path.read_bytes() if path.exists() else None

        images, annotations, kept, skipped, unreadable = [], [], [], collections.Counter(), []
        with ThreadPoolExecutor(16) as pool:
            for r, data in pool.map(fetch, picked):
                if data is None:
                    skipped['image file missing'] += 1
                    continue
                digest = hashlib.md5(data).hexdigest()
                if digest in seen:
                    skipped['duplicate image bytes'] += 1
                    continue
                try:
                    with Image.open(io.BytesIO(data)) as image:
                        width, height = image.size
                        image.load()
                except Exception:
                    skipped['unreadable image'] += 1
                    unreadable.append(r['name'])
                    continue
                seen[digest] = r['name']
                x1, y1 = max(0.0, r['box'][0]), max(0.0, r['box'][1])
                x2, y2 = min(float(width), r['box'][2]), min(float(height), r['box'][3])
                if x2 - x1 < MIN_FACE or y2 - y1 < MIN_FACE:
                    skipped['face box outside image'] += 1
                    continue
                file_name = r['name'].replace(' ', '_') + '.jpg'
                (image_dir / file_name).write_bytes(data)
                images.append({'id': len(images) + 1, 'file_name': file_name, 'width': width, 'height': height})
                annotations.append({'id': len(annotations) + 1, 'image_id': len(images), 'category_id': 1,
                                    'bbox': [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                                    'area': round((x2 - x1) * (y2 - y1), 2), 'iscrowd': 0})
                kept.append(r)
        (out / 'annotations' / f'{split}.json').write_text(json.dumps(
            {'images': images, 'annotations': annotations, 'categories': CATEGORIES}))
        fields = ['name', 'folder', 'phone', 'cigar', 'eyes', 'mouth', 'mask', 'glasses', 'cap']
        with open(out / f'manifest_{split}.csv', 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(fields)
            w.writerows([r[k] for k in fields] for r in kept)

        def share(sel):
            n = max(1, len(sel))
            rate = lambda test: round(100 * sum(map(test, sel)) / n, 2)
            return {'images': len(sel), 'phone': rate(lambda r: r['phone']), 'cigar': rate(lambda r: r['cigar']),
                    'eyes_closed': rate(lambda r: r['eyes'] == 'closed'), 'one_eye_closed': rate(lambda r: r['eyes'] == 'one_closed'),
                    'eyes_hidden': rate(lambda r: r['eyes'] == 'hidden'), 'mouth_open': rate(lambda r: r['mouth'] == 'open'),
                    'mouth_hidden': rate(lambda r: r['mouth'] == 'hidden'), 'mask': rate(lambda r: r['mask']),
                    'glasses': rate(lambda r: r['glasses']), 'cap': rate(lambda r: r['cap']),
                    'female': rate(lambda r: r['folder'].endswith('_F'))}
        per_person = collections.Counter(r['folder'] for r in kept)
        report[split] = {'labels': len(rows), 'dropped_labels': dict(dropped), 'people': len(per_person), 'clips': len(clips),
                         'frames_per_clip': dict(sorted(collections.Counter(collections.Counter(
                             (r['folder'], r['name'].replace(' ', '_').rsplit('_', 1)[0]) for r in kept).values()).items())),
                         'images_per_person': {'min': min(per_person.values()), 'median': sorted(per_person.values())[len(per_person) // 2],
                                               'max': max(per_person.values())},
                         'skipped_at_copy': dict(skipped), 'unreadable_images': unreadable,
'source_all_usable_frames': share(good), 'picked': share(kept),
                         'frames_with_state_in_source_vs_picked': {
                             k: [sum(1 for r in good if t(r)), sum(1 for r in kept if t(r))] for k, t in (
                                 ('phone', lambda r: r['phone']), ('cigar', lambda r: r['cigar']),
                                 ('eyes_closed', lambda r: r['eyes'] == 'closed'), ('mouth_open', lambda r: r['mouth'] == 'open'))},
                         'clips_with_state_in_source_vs_covered': {
                             k: [sum(1 for v in clips.values() if any(map(t, v))),
                                 len({(r['folder'], r['name'].replace(' ', '_').rsplit('_', 1)[0]) for r in kept if t(r)})] for k, t in (
                                 ('phone', lambda r: r['phone']), ('cigar', lambda r: r['cigar']),
                                 ('eyes_closed', lambda r: r['eyes'] == 'closed'), ('mouth_open', lambda r: r['mouth'] == 'open'))}}
        print(json.dumps(report[split], indent=1), flush=True)
    (out / 'build_report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
