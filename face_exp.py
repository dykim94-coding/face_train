"""YOLOX-s experiment for single-class face detection.

Datasets all share one layout (see tools/convert_to_coco.py):
    <root>/images/<split>/*.jpg
    <root>/annotations/<split>.json
Any number of (root, split) pairs are merged at run time into one COCO json whose
file names are absolute paths, so the roots do not need a common parent folder.
"""
import json
import random
from pathlib import Path

import torch
from pycocotools.cocoeval import COCOeval

import yolox.layers
from yolox.exp import Exp

# YOLOX's COCOeval_opt JIT-compiles a C++ extension on first use and crashes (not an
# ImportError, so its own fallback never triggers) when ninja/a compiler is missing.
# One class and a few thousand images do not need it: always use plain pycocotools.
yolox.layers.COCOeval_opt = COCOeval


def merge_coco(sources, out_path, limit=None, seed=0):
    """sources: [(dataset_root, split), ...] -> writes one merged COCO json."""
    images, annotations, categories = [], [], None
    for root, split in sources:
        root = Path(root)
        data = json.loads((root / 'annotations' / f'{split}.json').read_text())
        categories = categories or data['categories']
        picked = data['images']
        if limit and len(picked) > limit:
            picked = random.Random(seed).sample(picked, limit)
        new_id = {}
        for image in picked:
            new_id[image['id']] = len(images) + 1
            images.append({**image, 'id': len(images) + 1,
                           'file_name': str((root / 'images' / split / image['file_name']).resolve())})
        for annotation in data['annotations']:
            if annotation['image_id'] in new_id:
                annotations.append({**annotation, 'id': len(annotations) + 1,
                                    'image_id': new_id[annotation['image_id']]})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({'images': images, 'annotations': annotations, 'categories': categories}))
    return len(images), len(annotations)


# name -> depth, width, depthwise convolutions (the official YOLOX-s / -tiny / -nano definitions)
MODEL_SIZES = {'s': (0.33, 0.50, False), 'tiny': (0.33, 0.375, False), 'nano': (0.33, 0.25, True)}


def make_model(exp, depthwise):
    """Exp.get_model() with the depthwise switch YOLOX-nano needs (its own exp file does the same)."""
    from torch import nn
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
    if getattr(exp, 'model', None) is None:
        channels = [256, 512, 1024]
        exp.model = YOLOX(YOLOPAFPN(exp.depth, exp.width, in_channels=channels, act=exp.act, depthwise=depthwise),
                          YOLOXHead(exp.num_classes, exp.width, in_channels=channels, act=exp.act, depthwise=depthwise))
    for m in exp.model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps, m.momentum = 1e-3, 0.03
    exp.model.head.initialize_biases(1e-2)
    exp.model.train()
    return exp.model


def build_model(size='s'):
    """Bare YOLOX face model (same definition training uses), for export/inference."""
    exp = Exp()
    exp.num_classes = 1
    exp.depth, exp.width, depthwise = MODEL_SIZES[size]
    return make_model(exp, depthwise)


class FaceExp(Exp):
    def __init__(self, train_json, val_json, exp_name, output_dir, epochs=50, workers=4, size='s'):
        super().__init__()
        self.num_classes = 1
        self.depth, self.width, self.depthwise = MODEL_SIZES[size]
        if size != 's':                      # the lighter augmentation of the official tiny / nano recipes
            self.mosaic_scale = (0.5, 1.5)
            self.enable_mixup = False
        if size == 'nano':
            self.mosaic_prob = 0.5
        self.exp_name = exp_name
        self.output_dir = str(output_dir)
        self.train_json = str(Path(train_json).resolve())
        self.val_json = str(Path(val_json).resolve())
        self.data_num_workers = workers
        self.max_epoch = epochs
        self.warmup_epochs = min(3, max(1, epochs // 10))
        self.no_aug_epochs = min(10, max(1, epochs // 5))
        self.eval_interval = 5
        self.print_interval = 50
        self.save_history_ckpt = False   # keep latest + best only
        self.seed = 0

    def get_model(self):
        return make_model(self, self.depthwise)

    def _dataset(self, json_file, preproc, img_size):
        from yolox.data import COCODataset
        # An absolute json path / absolute file names make data_dir and name irrelevant.
        return COCODataset(data_dir=str(Path(json_file).parent), json_file=json_file, name='',
                           img_size=img_size, preproc=preproc)

    def get_dataset(self, cache=False, cache_type='ram'):
        from yolox.data import TrainTransform
        return self._dataset(self.train_json, TrainTransform(
            max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob), self.input_size)

    def get_eval_dataset(self, **kwargs):
        from yolox.data import ValTransform
        return self._dataset(kwargs.get('json_file', self.val_json),
                             ValTransform(legacy=kwargs.get('legacy', False)), self.test_size)

    def make_evaluator(self, json_file, batch_size):
        """Evaluator for an arbitrary merged json (used for the fixed final benchmarks)."""
        from yolox.evaluators import COCOEvaluator
        loader = torch.utils.data.DataLoader(
            self.get_eval_dataset(json_file=str(Path(json_file).resolve())), batch_size=batch_size,
            num_workers=self.data_num_workers, pin_memory=True, shuffle=False)
        return COCOEvaluator(dataloader=loader, img_size=self.test_size, confthre=self.test_conf,
                             nmsthre=self.nmsthre, num_classes=self.num_classes)
