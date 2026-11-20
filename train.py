"""Train YOLOX-s face detection on kaggle / soloface / both / all (= both + AI-Hub DMS subset).

Local:    python train.py --dataset kaggle --data-root data --no-clearml
Cluster:  python train.py --dataset both --queue slurm          (data comes from ClearML Datasets)
Continue: python train.py --dataset all --pretrained clearml:<task id> --epochs 20 --lr-scale 0.2 --queue slurm

Whatever it trains on, the best checkpoint is finally scored on the same fixed benchmarks
(Kaggle val, SoloFace test, DMS val) so all models are directly comparable.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'third_party' / 'YOLOX'))

DATASETS = {                      # --dataset choice -> names used on disk / in ClearML
    'kaggle': ['kaggle'],
    'soloface': ['soloface'],
    'both': ['kaggle', 'soloface'],
    'all': ['kaggle', 'soloface', 'ai_hub_dms_sampling'],
}
PROJECT = 'FD-FS (Face - object detection, feasibility study)'   # same in tools/upload_dataset.py
CLEARML_DATASET_NAME = {'kaggle': 'kaggle_face', 'soloface': 'soloface', 'ai_hub_dms_sampling': 'ai_hub_dms_sampling'}
TRAIN_VAL_SPLIT = {'kaggle': 'val', 'soloface': 'val', 'ai_hub_dms_sampling': 'val'}   # used during training
BENCHMARKS = {'kaggle_val': ('kaggle', 'val'), 'soloface_test': ('soloface', 'test'),
              'dms_val': ('ai_hub_dms_sampling', 'val')}
COCO_YOLOX_S = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth'
DOCKER_IMAGE = 'pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel'
PACKAGES = ['clearml', 'boto3', 'numpy<2', 'opencv-python-headless', 'loguru', 'tabulate',
            'pycocotools', 'thop', 'tensorboard', 'psutil', 'tqdm']   # torch comes with the image


def parse_args():
    p = argparse.ArgumentParser()
    # Not argparse-"required": on a ClearML agent the script starts with an empty command
    # line and the values arrive through task.connect() in main().
    p.add_argument('--dataset', choices=list(DATASETS), default='')
    p.add_argument('--data-root', default='', help='local folder holding kaggle/ and soloface/; '
                   'empty = fetch ClearML Datasets')
    p.add_argument('--epochs', type=int, default=50, help='0 = no training, only benchmark --pretrained')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--devices', type=int, default=1, help='number of GPUs')
    p.add_argument('--pretrained', default='coco', help="'coco', 'none', a .pth path, or 'clearml:<task id>' "
                   "(continue from that task's best_ckpt artifact)")
    p.add_argument('--lr-scale', type=float, default=1.0, help='multiplies the learning rate (e.g. 0.2 to fine-tune)')
    p.add_argument('--no-fp16', action='store_true')
    p.add_argument('--print-interval', type=int, default=50, help='log every N iterations')
    p.add_argument('--name', default='', help='experiment name (default: yolox_s_face_<dataset>)')
    # Empty = <repo>/YOLOX_outputs, resolved where the training actually runs (an absolute
    # default would be recorded on the submitting machine and reused on the GPU node).
    p.add_argument('--output-dir', default='')
    p.add_argument('--limit-train', type=int, default=0, help='debug: use only N training images per dataset')
    p.add_argument('--limit-eval', type=int, default=0, help='debug: use only N images per eval set')
    p.add_argument('--no-clearml', action='store_true')
    p.add_argument('--project', default=PROJECT)
    p.add_argument('--queue', default='', help='ClearML queue; empty = run here (still tracked)')
    return p.parse_args()


def resolve_roots(args, names):
    """name -> dataset root directory."""
    if args.data_root:
        return {n: Path(args.data_root).resolve() / n for n in names}
    if args.no_clearml:
        raise SystemExit('--no-clearml needs --data-root')
    from clearml import Dataset
    return {n: Path(Dataset.get(dataset_project=args.project,
                                dataset_name=CLEARML_DATASET_NAME[n]).get_local_copy()) for n in names}


def pretrained_path(args):
    if args.pretrained == 'none':
        return None
    if args.pretrained.startswith('clearml:'):
        from clearml import Task
        return Task.get_task(task_id=args.pretrained.split(':', 1)[1]).artifacts['best_ckpt'].get_local_copy()
    if args.pretrained != 'coco':
        return args.pretrained
    target = ROOT / 'weights' / 'yolox_s.pth'
    if not target.exists():
        import torch
        target.parent.mkdir(exist_ok=True)
        try:
            torch.hub.download_url_to_file(COCO_YOLOX_S, str(target))
        except Exception as exc:
            raise SystemExit(f'Could not download COCO weights ({exc}). '
                             'Pass --pretrained <file.pth> or --pretrained none.')
    return str(target)


def run_training(exp, trainer_args):
    import torch.backends.cudnn as cudnn
    from yolox.exp import check_exp_value
    from yolox.utils import configure_module
    configure_module()
    cudnn.benchmark = True
    check_exp_value(exp)
    exp.get_trainer(trainer_args).train()


def main():
    args = parse_args()

    task = None
    if not args.no_clearml:
        from clearml import Task
        task = Task.init(project_name=args.project,
                         task_name=args.name or f'yolox_s_face_{args.dataset}',
                         auto_connect_arg_parser=False)
        # Explicit two-way binding: stored on submit, and on the agent the stored values
        # replace the (empty-command-line) defaults.
        args = argparse.Namespace(**task.connect(vars(args), name='Args'))
        task.set_base_docker(docker_image=DOCKER_IMAGE, docker_arguments='--shm-size=8g')
        task.set_packages(PACKAGES)
        if args.queue:
            task.execute_remotely(queue_name=args.queue)   # local process ends here
    if args.dataset not in DATASETS:
        raise SystemExit(f'--dataset must be one of {list(DATASETS)} (got {args.dataset!r})')
    name = args.name or f'yolox_s_face_{args.dataset}'
    args.output_dir = args.output_dir or str(ROOT / 'YOLOX_outputs')
    # ---- everything below runs on the machine that actually trains ----
    import torch
    from yolox.core import launch
    from face_exp import FaceExp, merge_coco

    train_names = DATASETS[args.dataset]
    roots = resolve_roots(args, sorted(set(train_names) | {d for d, _ in BENCHMARKS.values()}))
    work = Path(args.output_dir) / name
    lists = work / 'lists'
    n_train = merge_coco([(roots[n], 'train') for n in train_names], lists / 'train.json',
                         limit=args.limit_train)
    n_val = merge_coco([(roots[n], TRAIN_VAL_SPLIT[n]) for n in train_names], lists / 'val.json',
                       limit=args.limit_eval)
    print(f'train: {n_train[0]} images / {n_train[1]} boxes | val: {n_val[0]} images / {n_val[1]} boxes')

    exp = FaceExp(lists / 'train.json', lists / 'val.json', exp_name=name, output_dir=args.output_dir,
                  epochs=args.epochs, workers=args.workers)
    exp.print_interval = args.print_interval
    exp.basic_lr_per_img *= args.lr_scale      # YOLOX then multiplies by the batch size
    start = pretrained_path(args)
    if start:
        # Our own checkpoints also hold numpy scalars (best_ap), which newer torch refuses to
        # unpickle inside YOLOX's loader. Hand it a weights-only copy.
        weights = torch.load(start, map_location='cpu', weights_only=False)['model']
        start = str(work / 'start_weights.pth')
        torch.save({'model': weights}, start)
    if args.epochs > 0:
        trainer_args = argparse.Namespace(
            experiment_name=name, batch_size=args.batch_size, fp16=not args.no_fp16, ckpt=start,
            resume=False, start_epoch=None, cache=None, occupy=False, logger='tensorboard', exp_file=None)
        launch(run_training, args.devices, 1, 0, backend='nccl', dist_url='auto', args=(exp, trainer_args))

    # ---- fixed benchmarks on the best checkpoint (or on --pretrained itself when --epochs 0) ----
    best = work / 'best_ckpt.pth' if args.epochs > 0 else Path(start)
    model = exp.get_model()
    model.load_state_dict(torch.load(best, map_location='cpu', weights_only=False)['model'])
    model.cuda().eval()
    results = {}
    for bench, (dataset, split) in BENCHMARKS.items():
        merge_coco([(roots[dataset], split)], lists / f'{bench}.json', limit=args.limit_eval)
        ap, ap50, summary = exp.make_evaluator(lists / f'{bench}.json', args.batch_size).evaluate(
            model, False, half=not args.no_fp16)
        results[bench] = {'AP50_95': float(ap), 'AP50': float(ap50)}
        print(f'[{bench}] AP50:95 = {ap:.4f}  AP50 = {ap50:.4f}\n{summary}')
    (work / 'benchmark.json').write_text(json.dumps({'trained_on': args.dataset, **results}, indent=1))
    print(json.dumps(results, indent=1))
    if task is not None:
        for bench, values in results.items():
            for metric, value in values.items():
                task.get_logger().report_single_value(f'{bench}/{metric}', value)
        if args.epochs > 0:
            task.upload_artifact('best_ckpt', str(best))
        task.upload_artifact('benchmark', str(work / 'benchmark.json'))


if __name__ == '__main__':
    main()
