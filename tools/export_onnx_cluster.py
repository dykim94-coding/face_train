"""Export checkpoints to ONNX on the cluster (for machines where PyTorch cannot run) and fetch the results.

    python tools/export_onnx_cluster.py --sources <train task id>,<train task id> --queue slurm
    python tools/export_onnx_cluster.py --fetch <export task id> --out ../weights      # needs no PyTorch

Every artifact of a source task whose name ends in "ckpt" (best_ckpt, epoch_10_ckpt, ...) becomes
<source task name>__<artifact>.onnx, stored as an artifact of the export task.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'third_party' / 'YOLOX'))
PROJECT = 'FD-FS (Face - object detection, feasibility study)'
DOCKER_IMAGE = 'pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel'
PACKAGES = ['clearml', 'boto3', 'numpy<2', 'opencv-python-headless', 'loguru', 'tabulate', 'pycocotools', 'thop',
            'tensorboard', 'psutil', 'tqdm', 'onnx==1.16.2', 'onnxruntime==1.19.2']   # newer onnx needs numpy 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sources', default='', help='comma-separated ClearML training task ids')
    p.add_argument('--queue', default='', help='ClearML queue; empty = run here')
    p.add_argument('--project', default=PROJECT)
    p.add_argument('--name', default='export_onnx')
    p.add_argument('--opset', type=int, default=13)
    p.add_argument('--fetch', default='', help='export task id: download its .onnx artifacts instead of exporting')
    p.add_argument('--out', default='weights', help='folder for --fetch')
    args = p.parse_args()
    from clearml import Task

    if args.fetch:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for name, artifact in sorted(Task.get_task(task_id=args.fetch).artifacts.items()):
            if name.endswith('.onnx'):
                shutil.copy(artifact.get_local_copy(), out / name)
                print('fetched', out / name)
        return

    task = Task.init(project_name=args.project, task_name=args.name, auto_connect_arg_parser=False)
    args = argparse.Namespace(**task.connect(vars(args), name='Args'))
    task.set_base_docker(docker_image=DOCKER_IMAGE, docker_arguments='--shm-size=8g')
    task.set_packages(PACKAGES)
    if args.queue:
        task.execute_remotely(queue_name=args.queue)   # local process ends here

    from export_onnx import export          # imports PyTorch
    work = Path('onnx_export')
    for source_id in filter(None, (s.strip() for s in str(args.sources).split(','))):
        source = Task.get_task(task_id=source_id)
        for name, artifact in sorted(source.artifacts.items()):
            if not name.endswith('ckpt'):
                continue
            target = work / f'{source.name}__{name}.onnx'
            diff = export(artifact.get_local_copy(), target, int(args.opset))
            task.upload_artifact(target.name, str(target), metadata={'source_task': source_id, 'max_abs_diff_vs_torch': diff})
    task.flush(wait_for_uploads=True)


if __name__ == '__main__':
    main()
