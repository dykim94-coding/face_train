"""Export a trained YOLOX-s face checkpoint to ONNX (static 1x3x640x640).

    python tools/export_onnx.py --task yolox_s_face_kaggle --out ../weights/yolox_s_face_kaggle.onnx
    python tools/export_onnx.py --ckpt YOLOX_outputs/x/best_ckpt.pth --out x.onnx

Input : BGR, float32, 0..255 (no normalisation), letterboxed to the top-left with 114 padding.
Output: (1, 8400, 6) = cx, cy, w, h (pixels of the 640 input), objectness, face score.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'third_party' / 'YOLOX'))
PROJECT = 'FD-FS (Face - object detection, feasibility study)'


def export(ckpt, out, opset=13, size=640, model_size='s', num_classes=1):
    """Checkpoint -> ONNX file, checked against the PyTorch output. Returns the largest difference."""
    from face_exp import build_model
    from yolox.models.network_blocks import SiLU
    from yolox.utils import replace_module

    model = build_model(model_size, num_classes)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'] if 'model' in state else state)
    model.eval()
    model = replace_module(model, nn.SiLU, SiLU)     # export-friendly activation
    model.head.decode_in_inference = True

    dummy = torch.rand(1, 3, size, size) * 255
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, str(out), input_names=['images'], output_names=['output'],
                      opset_version=opset, dynamo=False)

    import onnxruntime as ort
    session = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
    onnx_out = session.run(None, {'images': dummy.numpy()})[0]
    with torch.no_grad():
        torch_out = model(dummy).numpy()
    diff = float(np.abs(onnx_out - torch_out).max())
    print(f'saved {out} ({out.stat().st_size / 1e6:.1f} MB) | output {onnx_out.shape} | max |onnx - torch| = {diff:.5f}')
    return diff


def main():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--ckpt', help='local .pth checkpoint')
    source.add_argument('--task', help='ClearML task name; its best_ckpt artifact is downloaded')
    p.add_argument('--project', default=PROJECT)
    p.add_argument('--out', required=True)
    p.add_argument('--opset', type=int, default=13)
    p.add_argument('--model', choices=['s', 'tiny', 'nano'], default='s', help='YOLOX size of the checkpoint')
    p.add_argument('--size', type=int, default=640, help='square input size of the exported model')
    p.add_argument('--num-classes', type=int, default=1, help='number of classes the checkpoint was trained with')
    args = p.parse_args()

    ckpt = args.ckpt
    if args.task:
        from clearml import Task
        tasks = Task.get_tasks(project_name=args.project, task_name=f'^{args.task}$',
                               task_filter={'status': ['completed'], 'order_by': ['-last_update']})
        if not tasks:
            raise SystemExit(f'No completed task named {args.task!r} in {args.project!r}')
        print(f'using task {tasks[0].id} ({tasks[0].name})')
        ckpt = tasks[0].artifacts['best_ckpt'].get_local_copy()

    export(ckpt, args.out, args.opset, args.size, args.model, args.num_classes)


if __name__ == '__main__':
    main()
