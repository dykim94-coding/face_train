"""Upload one converted dataset folder as a ClearML Dataset (stored in the clearml-data bucket).

    python tools/upload_dataset.py --folder data/kaggle --name kaggle_face --version 1.0.0

The folder's contents (images/, annotations/) become the dataset root.
"""
import argparse
from pathlib import Path

from clearml import Dataset

STORAGE = 's3://192.168.0.229:9000/clearml-data'   # NOT the default (artifacts) bucket


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--folder', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--project', default='FD-FS (Face - object detection, feasibility study)')
    p.add_argument('--parent', default='', help='parent dataset id for an incremental version')
    p.add_argument('--tags', nargs='*', default=[])
    args = p.parse_args()

    folder = Path(args.folder).resolve()
    local = sum(1 for f in folder.rglob('*') if f.is_file())
    dataset = Dataset.create(dataset_project=args.project, dataset_name=args.name,
                             dataset_version=args.version, dataset_tags=args.tags,
                             parent_datasets=[args.parent] if args.parent else None)
    if args.parent:
        dataset.sync_folder(local_path=str(folder))      # handles add / modify / delete
    else:
        dataset.add_files(path=str(folder))
    dataset.upload(output_url=STORAGE, chunk_size=512, show_progress=True)
    dataset.finalize()
    listed = dataset.list_files()
    print(f'id={dataset.id} local_files={local} dataset_files={len(listed)} sample={listed[:2]}')
    if not args.parent and len(listed) != local:
        raise SystemExit('File count mismatch between the folder and the dataset!')


if __name__ == '__main__':
    main()
