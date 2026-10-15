"""Checkpoint, JSON and dataset manifest utilities."""
import hashlib
import json

import torch


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def save_checkpoint(path, data):
    tmp = path.with_suffix('.tmp')
    torch.save(data, tmp)
    tmp.replace(path)


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def data_manifest(root):
    records = []
    for path in sorted(root.glob('*/*/*.png')):
        records.append({'path': str(path.relative_to(root)),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return records
