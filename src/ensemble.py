"""Combine the released models' probabilities with frozen 30/35/35 weights."""
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
from PIL import Image

from metrics import PALETTE, evaluate_folder


def combine_probabilities(paths, weights):
    """Validate correspondence before accumulating float32 probabilities in order."""
    weights = np.asarray(weights, dtype=np.float32)
    if (weights.ndim != 1 or len(paths) != len(weights) or not len(paths)
            or not np.isfinite(weights).all() or (weights <= 0).any()
            or not np.isclose(weights.sum(), 1)):
        raise ValueError('Provide one positive weight per model, summing to one')
    with ExitStack() as stack:
        inputs = [stack.enter_context(np.load(path, allow_pickle=False)) for path in paths]
        hashes = [str(item['image_sha256'].item()) for item in inputs]
        if len(set(hashes)) != 1:
            raise ValueError('Models refer to different source images')
        combined = None
        for weight, item in zip(weights, inputs):
            if not np.array_equal(item['class_ids'], np.arange(4)):
                raise ValueError('Expected class order: Primary Si, Al3Ni, Eutectic Si, Al')
            probabilities = item['probabilities']
            if (probabilities.dtype != np.float32 or probabilities.ndim != 3
                    or probabilities.shape[0] != 4 or min(probabilities.shape[1:]) < 1
                    or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or (probabilities > 1).any()
                    or not np.allclose(probabilities.sum(0), 1, atol=3e-6)):
                raise ValueError('Expected finite float32 [4,H,W] normalized probabilities')
            if combined is None:
                combined = np.zeros_like(probabilities)
            if probabilities.shape != combined.shape:
                raise ValueError('Models have different output sizes')
            combined += weight * probabilities
    return combined, hashes[0]


def main(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    settings = json.loads((root / 'resources/settings.json').read_text())
    recipe = json.loads((root / 'resources/ensemble.json').read_text())
    split = settings['split']
    if split not in ('valid', 'test'):
        raise ValueError('split must be valid or test')
    models = recipe['models']
    if models != ['model1', 'model2', 'model3'] or recipe['weights'] != [0.30, 0.35, 0.35]:
        raise ValueError('Expected frozen model1/model2/model3 30/35/35 recipe')
    folders = [root / 'run/inference' / name / split / 'probabilities' for name in models]
    files = sorted(path.name for path in folders[0].glob('*.npz'))
    if len(files) != (20 if split == 'valid' else 10):
        raise ValueError('Expected 20 Validation or 10 Test probability files')
    if any({p.name for p in folder.glob('*.npz')} != set(files) for folder in folders):
        raise ValueError('Model probability filenames do not match')
    output = root / 'run/inference/ensemble' / split
    (output / 'probabilities').mkdir(parents=True, exist_ok=True)
    (output / 'masks').mkdir(exist_ok=True)
    for filename in files:
        combined, image_hash = combine_probabilities(
            [folder / filename for folder in folders], recipe['weights']
        )
        np.savez_compressed(output / 'probabilities' / filename, probabilities=combined,
                            image_sha256=np.array(image_hash), class_ids=np.arange(4))
        mask_name = Path(filename).stem.replace('_image', '_mask') + '.png'
        Image.fromarray(PALETTE[combined.argmax(0)]).save(output / 'masks' / mask_name)
    result = {'split': split, 'models': models, 'weights': recipe['weights'], 'count': len(files)}
    if split == 'valid':
        result['metrics'] = evaluate_folder(
            output / 'masks', (root / settings['data_root']).resolve() / 'valid/masks'
        )
    (output / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
