"""Frozen output contracts: soft voting, palette, metric and train-only Cutout."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ensemble import combine_probabilities, main as ensemble_main
from metrics import PALETTE
from model_factory import interior_cutout


def save_probabilities(path, probabilities, image_hash='same-image', class_ids=None):
    np.savez_compressed(path, probabilities=probabilities,
                        image_sha256=np.array(image_hash),
                        class_ids=np.arange(4) if class_ids is None else class_ids)


def test_soft_voting_uses_probabilities_and_released_float32_order(tmp_path):
    # Two hard-mask votes prefer class0; combined probabilities prefer class1.
    distributions = [[.51, .49, 0, 0], [.51, .49, 0, 0], [0, 1, 0, 0]]
    paths = []
    for i, distribution in enumerate(distributions):
        path = tmp_path / f'{i}.npz'
        save_probabilities(path, np.array(distribution, dtype=np.float32)[:, None, None])
        paths.append(path)
    output, image_hash = combine_probabilities(paths, [.30, .35, .35])
    assert output.dtype == np.float32 and output.argmax(0).item() == 1
    np.testing.assert_array_equal(output[:, 0, 0],
                                  np.array([.3315, .6685, 0, 0], dtype=np.float32))
    assert image_hash == 'same-image'


@pytest.mark.parametrize('problem', ['hash', 'classes', 'shape', 'negative', 'nan', 'dtype', 'sum'])
def test_misaligned_or_invalid_probabilities_are_rejected(tmp_path, problem):
    p = np.full((4, 2, 2), .25, dtype=np.float32)
    first, second = tmp_path / 'first.npz', tmp_path / 'second.npz'
    save_probabilities(first, p)
    q = p.copy()
    kwargs = {}
    if problem == 'hash':
        kwargs['image_hash'] = 'other-image'
    elif problem == 'classes':
        kwargs['class_ids'] = np.array([1, 0, 2, 3])
    elif problem == 'shape':
        q = np.full((4, 3, 2), .25, dtype=np.float32)
    elif problem == 'negative':
        q[:, 0, 0] = [-.1, .6, .25, .25]  # sums to one, still invalid
    elif problem == 'nan':
        q[0, 0, 0] = np.nan
    elif problem == 'dtype':
        q = q.astype(np.float64)
    elif problem == 'sum':
        q *= .5
    save_probabilities(second, q, **kwargs)
    with pytest.raises(ValueError):
        combine_probabilities([first, second], [.5, .5])


@pytest.mark.parametrize('weights', [[1], [.5, -.5], [.2, .2], [np.nan, .5]])
def test_invalid_weight_recipes_are_rejected(tmp_path, weights):
    with pytest.raises(ValueError):
        combine_probabilities([tmp_path / 'a', tmp_path / 'b'], weights)


@pytest.mark.parametrize('split,count', [('valid', 20), ('test', 10)])
def test_ensemble_end_to_end_rgb_outputs_and_image_macro_metric(tmp_path, split, count):
    resources = tmp_path / 'resources'
    resources.mkdir()
    (resources / 'settings.json').write_text(json.dumps(
        {'split': split, 'data_root': 'data', 'device': 'cpu'}))
    (resources / 'ensemble.json').write_text(json.dumps(
        {'models': ['model1', 'model2', 'model3'], 'weights': [.3, .35, .35]}))
    target_folder = tmp_path / 'data/valid/masks'
    target_folder.mkdir(parents=True)
    for index in range(count):
        p = np.zeros((4, 1, 1), dtype=np.float32)
        p[0 if index == 0 else 3] = 1
        for model in ['model1', 'model2', 'model3']:
            folder = tmp_path / 'run/inference' / model / split / 'probabilities'
            folder.mkdir(parents=True, exist_ok=True)
            save_probabilities(folder / f'A{index}_image.npz', p, image_hash=str(index))
        Image.fromarray(PALETTE[np.array([[3]])]).save(target_folder / f'A{index}_mask.png')
    ensemble_main(tmp_path)
    output = tmp_path / 'run/inference/ensemble' / split
    assert len(list((output / 'masks').glob('*.png'))) == count
    with Image.open(output / 'masks/A0_mask.png') as image:
        np.testing.assert_array_equal(np.array(image), PALETTE[np.array([[0]])])
    result = json.loads((output / 'metrics.json').read_text())
    if split == 'valid':
        assert result['metrics']['miou'] == pytest.approx(.975)
        np.testing.assert_allclose(result['metrics']['class_iou'], [.95, 1, 1, .95])
    else:
        assert 'metrics' not in result


def test_cutout_is_per_patch_in_bounds_mean_filled_and_preserves_target():
    images = torch.arange(4 * 448 * 448, dtype=torch.float32).reshape(4, 1, 448, 448)
    images /= images.numel()
    original = images.clone()
    labels = torch.arange(4 * 448 * 448).reshape(4, 448, 448) % 4
    output, target = interior_cutout(images, labels, torch.Generator().manual_seed(42))
    assert torch.equal(images, original) and torch.equal(target, labels)
    changed_patches = 0
    for source, result in zip(images, output):
        changed = result != source
        if changed.any():
            changed_patches += 1
            _, yy, xx = changed.nonzero(as_tuple=True)
            assert changed.sum() == 64 * 64
            assert yy.max() - yy.min() == 63 and xx.max() - xx.min() == 63
            assert torch.all(result[changed] == source.mean())
    assert 0 < changed_patches < len(images)
