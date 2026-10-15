"""Exercise each released inference route through its real D4 tiling implementation."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SOURCE))
import inference


class ToySegmenter(torch.nn.Module):
    def forward(self, image):
        return torch.cat((image, -image, image * .5, image * 0), dim=1)


@pytest.mark.parametrize('model_name', ['model1', 'model2', 'model3'])
def test_inference_exports_native_probabilities_and_masks(tmp_path, monkeypatch, model_name):
    """Real checkpoint checks, predictor import, D4, NPZ and RGB export without GPU weights."""
    monkeypatch.syspath_prepend(str(SOURCE / model_name / 'code'))
    monkeypatch.setattr(inference, '__file__', str(tmp_path / 'src/inference.py'))
    resources = tmp_path / 'resources'
    (resources / model_name).mkdir(parents=True)
    config = {'train': {'patch': 448, 'target': 224, 'stride': 112},
              'evaluation': {'batch_size': 4}}
    (resources / model_name / 'config.json').write_text(json.dumps(config))
    (resources / 'settings.json').write_text(json.dumps(
        {'data_root': 'data', 'device': 'cpu', 'split': 'test'}))
    images = tmp_path / 'data/test/images'
    images.mkdir(parents=True)
    for index in range(10):
        Image.fromarray(np.full((224, 224), 128 + index, dtype=np.uint8)).save(
            images / f'{index:02}_image.png')
    checkpoint = tmp_path / 'run/checkpoints' / model_name / 'best.pt'
    checkpoint.parent.mkdir(parents=True)
    torch.save({'epoch': 1, 'config': config, 'model': {}}, checkpoint)
    (resources / 'checkpoints.json').write_text(json.dumps({model_name: {
        'epoch': 1, 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}))

    inference.run_inference(model_name, ToySegmenter)

    output = tmp_path / 'run/inference' / model_name / 'test'
    assert len(list((output / 'probabilities').glob('*.npz'))) == 10
    assert len(list((output / 'masks').glob('*.png'))) == 10
    with np.load(output / 'probabilities/00_image.npz') as archive:
        probabilities = archive['probabilities']
        assert probabilities.shape == (4, 224, 224)
        expected = torch.tensor([128 / 255, -128 / 255, 64 / 255, 0.]).softmax(0).numpy()
        np.testing.assert_allclose(probabilities[:, 100, 100], expected, atol=1e-6)
        assert archive['image_sha256'].item() == hashlib.sha256(
            (images / '00_image.png').read_bytes()).hexdigest()
    with Image.open(output / 'masks/00_mask.png') as mask:
        np.testing.assert_array_equal(np.array(mask),
            np.broadcast_to([76, 178, 76], (224, 224, 3)))
    assert 'metrics' not in json.loads((output / 'metrics.json').read_text())

    # Checkpoint integrity must fail before inference, including when Python uses -O.
    metadata = json.loads((resources / 'checkpoints.json').read_text())
    metadata[model_name]['epoch'] = 2
    (resources / 'checkpoints.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='epoch mismatch'):
        inference.run_inference(model_name, ToySegmenter)
    metadata[model_name]['sha256'] = '0' * 64
    (resources / 'checkpoints.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        inference.run_inference(model_name, ToySegmenter)
