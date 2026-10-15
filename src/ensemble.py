"""Frozen 30/35/35 soft voting. Run model infer.py scripts first."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'src'))
    from metrics import evaluate_folder
    settings = json.loads((root/'resources/settings.json').read_text())
    recipe = json.loads((root/'resources/ensemble.json').read_text())
    split = settings['split']
    assert split in ('valid','test')
    folders = [root/'run/inference'/name/split/'probabilities' for name in recipe['models']]
    files = sorted(folders[0].glob('*.npz'))
    assert len(files) == (20 if split=='valid' else 10)
    assert all({p.name for p in f.glob('*.npz')} == {p.name for p in files} for f in folders)
    output = root/'run/inference/ensemble'/split
    (output/'probabilities').mkdir(parents=True,exist_ok=True)
    (output/'masks').mkdir(exist_ok=True)
    weights = np.array(recipe['weights'],dtype=np.float32)
    assert np.isclose(weights.sum(),1)
    colors = np.array([[76,178,76],[25,76,153],[204,204,204],[153,127,76]],dtype=np.uint8)
    for file in files:
        inputs = [np.load(f/file.name,allow_pickle=False) for f in folders]
        hashes = [str(a['image_sha256'].item()) for a in inputs]
        assert len(set(hashes)) == 1
        assert all(np.array_equal(a['class_ids'],np.arange(4)) for a in inputs)
        combined = np.zeros_like(inputs[0]['probabilities'])
        for weight,a in zip(weights,inputs):
            p = a['probabilities']
            assert p.dtype == np.float32 and p.shape == combined.shape and np.isfinite(p).all()
            assert np.allclose(p.sum(0),1,atol=3e-6)
            combined += weight*p
        np.savez_compressed(output/'probabilities'/file.name,probabilities=combined,
            image_sha256=np.array(hashes[0]),class_ids=np.arange(4))
        Image.fromarray(colors[combined.argmax(0)]).save(output/'masks'/(file.stem.replace('_image','_mask')+'.png'))
    result = {'split':split,'models':recipe['models'],'weights':recipe['weights'],'count':len(files)}
    if split=='valid':result['metrics']=evaluate_folder(output/'masks',(root/settings['data_root']).resolve()/'valid/masks')
    (output/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
