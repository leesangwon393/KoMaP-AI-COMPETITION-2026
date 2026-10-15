"""Strict-load released weights and export native-resolution D4 probabilities."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch

from metrics import PALETTE, evaluate_folder


def run_inference(model_name, build_model):
    """Export a released model's original-resolution probabilities and masks.

    Each model is run in its own process because its frozen imports differ.
    """
    if model_name not in ('model1', 'model2', 'model3'):
        raise ValueError(f'Unknown model: {model_name}')
    package = Path(__file__).resolve().parents[1]
    folder = package / 'src' / model_name
    settings = json.loads((package / 'resources/settings.json').read_text())
    config = json.loads((folder.parents[1] / 'resources' / folder.name / 'config.json').read_text())
    data = (package / settings['data_root']).resolve()
    split = settings['split']
    if split not in ('valid', 'test'):
        raise ValueError(f'Unsupported split: {split}')
    files = sorted((data / split / 'images').glob('*.png'))
    expected_count = 20 if split == 'valid' else 10
    if len(files) != expected_count:
        raise ValueError(f'Expected {expected_count} {split} images, found {len(files)}')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = folder.name != 'model3'
    torch.backends.cudnn.benchmark = False
    checkpoint_path = package / 'run/checkpoints' / folder.name / 'best.pt'
    record = json.loads((package / 'resources/checkpoints.json').read_text())[folder.name]
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError(f'{model_name} checkpoint SHA256 mismatch')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['epoch'] != record['epoch']:
        raise ValueError(f'{model_name} checkpoint epoch mismatch')
    if checkpoint['config'] != config:
        raise ValueError(f'{model_name} checkpoint config mismatch')
    model = build_model()
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(settings['device']).eval()
    del checkpoint
    output = package / 'run/inference' / folder.name / split
    (output / 'probabilities').mkdir(parents=True, exist_ok=True)
    (output / 'masks').mkdir(exist_ok=True)
    begun = time.perf_counter()
    for i, path in enumerate(files):
        with Image.open(path) as source:
            image = np.array(source.convert('L'))
        with torch.inference_mode():
            if folder.name == 'model3':
                from komap_phase.evaluation import predict_image
                probabilities = predict_image(model,image,config['train']['patch'],config['train']['target'],
                    config['train']['stride'],settings['device'],config['evaluation']['batch_size'],d4=True).cpu().numpy()
            else:
                from tiling import predict
                probabilities = predict(model,image,settings['device'],tta=True)[1].numpy()
        if probabilities.shape != (4,*image.shape):
            raise ValueError(f'Probability shape mismatch: {path.name}')
        if probabilities.dtype != np.float32 or not np.isfinite(probabilities).all():
            raise ValueError(f'Invalid probability dtype or values: {path.name}')
        if not np.allclose(probabilities.sum(0),1,atol=3e-6):
            raise ValueError(f'Probability sum mismatch: {path.name}')
        np.savez_compressed(output / 'probabilities' / (path.stem+'.npz'),probabilities=probabilities,
            image_filename=np.array(path.name),image_sha256=np.array(hashlib.sha256(path.read_bytes()).hexdigest()),class_ids=np.arange(4))
        Image.fromarray(PALETTE[probabilities.argmax(0)]).save(output / 'masks' / path.name.replace('_image','_mask'))
        print(folder.name,split,i+1,len(files),flush=True)
    result = {'model':folder.name,'split':split,'epoch':record['epoch'],'d4':True,'seconds':time.perf_counter()-begun,
              'checkpoint_sha256':record['sha256'],'device':settings['device']}
    if split == 'valid':
        result['metrics'] = evaluate_folder(output / 'masks',data / 'valid/masks')
    (output / 'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
