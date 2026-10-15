"""Strict-load released weights and export native-resolution D4 probabilities."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
from model import build_model

folder = Path(__file__).resolve().parent
package = folder.parents[1]
sys.path.insert(0, str(package / 'src'))
from metrics import evaluate_folder


def main():
    settings = json.loads((package / 'resources/settings.json').read_text())
    config = json.loads((folder.parents[1] / 'resources' / folder.name / 'config.json').read_text())
    data = (package / settings['data_root']).resolve()
    split = settings['split']
    assert split in ('valid','test')
    files = sorted((data / split / 'images').glob('*.png'))
    assert len(files) == (20 if split == 'valid' else 10)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = folder.name != 'model3'
    torch.backends.cudnn.benchmark = False
    checkpoint_path = package / 'run/checkpoints' / folder.name / 'best.pt'
    record = json.loads((package / 'resources/checkpoints.json').read_text())[folder.name]
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == record['sha256']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == record['epoch']
    assert checkpoint['config'] == config
    model = build_model()
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(settings['device']).eval()
    del checkpoint
    output = package / 'run/inference' / folder.name / split
    (output / 'probabilities').mkdir(parents=True, exist_ok=True)
    (output / 'masks').mkdir(exist_ok=True)
    palette = np.array([[76,178,76],[25,76,153],[204,204,204],[153,127,76]],dtype=np.uint8)
    begun = time.perf_counter()
    for i, path in enumerate(files):
        image = np.array(Image.open(path).convert('L'))
        with torch.inference_mode():
            if folder.name == 'model3':
                from komap_phase.evaluation import predict_image
                probabilities = predict_image(model,image,config['train']['patch'],config['train']['target'],
                    config['train']['stride'],settings['device'],config['evaluation']['batch_size'],d4=True).cpu().numpy()
            else:
                from evaluate_context_halo import predict
                probabilities = predict(model,image,settings['device'],tta=True)[1].numpy()
        assert probabilities.shape == (4,*image.shape)
        assert probabilities.dtype == np.float32 and np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(0),1,atol=3e-6)
        np.savez_compressed(output / 'probabilities' / (path.stem+'.npz'),probabilities=probabilities,
            image_filename=np.array(path.name),image_sha256=np.array(hashlib.sha256(path.read_bytes()).hexdigest()),class_ids=np.arange(4))
        Image.fromarray(palette[probabilities.argmax(0)]).save(output / 'masks' / path.name.replace('_image','_mask'))
        print(folder.name,split,i+1,len(files),flush=True)
    result = {'model':folder.name,'split':split,'epoch':record['epoch'],'d4':True,'seconds':time.perf_counter()-begun,
              'checkpoint_sha256':record['sha256'],'device':settings['device']}
    if split == 'valid':
        result['metrics'] = evaluate_folder(output / 'masks',data / 'valid/masks')
    (output / 'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
