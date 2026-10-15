import numpy as np
from PIL import Image

CLASS_NAMES = ['Primary Si', 'Al3Ni', 'Eutectic Si', 'Al']
PALETTE = np.array([[76, 178, 76], [25, 76, 153], [204, 204, 204],
                    [153, 127, 76]], dtype=np.uint8)


def decode(path):
    with Image.open(path) as image:
        rgb = np.array(image.convert('RGB'))
    matches = (rgb[:,:,None,:] == PALETTE[None,None,:,:]).all(-1)
    assert matches.any(-1).all(), str(path)
    return matches.argmax(-1)


def evaluate_folder(predictions, targets):
    names, scores = [], []
    for path in sorted(predictions.glob('*.png')):
        predicted, target = decode(path), decode(targets / path.name)
        assert predicted.shape == target.shape
        matrix = np.bincount((target*4+predicted).ravel(),minlength=16).reshape(4,4)
        intersection = np.diag(matrix).astype(np.float64)
        union = matrix.sum(0)+matrix.sum(1)-intersection
        score = np.divide(intersection,union,out=np.ones(4),where=union>0)
        names.append(path.name);scores.append(score)
    assert len(names) == 20
    values = np.array(scores)
    return {'miou':float(values.mean(1).mean()),'class_iou':values.mean(0).tolist(),
            'filenames':names,'per_image_miou':values.mean(1).tolist(),'absent_policy':'one'}
