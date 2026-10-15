"""Original-resolution single-view and D4 validation."""


def evaluate_halo(model, dataset, device, batch_size=4, criterion=None, absent='one', export=None, tta=False):
    import numpy as np
    from PIL import Image
    from tiling import predict
    from mimu.data import PALETTE
    from mimu.metrics import confusion_matrix, summarize
    from checkpoint_io import write_json
    model.eval()
    matrices = []
    if export is not None:
        export.mkdir(parents=True, exist_ok=True)
    for path, image, target in zip(dataset.paths, dataset.images, dataset.original_labels):
        prob = predict(model, image, device, tta=tta)[1 if tta else 0]
        pred = prob.argmax(0).numpy().astype(np.uint8)
        matrices.append(confusion_matrix(target, pred))
        if export is not None:
            Image.fromarray(PALETTE[pred]).save(export/path.name.replace('_image', '_mask'))
    result = summarize(matrices, absent)
    result['confusion_matrix_sum'] = np.sum(matrices, axis=0).tolist()
    result['filenames'] = [p.name for p in dataset.paths]
    result['inference'] = {'input':448, 'output_center':224, 'stride':112, 'padding':'reflect', 'd4':tta}
    if export is not None:
        write_json(export/'metrics.json', result)
    return result
