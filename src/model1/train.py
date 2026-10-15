"""Fresh training of the frozen recipe. Does not overwrite the released best.pt."""
from functools import partial
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

folder = Path(__file__).resolve().parent
package = folder.parents[1]
sys.path.insert(0, str(folder / 'code'))
sys.path.insert(0, str(package / 'src'))
import model_factory as factory
import research_sweep as engine
import context_corrected_sweep as halo
from mimu.pro6000_context import Pro6000ContextDataset
from mimu.iou_loss import LovaszCompositeLoss


def configure():
    settings = json.loads((package / 'resources/settings.json').read_text())
    config = json.loads((folder.parents[1] / 'resources' / folder.name / 'config.json').read_text())
    engine.ResearchUNet = partial(factory.make_model, branch={'model1': 'R052', 'model2': 'R073'}[folder.name], model_options=config['model_options'])
    data_options = {k: config[k] for k in ('boundary_sampling','distinct_batch_sources','full_context_supervision') if config.get(k)}
    engine.NativePatchDataset = partial(Pro6000ContextDataset, sampling='balanced_rare', photometric=True,
        targeted_probability=config['targeted_probability'], **data_options)
    loss_options = {k: config[k] for k in ('squared_dice','lovasz_all_classes') if config.get(k)}
    engine.HybridLoss = partial(LovaszCompositeLoss, **loss_options)
    if config.get('interior_cutout'):
        engine.batch_augment = factory.interior_cutout
    if config.get('gradient_centralization'):
        engine.centralize_gradients = factory.centralize_gradients
    engine.WEIGHTS['context448'] = 0.0
    engine.evaluate = halo.evaluate_halo
    engine.d4_eval = final_d4
    return SimpleNamespace(seed=config['seed'], epochs=config['epochs'], device=settings['device'],
        precision='fp32', output=package / 'run/retraining' / folder.name,
        data_root=(package / settings['data_root']).resolve(), learning_rate=config['lr'],
        weight_decay=config['weight_decay'], eutectic_fraction=config['eutectic_fraction'],
        targeted_probability=config['targeted_probability'], extra_config=config)


def final_d4(model, dataset, output, device):
    return halo.evaluate_halo(model, dataset, device, export=output, tta=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    args = configure()
    assert not args.output.exists(), 'Choose a new retraining output; released results are protected.'
    engine.run_method('context448', args)
