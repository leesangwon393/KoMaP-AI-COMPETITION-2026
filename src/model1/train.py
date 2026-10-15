"""Fresh training of the released recipe; checkpoints are written under run/retraining/."""
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
from training import run_training
from mimu.pro6000_context import Pro6000ContextDataset
from mimu.iou_loss import LovaszCompositeLoss


def configure():
    settings = json.loads((package / 'resources/settings.json').read_text())
    config = json.loads((package / 'resources' / folder.name / 'config.json').read_text())
    return SimpleNamespace(seed=config['seed'], epochs=config['epochs'], device=settings['device'],
        precision='fp32', output=package / 'run/retraining' / folder.name,
        data_root=(package / settings['data_root']).resolve(), learning_rate=config['lr'],
        weight_decay=config['weight_decay'], eutectic_fraction=config['eutectic_fraction'],
        targeted_probability=config['targeted_probability'], extra_config=config)


if __name__ == '__main__':
    torch.set_num_threads(4)
    args = configure()
    if args.output.exists():
        raise FileExistsError(f'Retraining output already exists: {args.output}')
    config = args.extra_config
    data_options = {k: config[k] for k in ('boundary_sampling','distinct_batch_sources','full_context_supervision') if config.get(k)}
    loss_options = {k: config[k] for k in ('squared_dice','lovasz_all_classes') if config.get(k)}
    run_training(args,
        build_model=partial(factory.make_model, branch='R052', model_options=config['model_options']),
        dataset_factory=partial(Pro6000ContextDataset, sampling='balanced_rare', photometric=True,
            targeted_probability=config['targeted_probability'], **data_options),
        loss_factory=partial(LovaszCompositeLoss, **loss_options),
        batch_augment=factory.interior_cutout if config.get('interior_cutout') else None,
        gradient_transform=factory.centralize_gradients if config.get('gradient_centralization') else None,
        digest_exclusions=('projection', 'deep_supervision_head', 'boundary_head', '.snake.', '.scale_context.', 'skip_gate.', 'global_context.'))
