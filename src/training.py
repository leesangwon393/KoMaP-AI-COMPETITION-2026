"""Training loop for the released final recipe."""
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from mimu.data import FullResolutionDataset
from checkpoint_io import save_checkpoint, write_json, cpu_tree, data_manifest
from evaluation import evaluate_halo


def model_digest(model, exclusions):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if any(key in name for key in exclusions):
            continue
        name = name.replace('decoder1.ordinary.', 'decoder1.')
        digest.update(name.encode())
        digest.update(tensor.cpu().numpy().tobytes())
    return digest.hexdigest()


def run_training(args, *, build_model, dataset_factory, loss_factory,
                 digest_exclusions, batch_augment=None, gradient_transform=None):
    """Train one released full-context recipe with explicit model-specific dependencies."""
    method = 'context448'
    output = args.output/method
    output.mkdir(parents=True, exist_ok=True)
    status_path = output/'status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('state') == 'completed':
        print('ALREADY COMPLETED', method, flush=True)
        return
    seed = getattr(args, 'seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    precision = getattr(args, 'precision', 'fp32')
    if precision not in ('fp32', 'bf16'):
        raise ValueError(f'Unsupported precision: {precision}')
    if precision == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('BF16 requires a CUDA device with BF16 support')
    model = build_model()
    initial_digest = model_digest(model, digest_exclusions)
    model.to(device)
    criterion = loss_factory(pixel_loss='ce', pixel_weight=.5, dice_weight=.5,
                           class_weights=(1, 1, 2, 1)).to(device)
    learning_rate = getattr(args, 'learning_rate', 3e-4)
    weight_decay = getattr(args, 'weight_decay', 1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    targeted_probability = getattr(args, 'targeted_probability', .5)
    train = dataset_factory(Path(args.data_root), patch=224, stride=112,
                               eutectic_fraction=getattr(args, 'eutectic_fraction', .1),
                               enriched_probability=targeted_probability)
    valid = FullResolutionDataset(Path(args.data_root), 'valid')
    valid.size, valid.stride = 224, 112
    loader_rng = torch.Generator().manual_seed(seed)
    aux_rng = torch.Generator().manual_seed(314159)
    loader = DataLoader(train, batch_size=4, shuffle=True, num_workers=0, generator=loader_rng)
    manifest = data_manifest(Path(args.data_root))
    config = {'method': method, 'epochs': args.epochs, 'seed': seed, 'size': 224, 'tile_stride': 112,
              'batch_size': 4, 'samples_per_epoch': 420, 'lr': learning_rate, 'optimizer': 'AdamW',
              'weight_decay': weight_decay, 'scheduler': 'cosine', 'precision': precision,
              'dice_class_weights': [1,1,2,1],
              'loss': '.5 CE + .5 weighted Dice + ramp * auxiliary_weight * auxiliary',
              'auxiliary_weight': 0.0, 'auxiliary_warmup': '0 through epoch 10; linear ramp to full at 20',
              'initial_common_weights_sha256': initial_digest,
              'parameters': sum(p.numel() for p in model.parameters()),
              'architecture': getattr(args, 'architecture', 'resnet34-unet-scse-decoder'), 'no_imagenet_pretrained': False,
              'sampling': f'{targeted_probability:.0%} rich (eutectic >=10%), {1-targeted_probability:.0%} all; source-uniform within pool',
              'targeted_probability': targeted_probability,
              'evaluator': 'original-resolution per-image four-class mean IoU, absent=one'}
    config.update(getattr(args, 'extra_config', {}))
    write_json(output/'config.json', config)
    history, best, best_epoch, start = [], -1., 0, 0
    if (output/'last.pt').exists():
        checkpoint = torch.load(output/'last.pt', map_location='cpu', weights_only=True)
        if checkpoint['config'] != config or checkpoint['manifest'] != manifest:
            raise ValueError('Resume config or dataset mismatch')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        history, start = checkpoint['history'], checkpoint['epoch']
        best, best_epoch = checkpoint['best_miou'], checkpoint['best_epoch']
        loader_rng.set_state(checkpoint['loader_rng'])
        aux_rng.set_state(checkpoint['aux_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state(checkpoint['cuda_rng'], device)
    write_json(output/'data_manifest.json', manifest)
    print('START', method, 'epoch', start+1, 'parameters', config['parameters'], flush=True)
    try:
        for epoch in range(start+1, args.epochs+1):
            begun = time.monotonic()
            model.train()
            sums = np.zeros(3)
            count = 0
            factor = min(1., max(0., (epoch-10)/10))
            epoch_lr = optimizer.param_groups[0]['lr']
            for images, labels, sources in loader:
                if config.get("distinct_batch_sources", False):
                    assert sources.unique().numel() == 4
                images, labels = images.to(device), labels.to(device)
                if config.get('interior_cutout', False):
                    images, labels = batch_augment(images, labels, aux_rng)
                optimizer.zero_grad(set_to_none=True)
                amp_context = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
                with amp_context:
                    logits, features = model(images, auxiliary=True)
                primary = criterion(logits.float(), labels)
                auxiliary = logits.sum() * 0
                loss = primary + factor * 0.0 * auxiliary
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                if config.get('gradient_centralization', False):
                    gradient_transform(model)
                optimizer.step()
                sums += np.array([loss.item(), primary.item(), auxiliary.item()]) * len(images)
                count += len(images)
            metrics = evaluate_halo(model, valid, device, 4, criterion, 'one')
            scheduler.step()
            improved = metrics['miou'] > best
            if improved:
                best, best_epoch = metrics['miou'], epoch
            row = {'epoch': epoch, 'train_loss': float(sums[0]/count), 'seg_loss': float(sums[1]/count),
                   'aux_loss': float(sums[2]/count), 'aux_factor': factor, 'lr': epoch_lr,
                   'val_miou': metrics['miou'], 'class_iou': metrics['class_iou'],
                   'per_image_miou': metrics['per_image_miou'], 'seconds': time.monotonic()-begun}
            history.append(row)
            checkpoint = {'epoch': epoch, 'method': method, 'model': cpu_tree(model.state_dict()),
                          'optimizer': cpu_tree(optimizer.state_dict()), 'scheduler': scheduler.state_dict(),
                          'config': config, 'manifest': manifest, 'best_miou': best, 'best_epoch': best_epoch,
                          'history': history, 'loader_rng': loader_rng.get_state(),
                          'aux_rng': aux_rng.get_state(), 'torch_rng': torch.get_rng_state(),
                          'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}
            save_checkpoint(output/'last.pt', checkpoint)
            if improved:
                save_checkpoint(output/'best.pt', checkpoint)
            write_json(output/'history.json', history)
            write_json(status_path, {'state': 'training', 'method': method, 'pid': os.getpid(),
                                    'completed_epochs': epoch, 'target_epochs': args.epochs,
                                    'best_epoch': best_epoch, 'best_miou': best, 'latest': row})
            print(f'{method} {epoch}/{args.epochs} loss={row["train_loss"]:.4f} '
                  f'mIoU={metrics["miou"]:.5f} best={best:.5f}@{best_epoch} '
                  f'seconds={row["seconds"]:.1f}', flush=True)
        checkpoint = torch.load(output/'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'])
        single = evaluate_halo(model, valid, device, 4, criterion, 'one', output/'valid_masks')
        write_json(status_path, {'state': 'd4_evaluating', 'completed_epochs': args.epochs,
                                'best_epoch': best_epoch, 'best_miou': best})
        tta = evaluate_halo(model, valid, device, export=output/'d4_tta', tta=True)
        write_json(status_path, {'state': 'completed', 'completed_epochs': args.epochs,
                                'best_epoch': best_epoch, 'best_miou': best,
                                'single': single, 'd4': tta})
    except BaseException as exc:
        write_json(status_path, {'state': 'failed', 'error': repr(exc),
                                'completed_epochs': history[-1]['epoch'] if history else start})
        raise
