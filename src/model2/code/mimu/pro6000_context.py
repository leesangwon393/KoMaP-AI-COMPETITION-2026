from __future__ import annotations

import torch
import numpy as np

from .context_data import ContextDataset


def build_class_tile_pools(labels, origins, patch_size=224, min_fraction=0.10, classes=(1, 2)):
    """Index class-rich center crops without materializing image patches."""
    pools = {int(cls): {} for cls in classes}
    for source, (label, source_origins) in enumerate(zip(labels, origins)):
        integral = np.pad((label.astype(np.int64)[:, :, None] == np.asarray(classes)).astype(np.int64),
                          ((1, 0), (1, 0), (0, 0))).cumsum(0).cumsum(1)
        for cls_index, cls in enumerate(classes):
            selected = []
            for y, x in source_origins:
                count = (integral[y + patch_size, x + patch_size, cls_index]
                         - integral[y, x + patch_size, cls_index]
                         - integral[y + patch_size, x, cls_index]
                         + integral[y, x, cls_index])
                if count >= min_fraction * patch_size * patch_size:
                    selected.append((y, x))
            if selected:
                pools[int(cls)][source] = selected
    return {cls: rows for cls, rows in pools.items() if rows}


def _draw_index(length, generator=None):
    return int(torch.randint(length, (), generator=generator).item())


def build_boundary_tile_scores(labels, origins, patch_size=224):
    scores = []
    for label, source_origins in zip(labels, origins):
        boundary = np.zeros(label.shape, dtype=bool)
        vertical = label[1:] != label[:-1]
        horizontal = label[:, 1:] != label[:, :-1]
        boundary[1:] |= vertical
        boundary[:-1] |= vertical
        boundary[:, 1:] |= horizontal
        boundary[:, :-1] |= horizontal
        integral = np.pad(boundary.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        scores.append({(y, x): int(integral[y + patch_size, x + patch_size]
                       - integral[y, x + patch_size] - integral[y + patch_size, x]
                       + integral[y, x]) for y, x in source_origins})
    return scores


def choose_balanced_rare_origin(pools, origins, generator=None, targeted_probability=0.5, boundary_scores=None, excluded_sources=()):
    """Choose a positive Al3Ni/Eutectic crop half the time, otherwise any crop."""
    classes = tuple(sorted(cls for cls, rows in pools.items() if rows))
    if classes and float(torch.rand((), generator=generator)) < targeted_probability:
        cls = classes[_draw_index(len(classes), generator)]
        sources = tuple(sorted(source for source in pools[cls] if source not in excluded_sources))
        source = sources[_draw_index(len(sources), generator)]
        if boundary_scores is None:
            y, x = pools[cls][source][_draw_index(len(pools[cls][source]), generator)]
        else:
            candidates = [pools[cls][source][_draw_index(len(pools[cls][source]), generator)] for _ in range(4)]
            y, x = max(candidates, key=boundary_scores[source].__getitem__)
        return source, y, x, cls
    sources = tuple(source for source in range(len(origins)) if source not in excluded_sources)
    source = sources[_draw_index(len(sources), generator)]
    y, x = origins[source][_draw_index(len(origins[source]), generator)]
    return source, y, x, None


def apply_photometric(image, generator=None):
    """Apply the notebook's mild grayscale contrast/brightness, gamma and noise."""
    image = image.clone()
    rand = lambda: float(torch.rand((), generator=generator).item())
    if rand() < 0.5:
        mean = image.mean()
        contrast = 0.9 + 0.2 * rand()
        brightness = -0.08 + 0.16 * rand()
        image = (image - mean) * contrast + mean + brightness
    if rand() < 0.3:
        gamma = 0.9 + 0.2 * rand()
        image = image.clamp(0, 1).pow(gamma)
    if rand() < 0.15:
        noise = torch.randn(image.shape, generator=generator, dtype=image.dtype, device=image.device)
        image = image + noise * 0.01
    return image.clamp(0, 1)


class Pro6000ContextDataset(ContextDataset):
    """Context448 input with optional class-balanced rare-phase crop selection."""
    def __init__(self, *args, sampling="eutectic", photometric=False, targeted_probability=0.5, boundary_sampling=False, full_context_supervision=False, distinct_batch_sources=False, **kwargs):
        if sampling not in ("eutectic", "balanced_rare"):
            raise ValueError(f"Unknown context sampling policy: {sampling}")
        super().__init__(*args, **kwargs)
        self.distinct_batch_sources = distinct_batch_sources
        self.batch_sources = []
        self.full_context_supervision = full_context_supervision
        self.padded_labels = [np.pad(label, 112, mode="reflect") for label in self.labels] if full_context_supervision else None
        self.sampling = sampling
        self.photometric = photometric
        self.targeted_probability = targeted_probability
        self.boundary_scores = build_boundary_tile_scores(self.labels, self.origins) if boundary_sampling else None
        self.class_pools = None
        if sampling == "balanced_rare":
            self.class_pools = build_class_tile_pools(
                self.labels, self.origins, patch_size=224,
                min_fraction=self.eutectic_fraction, classes=(1, 2),
            )
            if not self.class_pools:
                raise ValueError("No Al3Ni/Eutectic-rich center tiles available")

    def __getitem__(self, index):
        if self.sampling == "eutectic":
            image, target, source = super().__getitem__(index)
        else:
            if self.distinct_batch_sources and len(self.batch_sources) == 4:
                self.batch_sources.clear()
            source, y, x, _ = choose_balanced_rare_origin(
                self.class_pools, self.origins, targeted_probability=self.targeted_probability, boundary_scores=self.boundary_scores,
                excluded_sources=self.batch_sources if self.distinct_batch_sources else (),
            )
            if self.distinct_batch_sources:
                self.batch_sources.append(source)
            image = torch.from_numpy(self.padded[source][y:y+448, x:x+448].copy()).float()[None] / 255.0
            if self.full_context_supervision:
                target = torch.from_numpy(self.padded_labels[source][y:y+448, x:x+448].astype(np.int64, copy=True))
            else:
                target = torch.from_numpy(self.labels[source][y:y+224, x:x+224].astype(np.int64, copy=True))
            k = int(torch.randint(4, ()).item())
            image, target = image.rot90(k, (-2, -1)), target.rot90(k, (-2, -1))
            if bool(torch.randint(2, ()).item()):
                image, target = image.flip(-1), target.flip(-1)
        if self.photometric:
            image = apply_photometric(image)
        return image, target, source
