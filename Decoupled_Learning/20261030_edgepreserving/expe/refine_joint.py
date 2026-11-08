"""Exp E coarse models followed by the original external three-stage Refine."""
from __future__ import annotations

import hashlib
from pathlib import Path
import torch
from torch import nn
from expe.model import build_expe_model
from exp_b.src.model_exp_b import FullExpB, ThreeStageRefine


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class CenteredBaseAdapter(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.network = build_expe_model(variant)

    def forward(self, lr_centered, hr_centered):
        return self.network(hr_centered + .5, lr_centered + .5)


class FullExpE(FullExpB):
    def __init__(self, variant):
        nn.Module.__init__(self)
        self.base = CenteredBaseAdapter(variant)
        self.refine = ThreeStageRefine()


def initialize_phase(model, args, protocol):
    if args.phase not in ('refine', 'joint'):
        raise ValueError('Exp E external chain supports refine/joint; train Base with train_expc.py')
    if not args.base_checkpoint:
        raise ValueError('--base-checkpoint is required')
    checkpoint = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    if checkpoint.get('protocol', {}).get('variant') != args.expe_variant:
        raise ValueError('Base checkpoint variant does not match --expe-variant')
    model.base.network.load_state_dict(checkpoint['model'], strict=True)
    if args.phase == 'joint':
        if not args.refine_checkpoint:
            raise ValueError('--refine-checkpoint is required for joint')
        checkpoint = torch.load(args.refine_checkpoint, map_location='cpu', weights_only=False)
        if checkpoint.get('phase') != 'refine':
            raise ValueError('Joint requires a refine checkpoint')
        saved = checkpoint.get('protocol', {})
        for key in ('architecture', 'base_checkpoint_sha256'):
            if saved.get(key) != protocol.get(key):
                raise ValueError(f'Refine checkpoint differs in {key}')
        model.load_state_dict(checkpoint['model_state'], strict=True)


def extend_protocol(protocol, args):
    protocol.update(
        architecture=f'expe_{args.expe_variant}_three_stage_refine',
        expe_variant=args.expe_variant,
        validation_geometry='full_image' if args.full_image_validation else 'center_crop',
        base_checkpoint_sha256=file_sha256(args.base_checkpoint),
        refine_checkpoint_sha256=(file_sha256(args.refine_checkpoint) if args.refine_checkpoint else None),
        loss='delta_cascade_0.1_0.3_0.6_edge_guided' + ('_plus_0.5_coarse_smooth_l1' if args.phase == 'joint' else ''),
    )


def canonical_ids(identifiers):
    # Exp B stores the input TIFF stem; E1 hashes source IDs without -log.
    return [key[:-4] if key.endswith("-log") else key for key in identifiers]


def check_split(args, identifiers, train_indices, val_indices, protocol):
    identifiers = canonical_ids(identifiers)
    checkpoint = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    saved = checkpoint['protocol']
    for key, indices in [('train_ids_sha256', train_indices), ('validation_ids_sha256', val_indices)]:
        digest = hashlib.sha256('\n'.join(identifiers[i] for i in indices).encode()).hexdigest()
        if saved.get(key) != digest:
            raise ValueError(f'Base data split differs in {key}; preserve Base data root/seed/val-count')
        protocol[key] = digest
    if Path(saved['data_root']).resolve() != Path(args.data_root).resolve():
        raise ValueError('Base data_root differs')
