"""Freeze fixed Soft city policy and train its residual Ball Head for ten epochs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import tomllib
from pathlib import Path
from statistics import mean

import torch

from gb_2_maco_POMO.runner import prepare_tsp
from gb_2_maco_POMO.experiments.shared.data import batches_by_city_count, load_manifest
from gb_2_maco_POMO.experiments.learnable_ball_bias.rollout import geometry_batch, rollout_batch
from .model import PROJECT, load_fixed
from .validation import evaluate, select_checkpoints


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@torch.inference_mode()
def residual_probe(model, item, device):
    model.eval()
    problem = prepare_tsp(item.path)
    features, mapping, geometry = geometry_batch([problem], device)
    _, encoded = model.encode(features)
    n = len(problem.cities)
    first = torch.arange(n, device=device)[None]
    available = torch.ones((1, n, n), dtype=torch.bool, device=device)
    available.scatter_(2, first[..., None], False)
    ball_logits, city_logits = model.action_logits(encoded, first, first, available,
                                                   mapping, geometry)
    groups = ball_logits.shape[-1]
    membership = mapping[:, None, None, :] == torch.arange(groups, device=device)[None, None, :, None]
    base = torch.logsumexp(city_logits[:, :, None, :].masked_fill(~membership, -torch.inf), -1)
    valid = torch.isfinite(base)
    residual = (ball_logits - base)[valid]
    original = torch.softmax(base, -1)
    current = torch.softmax(ball_logits, -1)
    return {'residual_mean': float(residual.mean()),
            'residual_std': float(residual.std(unbiased=False)),
            'residual_max_abs': float(residual.abs().max()),
            'ball_probability_mean_abs_change': float((current-original).abs().mean())}


def checkpoint(path, model, optimizer, arch, cfg, epoch, updates, fixed_sha):
    torch.save({'experiment_type': 'soft_residual_ball_v1',
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'architecture': arch, 'residual_bound': model.residual_bound,
                'epoch': epoch, 'updates': updates, 'config': cfg,
                'fixed_checkpoint_sha256': fixed_sha}, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.toml'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error('output directory must be new')
    with args.config.open('rb') as f:
        cfg = tomllib.load(f)
    if int(cfg['epochs']) != 10 or int(cfg['seed']) != 42:
        parser.error('this formal experiment requires 10 epochs and seed 42')
    if any(int(cfg[key]) < 1 for key in ('batch_size', 'validation_optb_checks')) or int(cfg['validation_samples_per_start']) < 0:
        parser.error('invalid batch or validation budget')
    device = torch.device(cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    if device.type == 'cpu':
        torch.set_num_threads(1)
    seed = int(cfg['seed'])
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    data_root = (PROJECT / cfg['data_root']).resolve()
    train, validation = load_manifest({'root': str(data_root), 'train_dir': 'train',
                                       'validation_dir': 'validation'})
    if len(train) != 500 or len(validation) != 100:
        raise RuntimeError('expected ordinary MATSP 500/100 split')
    train_hashes = {hashlib.sha256(x.path.read_bytes()).digest() for x in train}
    val_hashes = {hashlib.sha256(x.path.read_bytes()).digest() for x in validation}
    if train_hashes & val_hashes:
        raise RuntimeError('training and validation overlap by file content')
    model, arch, fixed_sha = load_fixed(cfg['residual_bound'], device)
    for p in (*model.encoder.parameters(), *model.decoder.parameters()):
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(model.ball_head.parameters(), lr=float(cfg['learning_rate']))
    output.mkdir(parents=True)
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir()
    effective = dict(cfg, data_root_resolved=str(data_root), train_cases=len(train),
                     validation_cases=len(validation), fixed_checkpoint_sha256=fixed_sha,
                     train_manifest_sha256=hashlib.sha256((data_root/'train_manifest.csv').read_bytes()).hexdigest(),
                     validation_manifest_sha256=hashlib.sha256((data_root/'validation_manifest.csv').read_bytes()).hexdigest(),
                     validation_truth='no complete optimal tour set; local-best proxy for selection')
    write_json(output / 'effective_config.json', effective)
    shutil.copy2(args.config, output / 'config.toml')
    probe_item = next(x for x in validation if len(prepare_tsp(x.path).granular_balls) > 1)
    history, records, rows_by_epoch = [], [], {}
    batch_rng = random.Random(seed)
    updates = 0
    for epoch in range(0, int(cfg['epochs']) + 1):
        if epoch:
            lengths, losses, grad_norms = [], [], []
            positive_grad_batches = 0
            for batch in batches_by_city_count(train, int(cfg['batch_size']), batch_rng):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                outcome = rollout_batch(model, batch, device, sample=True)
                if outcome.joint_log_probs is None:
                    raise RuntimeError('missing joint log probabilities')
                reward = -outcome.lengths.detach()
                advantage = reward - reward.mean(1, keepdim=True)
                loss = -(advantage * outcome.joint_log_probs).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.ball_head.parameters(),
                                                       float(cfg['gradient_clip_norm']))
                if not torch.isfinite(norm):
                    raise RuntimeError('nonfinite Ball Head gradient')
                # A homogeneous POMO batch can have zero centered advantage.
                # Check the entire epoch, not each individual batch.
                positive_grad_batches += int(float(norm) > 0)
                if any(p.grad is not None for p in (*model.encoder.parameters(),
                                                    *model.decoder.parameters())):
                    raise RuntimeError('frozen city backbone received gradients')
                optimizer.step()
                updates += 1
                lengths.append((float(outcome.lengths.detach().mean()), len(batch)))
                losses.append(float(loss.detach()))
                grad_norms.append(float(norm))
            if sum(size for _, size in lengths) != 500:
                raise RuntimeError('epoch did not cover all training cases')
            if positive_grad_batches == 0:
                raise RuntimeError('entire epoch lacks Ball Head gradient')
            train_row = {'epoch': epoch, 'updates': updates,
                         'mean_raw_length': sum(value*size for value, size in lengths)/500,
                         'mean_reward': -sum(value*size for value, size in lengths)/500,
                         'mean_loss': mean(losses), 'mean_ball_grad_norm': mean(grad_norms),
                         'max_ball_grad_norm': max(grad_norms),
                         'positive_grad_batches': positive_grad_batches}
            history.append(train_row)
            write_csv(output / 'training_history.csv', history)
        path = checkpoints / f'epoch_{epoch:03d}.pt'
        checkpoint(path, model, optimizer, arch, effective, epoch, updates, fixed_sha)
        shutil.copy2(path, checkpoints / 'last.pt')
        residual = residual_probe(model, probe_item, device)
        # The route generator seeds each validation case. Restore the training
        # RNG afterward so checkpoint validation cannot perturb RL sampling.
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
        rows, metrics = evaluate(model, validation, data_root, device,
                                 samples=int(cfg['validation_samples_per_start']),
                                 optb_checks=int(cfg['validation_optb_checks']), seed=seed)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        write_csv(output / f'validation_epoch_{epoch:03d}.csv', rows)
        record = {'epoch': epoch, 'updates': updates, 'checkpoint': str(path),
                  'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'metrics': metrics, 'residual': residual}
        records.append(record)
        rows_by_epoch[epoch] = {r['instance']: r for r in rows}
        write_json(output / 'validation_history.json', records)
        print(f'epoch {epoch}/10 updates={updates} metrics={metrics} residual={residual}', flush=True)
    length, multi, rule = select_checkpoints(records, rows_by_epoch,
                         tolerance_pct=float(cfg['multi_quality_gap_tolerance_pct']))
    shutil.copy2(length['checkpoint'], checkpoints / 'best_length.pt')
    shutil.copy2(multi['checkpoint'], checkpoints / 'best_multi.pt')
    write_json(output / 'selection.json', {'best_length': length, 'best_multi': multi,
               'rule': rule, 'test_set_used_for_selection': False,
               'same_checkpoint': length['epoch'] == multi['epoch']})
    write_json(output / 'train_result.json', {'epochs': 10, 'updates': updates,
               'best_length_epoch': length['epoch'], 'best_multi_epoch': multi['epoch'],
               'frozen_backbone': True, 'fixed_checkpoint_sha256': fixed_sha})
    print(f'complete: best_length={length["epoch"]}, best_multi={multi["epoch"]}', flush=True)


if __name__ == '__main__':
    main()
