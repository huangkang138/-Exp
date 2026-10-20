"""Run the historical 25-case CPU protocol for one residual checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from gb_2_maco_POMO.experiments.hierarchical.original25 import PAPER, _csv, run_case
from gb_2_maco_POMO.experiments.learnable_ball_bias.rollout import generate_routes
from .model import FIXED_SHA, SoftResidualBallPolicy
from .train import write_csv, write_json


def load_selected_checkpoint(path: Path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if (state.get('experiment_type') != 'soft_residual_ball_v1' or
            state.get('fixed_checkpoint_sha256') != FIXED_SHA):
        raise RuntimeError('not a compatible Soft residual checkpoint')
    arch = state['architecture']
    model = SoftResidualBallPolicy(int(arch['embedding_dim']), int(arch['num_heads']),
                                   int(arch['encoder_layers']),
                                   residual_bound=float(state['residual_bound']))
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.eval()
    return model, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--selector', choices=('epoch_000', 'best_length', 'best_multi'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        parser.error('output directory must be new')
    checkpoint = args.train_dir / 'checkpoints' / f'{args.selector}.pt'
    model, state = load_selected_checkpoint(checkpoint)
    torch.set_num_threads(1)
    output.mkdir(parents=True)
    label = f'residual_{args.selector}'
    for folder in ('raw_routes', 'best_routes'):
        (output / label / folder).mkdir(parents=True)

    def route_generator(policy, prepared, starts, samples, seed):
        if starts != list(range(len(prepared.cities))) or samples != 8:
            raise RuntimeError('historical original-25 budget changed')
        return generate_routes(policy, prepared, samples=samples, seed=seed,
                               device=torch.device('cpu')), None

    rows = []
    paper = _csv(PAPER)
    for index in range(1, 26):
        row = run_case(model, index, label, 0.0, output,
                       float(paper[f'MSTSP{index}']['paper_fbeta']),
                       route_generator=route_generator)
        row['mode'] = label  # run_case otherwise uses a generic hierarchical label.
        rows.append(row)
        print(f'{label} MSTSP{index}: A={row["a_hits"]} B={row["b_hits"]}', flush=True)
    write_csv(output / 'per_instance.csv', rows)
    summary = {'experiment_type': state['experiment_type'], 'selector': args.selector,
               'epoch': state['epoch'], 'checkpoint': str(checkpoint),
               'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               'fixed_checkpoint_sha256': FIXED_SHA, 'cases': len(rows),
               'raw_distance_hits': sum(r['raw_best'] == r['reference_length'] for r in rows),
               'a_distance_hits': sum(r['a_distance_hit'] for r in rows),
               'b_distance_hits': sum(r['b_distance_hit'] for r in rows),
               'mean_raw_gap_pct': sum(r['raw_gap_pct'] for r in rows) / 25,
               'mean_a_gap_pct': sum(r['a_gap_pct'] for r in rows) / 25,
               'mean_b_gap_pct': sum(r['b_gap_pct'] for r in rows) / 25,
               'a_true_optimal_routes': sum(r['a_hits'] for r in rows),
               'b_true_optimal_routes': sum(r['b_hits'] for r in rows),
               'mean_a_fbeta': sum(r['a_fbeta'] for r in rows) / 25,
               'mean_b_fbeta': sum(r['b_fbeta'] for r in rows) / 25,
               'inference_seconds': sum(r['inference_seconds'] for r in rows),
               'two_opt_seconds': sum(r['two_opt_seconds'] for r in rows),
               'optb_seconds': sum(r['optb_seconds'] for r in rows)}
    write_json(output / 'summary.json', summary)


if __name__ == '__main__':
    main()
