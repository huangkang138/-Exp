"""Make a source-checked seven-model comparison after all formal tests finish."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean

import torch

from .model import PROJECT, FIXED, FIXED_SHA

RESULTS = PROJECT / 'gb_2_maco_POMO/results'
PALETTE = {'fixed': '#34D399', 'short': '#67E8F9', 'multiopt': '#C4B5FD',
           'matsp': '#FB923C', 'epoch0': '#FDE68A', 'length': '#FBBF24',
           'multi': '#F59E0B'}
LABELS = {'fixed': '固定 Soft', 'short': '历史可学习选球',
          'multiopt': 'Multiopt B', 'matsp': 'MATSP B',
          'epoch0': '新模型 epoch 0', 'length': '新 best_length',
          'multi': '新 best_multi'}
ORDER = list(LABELS)


def load_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def source_paths(train_dir):
    return {
        'fixed': RESULTS / 'original25_hierarchical_20261015/per_instance.csv',
        'short': RESULTS / 'learnable_ball_test25_20261015/per_instance.csv',
        'multiopt': RESULTS / 'ball_bias_ABC_test25_20261016/B_multi_25/per_instance.csv',
        'matsp': RESULTS / 'ball_bias_B_matsp500_run10_test25_length/per_instance.csv',
        'epoch0': train_dir / 'test_epoch0/per_instance.csv',
        'length': train_dir / 'test_best_length/per_instance.csv',
        'multi': train_dir / 'test_best_multi/per_instance.csv',
    }


def normalize(key, row, truth):
    if key == 'short':
        r = {'instance': row['instance'], 'reference_length': row['reference'],
             'truth_count': truth['truth_count'], 'paper_fbeta': row['paper_fbeta'],
             'seed': truth['seed'], 'starts': truth['starts'],
             'samples_per_start': truth['samples_per_start']}
        for stage in ('raw', 'a', 'b'):
            for field in (('best', 'gap_pct') if stage == 'raw' else
                          ('best', 'gap_pct', 'hits', 'precision', 'recall', 'fbeta')):
                r[f'{stage}_{field}'] = row[f'learned_{stage}_{field}']
        for stage in ('a', 'b'):
            artifact = RESULTS / 'learnable_ball_test25_20261015/learned_ball/best_routes' / f'{r["instance"]}.json'
            routes = json.loads(artifact.read_text(encoding='utf-8'))
            r[f'{stage}_routes'] = str(len(routes[stage.upper()]['routes']))
            r[f'{stage}_distance_hit'] = str(int(int(r[f'{stage}_best']) == int(r['reference_length'])))
        for stage in ('inference', 'two_opt', 'optb'):
            r[f'{stage}_seconds'] = row[f'learned_{stage}_seconds']
        return r
    return row


def metrics(rows):
    return {'raw_hits': sum(int(r['raw_best']) == int(r['reference_length']) for r in rows),
            'raw_gap': mean(float(r['raw_gap_pct']) for r in rows),
            'a_hits': sum(int(r['a_best']) == int(r['reference_length']) for r in rows),
            'a_routes': sum(int(r['a_hits']) for r in rows),
            'a_fbeta': mean(float(r['a_fbeta']) for r in rows),
            'b_hits': sum(int(r['b_best']) == int(r['reference_length']) for r in rows),
            'b_routes': sum(int(r['b_hits']) for r in rows),
            'b_fbeta': mean(float(r['b_fbeta']) for r in rows),
            'b_gap': mean(float(r['b_gap_pct']) for r in rows),
            'inference': sum(float(r['inference_seconds']) for r in rows),
            'two_opt': sum(float(r['two_opt_seconds']) for r in rows),
            'optb': sum(float(r['optb_seconds']) for r in rows)}


def color(value, key):
    return f'<span style="color:{PALETTE[key]}">{value}</span>'


def html_table(headers, rows):
    return ('<table border="1" cellspacing="0" cellpadding="6">\n<thead><tr>'
            + ''.join(f'<th>{x}</th>' for x in headers) + '</tr></thead>\n<tbody>\n'
            + '\n'.join('<tr>' + ''.join(f'<td>{x}</td>' for x in row) + '</tr>' for row in rows)
            + '\n</tbody>\n</table>')


def plot_history(train_dir, records, train_rows, proxy_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = [r['epoch'] for r in records]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    axs[0, 0].plot([int(r['epoch']) for r in train_rows],
                   [float(r['mean_loss']) for r in train_rows], marker='o')
    axs[0, 0].set_title('Training loss')
    axs[0, 1].plot(x, [r['proxy_gap_pct'] for r in proxy_rows], marker='o')
    axs[0, 1].set_title('Validation gap vs across-epoch local best (%)')
    axs[1, 0].plot(x, [r['proxy_b_routes'] for r in proxy_rows], marker='o')
    axs[1, 0].set_title('Validation high-quality Opt-B routes')
    axs[1, 1].plot(x, [r['residual']['residual_max_abs'] for r in records], marker='o')
    axs[1, 1].set_title('Residual max absolute value')
    for ax in axs.flat:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(train_dir / 'training_curves.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, required=True)
    args = parser.parse_args()
    train_dir = args.train_dir.resolve()
    selection = json.loads((train_dir / 'selection.json').read_text(encoding='utf-8'))
    records = json.loads((train_dir / 'validation_history.json').read_text(encoding='utf-8'))
    train_rows = load_csv(train_dir / 'training_history.csv')
    assert len(records) == 11 and len(train_rows) == 10
    val_cases = {r['epoch']: {v['instance']: v for v in
                 load_csv(train_dir / f"validation_epoch_{r['epoch']:03d}.csv")}
                 for r in records}
    case_names = set(val_cases[0])
    assert len(case_names) == 100 and all(set(rows) == case_names for rows in val_cases.values())
    local_best = {case: min(int(val_cases[epoch][case]['two_opt_best'])
                            for epoch in val_cases) for case in case_names}
    proxy_rows = []
    for r in records:
        rows = val_cases[r['epoch']]
        hits = [case for case in case_names if int(rows[case]['two_opt_best']) == local_best[case]]
        proxy_rows.append({'epoch': r['epoch'], 'proxy_hits': len(hits),
                           'proxy_gap_pct': mean(100*(int(rows[case]['two_opt_best']) /
                                                 local_best[case]-1) for case in case_names),
                           'proxy_b_routes': sum(int(rows[case]['b_routes']) for case in hits)})
    with (train_dir / 'validation_proxy_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(proxy_rows[0]))
        writer.writeheader()
        writer.writerows(proxy_rows)
    train_by_epoch = {int(r['epoch']): r for r in train_rows}
    epoch_log = []
    for r, proxy in zip(records, proxy_rows):
        train = train_by_epoch.get(r['epoch'], {})
        epoch_log.append({'epoch': r['epoch'], 'updates': r['updates'],
                          'train_mean_raw_length': train.get('mean_raw_length', ''),
                          'train_mean_loss': train.get('mean_loss', ''),
                          'train_mean_ball_grad_norm': train.get('mean_ball_grad_norm', ''),
                          'train_positive_grad_batches': train.get('positive_grad_batches', ''),
                          'validation_2opt_vs_heuristic_pct': r['metrics']['mean_relative_2opt_pct'],
                          'validation_proxy_hits': proxy['proxy_hits'],
                          'validation_proxy_gap_pct': proxy['proxy_gap_pct'],
                          'validation_proxy_b_routes': proxy['proxy_b_routes'],
                          'residual_mean': r['residual']['residual_mean'],
                          'residual_std': r['residual']['residual_std'],
                          'residual_max_abs': r['residual']['residual_max_abs'],
                          'ball_probability_mean_abs_change': r['residual']['ball_probability_mean_abs_change'],
                          'checkpoint_sha256': r['checkpoint_sha256']})
    with (train_dir / 'epoch_log.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(epoch_log[0]))
        writer.writeheader()
        writer.writerows(epoch_log)
    initial_state = torch.load(train_dir / 'checkpoints/epoch_000.pt', map_location='cpu',
                               weights_only=True)['model_state_dict']
    final_state = torch.load(train_dir / 'checkpoints/last.pt', map_location='cpu',
                             weights_only=True)['model_state_dict']
    assert all(torch.equal(initial_state[k], final_state[k]) for k in initial_state
               if k.startswith(('encoder.', 'decoder.'))), 'fixed Soft backbone changed'
    assert any(not torch.equal(initial_state[k], final_state[k]) for k in initial_state
               if k.startswith('ball_head.')), 'residual head did not train'
    assert hashlib.sha256(FIXED.read_bytes()).hexdigest() == FIXED_SHA
    paths = source_paths(train_dir)
    source = {k: load_csv(p) for k, p in paths.items()}
    source['fixed'] = [r for r in source['fixed'] if r['mode'] == 'fixed_soft']
    truth = {r['instance']: r for r in source['fixed']}
    cases = [f'MSTSP{i}' for i in range(1, 26)]
    assert set(truth) == set(cases)
    data = {}
    for key, rows in source.items():
        mapping = {r['instance']: normalize(key, r, truth[r['instance']]) for r in rows}
        assert set(mapping) == set(cases), (key, len(mapping))
        for case in cases:
            r = mapping[case]
            for field in ('reference_length', 'truth_count', 'paper_fbeta', 'seed',
                          'starts', 'samples_per_start'):
                assert abs(float(r[field]) - float(truth[case][field])) < 1e-8, (key, case, field)
            assert int(r['samples_per_start']) == 8 and int(r['starts']) == int(truth[case]['city_count'])
        data[key] = mapping
    stats = {k: metrics(list(rows.values())) for k, rows in data.items()}
    assert (stats['fixed']['b_hits'], stats['fixed']['b_routes']) == (25, 663)
    assert abs(stats['fixed']['b_fbeta'] - 0.8968434047253819) < 1e-9
    short_protocol = json.loads((RESULTS / 'learnable_ball_test25_20261015/protocol.json').read_text(encoding='utf-8'))
    assert (short_protocol['device'], short_protocol['seed'], short_protocol['argmax'],
            short_protocol['sampling'], short_protocol['optb_extra_pair_checks_per_case']) == (
                'cpu', 42, 1, 8, 50000)
    historical_hashes = {'short': (Path(short_protocol['trained_checkpoint']),
                                    short_protocol['trained_checkpoint_sha256'])}
    for key in ('multiopt', 'matsp'):
        summary = json.loads((paths[key].parent / 'summary.json').read_text(encoding='utf-8'))
        historical_hashes[key] = (Path(summary['checkpoint']), summary['checkpoint_sha256'])
    for key, (path, digest) in historical_hashes.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, key
    for key in ('epoch0', 'length', 'multi'):
        test_dir = paths[key].parent
        summary = json.loads((test_dir / 'summary.json').read_text(encoding='utf-8'))
        checkpoint = Path(summary['checkpoint'])
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == summary['checkpoint_sha256']
        assert stats[key]['b_hits'] == summary['b_distance_hits']
        assert abs(stats[key]['b_fbeta'] - summary['mean_b_fbeta']) < 1e-9

    output = train_dir / 'comparison.md'
    lines = ['# Soft 引导残差式粒球决策：原始 25 例正式对照', '']
    same = selection['same_checkpoint']
    changed = any(r['residual']['residual_max_abs'] > 1e-6 for r in records[1:])
    lines += ['## 七个问题的直接回答', '',
              '1. **初始等价**：零残差数值测试通过，逐步联合城市概率与固定 Soft 一致，完整贪心 POMO 路线一致。epoch 0 使用两级采样，随机数调用顺序可能不同，故不要求逐条采样路线相同。',
              f"2. **25/25 距离命中**：best_length {'保持' if stats['length']['b_hits']==25 else '未保持'}（{stats['length']['b_hits']}/25）；best_multi {'保持' if stats['multi']['b_hits']==25 else '未保持'}（{stats['multi']['b_hits']}/25）。",
              f"3. **最优回路 A/B**：固定 Soft {stats['fixed']['a_routes']}/{stats['fixed']['b_routes']}、历史可学习选球 {stats['short']['a_routes']}/{stats['short']['b_routes']}、Multiopt B {stats['multiopt']['a_routes']}/{stats['multiopt']['b_routes']}、MATSP B {stats['matsp']['a_routes']}/{stats['matsp']['b_routes']}；新 best_length {stats['length']['a_routes']}/{stats['length']['b_routes']}、best_multi {stats['multi']['a_routes']}/{stats['multi']['b_routes']}。",
              f"4. **Fβ 是否超越**：新 best_length {stats['length']['b_fbeta']:.6f}、best_multi {stats['multi']['b_fbeta']:.6f}；对固定 Soft 的 0.896843 {'至少一个超过' if max(stats['length']['b_fbeta'],stats['multi']['b_fbeta'])>0.8968434047253819 else '均未超过'}，对历史可学习选球的 0.901198 {'至少一个超过' if max(stats['length']['b_fbeta'],stats['multi']['b_fbeta'])>0.901198 else '均未超过'}。",
              f"5. **两个选模结果**：best_length 选 epoch {selection['best_length']['epoch']}，best_multi 选 epoch {selection['best_multi']['epoch']}；{'同一权重，分别保留选择元数据' if same else '不同权重'}。",
              f"6. **残差是否生效**：{'训练后残差已偏离零' if changed else '未观察到残差偏离零'}；相对 epoch 0，best_multi Fβ 变化 {stats['multi']['b_fbeta']-stats['epoch0']['b_fbeta']:+.6f}，距离命中变化 {stats['multi']['b_hits']-stats['epoch0']['b_hits']:+d}。",
              f"7. **退步位置**：固定 Soft 与新 best_multi 的原始命中 {stats['fixed']['raw_hits']}→{stats['multi']['raw_hits']}、2-opt 命中 {stats['fixed']['a_hits']}→{stats['multi']['a_hits']}、Opt-B 命中 {stats['fixed']['b_hits']}→{stats['multi']['b_hits']}，逐例差异见下表。", '']
    lines += ['## 协议与颜色', '',
              '七个模型均在原始 MSTSP1–25 采用 CPU、逐例相同 seed（43–67）、全部起点、每起点 1 条 Argmax + 8 条 Sampling、全部去重原始路线完整普通 2-opt、每例 Opt-B 最多 50,000 次额外换边检查。`.solution` 仅在搜索后用于评估；论文 Table 2 Fβ 是参考值，并非同预算复测。',
              '训练用普通 MATSP 500/100；验证集无完整最优回路集合，选模使用各 epoch 在同一验证集上的局部最佳距离和多解代理，**不称其为真实 Recall 或 Fβ**。最终 25 例不参与选模。', '']
    lines.append('；'.join(color(LABELS[k], k) for k in ORDER) + '；<span style="color:#38BDF8">参考真值/论文</span>。')
    lines += ['', '## 汇总', '']
    headers = ['模型', '原始命中', '原始 Gap %', '2-opt 命中', 'A', 'Opt-B 命中', 'B', '平均 Fβ', '推理/2-opt/Opt-B 秒']
    rows = []
    for k in ORDER:
        s = stats[k]
        rows.append([color(LABELS[k], k), f"{s['raw_hits']}/25", f"{s['raw_gap']:.4f}",
                     f"{s['a_hits']}/25", s['a_routes'], f"{s['b_hits']}/25", s['b_routes'],
                     f"{s['b_fbeta']:.6f}", f"{s['inference']:.2f}/{s['two_opt']:.2f}/{s['optb']:.3f}"])
    lines += [html_table(headers, rows), '']

    for stage, field, title in [('原始路线', 'raw_best', '原始路线距离'),
                                ('普通 2-opt', 'a_best', '2-opt 后距离'),
                                ('Opt-B', 'b_best', 'Opt-B 后距离')]:
        lines += [f'## {title}', '']
        headers = ['算例', '<span style="color:#38BDF8">参考最短</span>']
        headers += [color(LABELS[k], k) for k in ORDER]
        if stage == 'Opt-B':
            headers += [color(LABELS[k]+' Gap %', k) for k in ('length', 'multi')]
        rows = []
        for case in cases:
            r = truth[case]
            row = [case, f'<span style="color:#38BDF8">{r["reference_length"]}</span>']
            row += [color(data[k][case][field], k) for k in ORDER]
            if stage == 'Opt-B':
                row += [color(f"{float(data[k][case]['b_gap_pct']):.4f}", k)
                        for k in ('length', 'multi')]
            rows.append(row)
        lines += [html_table(headers, rows), '']

    lines += ['## 真最优回路覆盖', '',
              'A/B 分别是普通 2-opt/Opt-B 最佳距离下的不同真最优回路数；旋转与反向等价只算一条。P=命中/最佳距离去重回路数，R=命中/真最优总数。', '']
    for stage in ('a', 'b'):
        headers = ['算例', '<span style="color:#38BDF8">真最优总数</span>']
        headers += [color(LABELS[k]+' '+stage.upper()+' 命中', k) for k in ORDER]
        headers += [color(LABELS[k]+' '+stage.upper()+' P/R', k) for k in ('fixed', 'length', 'multi')]
        headers += [f'新 {k}−固定命中' for k in ('length', 'multi')]
        rows = []
        for case in cases:
            row = [case, f'<span style="color:#38BDF8">{truth[case]["truth_count"]}</span>']
            row += [color(data[k][case][f'{stage}_hits'], k) for k in ORDER]
            row += [color(f"{float(data[k][case][f'{stage}_precision']):.4f}/"
                          f"{float(data[k][case][f'{stage}_recall']):.4f}", k)
                    for k in ('fixed', 'length', 'multi')]
            row += [f"{int(data[k][case][f'{stage}_hits'])-int(data['fixed'][case][f'{stage}_hits']):+d}"
                    for k in ('length', 'multi')]
            rows.append(row)
        lines += [f'### {stage.upper()} 阶段', '', html_table(headers, rows), '']

    lines += ['## Fβ 与论文 Table 2', '',
              '同阶段各实验并排；末尾是新模型 Opt-B 减固定 Soft、减论文的逐例 Fβ。论文只有 Fβ 对照，不能据此填写论文距离、命中或 Gap。', '']
    headers = ['算例'] + [color(LABELS[k]+' A', k) for k in ORDER] + [color(LABELS[k]+' B', k) for k in ORDER]
    headers += ['<span style="color:#38BDF8">论文</span>', '新 length B−固定', '新 multi B−固定', '新 length B−论文', '新 multi B−论文']
    rows = []
    for case in cases:
        paper = float(truth[case]['paper_fbeta'])
        fixed = float(data['fixed'][case]['b_fbeta'])
        row = [case]
        for stage in ('a', 'b'):
            row += [color(f"{float(data[k][case][f'{stage}_fbeta']):.4f}", k) for k in ORDER]
        row += [f'<span style="color:#38BDF8">{paper:.4f}</span>']
        row += [f"{float(data[k][case]['b_fbeta'])-fixed:+.4f}" for k in ('length', 'multi')]
        row += [f"{float(data[k][case]['b_fbeta'])-paper:+.4f}" for k in ('length', 'multi')]
        rows.append(row)
    rows.append(['25 例均值'] + [color(f"{stats[k][stage+'_fbeta']:.6f}", k)
                                    for stage in ('a', 'b') for k in ORDER]
                + [f"{mean(float(truth[c]['paper_fbeta']) for c in cases):.6f}"]
                + [f"{stats[k]['b_fbeta']-stats['fixed']['b_fbeta']:+.6f}" for k in ('length', 'multi')]
                + [f"{stats[k]['b_fbeta']-mean(float(truth[c]['paper_fbeta']) for c in cases):+.6f}"
                   for k in ('length', 'multi')])
    lines += [html_table(headers, rows), '']

    lines += ['## 关键实例与训练过程', '']
    for case in ('MSTSP21', 'MSTSP22', 'MSTSP24', 'MSTSP25'):
        lines.append('- '+case+'：参考 '+truth[case]['reference_length']+'；'+ '，'.join(
            f"{LABELS[k]} 原始/2-opt/Opt-B={data[k][case]['raw_best']}/{data[k][case]['a_best']}/{data[k][case]['b_best']}，B 真最优={data[k][case]['b_hits']}，Fβ={float(data[k][case]['b_fbeta']):.4f}"
            for k in ('fixed', 'epoch0', 'length', 'multi'))+'。')
    lines.append('')
    lines.append(f"最佳验证 checkpoint：length=epoch {selection['best_length']['epoch']}；multi=epoch {selection['best_multi']['epoch']}。质量门槛为最高局部最佳距离命中数，平均代理 Gap 不高于候选最小值 0.05 个百分点；在门槛内比较高质量去重回路数。")
    lines.append('')
    lines.append('训练曲线见 `training_curves.png`；`epoch_log.csv` 汇总 epoch 0–10 的训练、验证、代理 Gap、残差探针和 checkpoint SHA。原始记录分别在 `training_history.csv`、`validation_history.json` 与 `validation_proxy_metrics.csv`。残差来自单个固定多球验证算例，不是整个验证集的分布；改进与否应以最终测试判断。')
    lines += ['', '## 数据来源与边界', '']
    for k in ORDER:
        lines.append(f'- {LABELS[k]}：`{paths[k]}`。')
    for k in ('epoch0', 'length', 'multi'):
        summary = json.loads((paths[k].parent / 'summary.json').read_text(encoding='utf-8'))
        lines.append(f'- {LABELS[k]} checkpoint SHA-256：`{summary["checkpoint_sha256"]}`。')
    for k, (_, digest) in historical_hashes.items():
        lines.append(f'- {LABELS[k]} checkpoint SHA-256：`{digest}`。')
    lines.append(f'- 初始固定 Soft checkpoint SHA-256：`{FIXED_SHA}`。所有历史来源仅在原始 25 例协议和真值口径核对一致后纳入；历史训练数据和 checkpoint 不同，推理耗时含模型结构与历史运行差异，不能仅据单次计时定论。')
    lines.append('- 本实验只有一个种子和固定 25 例测试；普通 MATSP 验证集无完整真最优集合，代理指标不能当作真实 Fβ。')
    wide = []
    for case in cases:
        item = {'instance': case, 'reference_length': truth[case]['reference_length'],
                'truth_count': truth[case]['truth_count'], 'paper_fbeta': truth[case]['paper_fbeta']}
        for k in ORDER:
            for field in ('raw_best', 'raw_gap_pct', 'a_best', 'a_gap_pct', 'a_routes',
                          'a_hits', 'a_precision', 'a_recall', 'a_fbeta', 'b_best',
                          'b_gap_pct', 'b_routes', 'b_hits', 'b_precision', 'b_recall',
                          'b_fbeta', 'inference_seconds', 'two_opt_seconds', 'optb_seconds'):
                item[f'{k}_{field}'] = data[k][case][field]
        wide.append(item)
    with (train_dir / 'paired_25_full.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(wide[0]))
        writer.writeheader()
        writer.writerows(wide)
    output.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    plot_history(train_dir, records, train_rows, proxy_rows)
    print(output)


if __name__ == '__main__':
    main()
