"""Tables and static scientific plots from completed and in-progress cells."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics
import shutil


def collect(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results = [dict(json.loads(p.read_text()), stage=p.relative_to(root).parts[0])
               for p in sorted(root.glob('*/*/*/result.json'))]
    if results:
        fields = sorted(set().union(*(r.keys() for r in results)))
        with (root/'results.csv').open('w') as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            w.writerows(results)
    paired = []
    lookup = {(r['stage'], r['task'], r['width'], r['load'], r['seed'], r['family'], r['d']): r for r in results}
    for key, r in lookup.items():
        baseline = lookup.get((*key[:-1], 1))
        if r['d'] == 1 or baseline is None: continue
        paired.append(dict(stage=r['stage'], task=r['task'], width=r['width'], load=r['load'],
            seed=r['seed'], family=r['family'], d=r['d'], steps=r['steps'],
            baseline_accuracy=baseline['accuracy'], smat_accuracy=r['accuracy'],
            gap_pp=100*(r['accuracy']-baseline['accuracy']),
            query_blind_oracle_accuracy=r.get('query_blind_oracle_accuracy'),
            baseline_parameters=baseline['parameters'], smat_parameters=r['parameters'],
            baseline_binding_gap_pp=baseline.get('binding_gap_pp'),
            smat_binding_gap_pp=r.get('binding_gap_pp')))
    if paired:
        with (root/'paired_comparisons.csv').open('w') as f:
            w = csv.DictWriter(f, list(paired[0]))
            w.writeheader()
            w.writerows(paired)
    confirmed = defaultdict(list)
    for r in results:
        if r['family'] in ('mamba2', 'gdn_current') and r['steps']==500:
            confirmed[(r['stage'], r['task'], r['width'], r['load'], r['family'], r['d'])].append(r)
    confirmation = ['# Three-seed confirmation at 500 steps', '',
        'Mean ± sample standard deviation across seeds. Paired gains compare the same seed and backbone. '
        'Only settings with at least three completed seeds appear. Binding is correct-query accuracy '
        'minus accuracy against another entity/chain in the same example, in percentage points.', '',
        '| Task | Width | Load | Model | Seeds | Accuracy (%) | Exact set (%) | Binding (pp) | Paired gain (pp) | Total / trainable parameters |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for (stage, task, width, load, family, d), entries in sorted(confirmed.items()):
        if len(entries)<3: continue
        def fmt(values):
            return f'{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}'
        accuracy = fmt([100*r['accuracy'] for r in entries])
        exact = fmt([100*r['exact_accuracy'] for r in entries])
        binding = fmt([r['binding_gap_pp'] for r in entries]) if all('binding_gap_pp' in r for r in entries) else '—'
        gain = '—'
        if d>1:
            baselines = [lookup.get((stage,task,width,load,r['seed'],family,1)) for r in entries]
            if all(baselines): gain = fmt([100*(r['accuracy']-b['accuracy']) for r,b in zip(entries,baselines)])
        label = 'Mamba-2' if family=='mamba2' else 'GDN'
        if d>1: label += f' + SMat d={d}'
        params = f'{entries[0]["parameters"]:,} / {entries[0]["trainable_parameters"]:,}'
        confirmation.append(f'| {task} | {width} | {load} | {label} | {len(entries)} | {accuracy} | {exact} | {binding} | {gain} | {params} |')
    (root/'confirmation.md').write_text('\n'.join(confirmation)+'\n')
    groups = defaultdict(list)
    for r in results:
        groups[(r['stage'], r['task'], r['width'])].append(r)
    report = ['# Synthetic memory experiments', '',
        'Final-step, independent-test accuracy; chance = 6.25% (16 equiprobable values). '
        'Unrestricted vocabulary predictions. Missing cells are pending, not zero. '
        'Single-seed results are preliminary. Error bars are sample standard deviations across seeds, not confidence intervals.', '']
    colors = {1:'#222222', 2:'#0072B2', 3:'#D55E00', 4:'#009E73'}
    for (stage, task, width), rows in groups.items():
        title = f'{stage}: {task}, width {width}'
        report += [f'## {title}', '', '| Model | Load | Seeds | Accuracy (%) | Exact set (%) | Binding gap (pp) | Parameters |',
                   '|---|---:|---:|---:|---:|---:|---:|']
        families = [f for f in ('mamba2', 'gdn_current', 'gdn') if any(r['family']==f for r in rows)]
        family_labels = {'mamba2':'Mamba-2', 'gdn_current':'GDN (current transport)', 'gdn':'GDN (legacy delta-pool)'}
        fig, axes = plt.subplots(1, len(families), figsize=(5*len(families), 4), sharey=True, squeeze=False)
        axes = axes[0]
        cells = defaultdict(list)
        for r in rows: cells[(r['family'], r['d'], r['load'])].append(r)
        for (family, d, load), entries in sorted(cells.items()):
            def stat(key):
                vals = [100*r[key] for r in entries]
                return f'{statistics.mean(vals):.2f}' + (f' ± {statistics.stdev(vals):.2f}' if len(vals)>1 else '')
            label = family_labels[family] if d == 1 else f'{family_labels[family]} + SMat d={d}'
            binding = '—'
            if all('binding_gap_pp' in r for r in entries):
                vv = [r['binding_gap_pp'] for r in entries]
                binding = f'{statistics.mean(vv):.2f}' + (f' ± {statistics.stdev(vv):.2f}' if len(vv)>1 else '')
            report += [f'| {label} | {load} | {len(entries)} | {stat("accuracy")} | {stat("exact_accuracy")} | {binding} | {entries[0]["parameters"]:,} |']
        for ax, family in zip(axes, families):
            for d in (1, 2, 3, 4):
                loads = sorted(n for f, dd, n in cells if f==family and dd==d)
                if not loads: continue
                means = [statistics.mean(100*r['accuracy'] for r in cells[(family,d,n)]) for n in loads]
                stds = [statistics.stdev([100*r['accuracy'] for r in cells[(family,d,n)]]) if len(cells[(family,d,n)])>1 else 0 for n in loads]
                ax.errorbar(loads, means, yerr=stds, marker='o', color=colors[d], label='Baseline' if d==1 else f'SMat d={d}')
            control_by_load = defaultdict(dict)
            for r in rows:
                if 'query_blind_oracle_accuracy' in r:
                    control_by_load[r['load']][r['seed']] = 100*r['query_blind_oracle_accuracy']
            if control_by_load:
                xx = sorted(control_by_load)
                ax.plot(xx, [statistics.mean(control_by_load[n].values()) for n in xx],
                        color='grey', ls='--', label='Query-blind oracle')
            ax.axhline(6.25, color='grey', ls=':', lw=1)
            ax.set(title=family_labels[family], xlabel='Entities (log₂ scale)' if task=='state' else 'Chains (log₂ scale)', ylim=(0, 101))
            loads_on_plot = sorted(set(r['load'] for r in rows))
            ax.set_xscale('log', base=2)
            ax.set_xticks(loads_on_plot, labels=[str(n) for n in loads_on_plot])
            ax.grid(alpha=.2)
            if ax.lines: ax.legend(fontsize=8)
        axes[0].set_ylabel('Test accuracy (%)')
        seed_counts = [len(entries) for entries in cells.values()]
        seed_label = str(min(seed_counts)) if min(seed_counts)==max(seed_counts) else f'{min(seed_counts)}–{max(seed_counts)}'
        fig.suptitle(f'{title}; n={seed_label} seeds per point')
        fig.tight_layout()
        name = f'{stage}-{task}-w{width}-accuracy'
        fig.savefig(root/(name+'.png'), dpi=160)
        fig.savefig(root/(name+'.pdf'))
        plt.close(fig)
        report += ['', f'![{title}]({name}.png)', '',
                   f'[Learning curves]({stage}-{task}-w{width}-curves.png)', '',
                   '| Load | Query-blind majority (%) | Query-blind last value (%) | Query-blind oracle (%) |',
                   '|---:|---:|---:|---:|---:|']
        for load in sorted(set(r['load'] for r in rows)):
            controls = {r['seed']: r for r in rows if r['load']==load and 'majority_value_accuracy' in r}
            if controls:
                majority = statistics.mean(100*r['majority_value_accuracy'] for r in controls.values())
                last = statistics.mean(100*r['last_value_accuracy'] for r in controls.values())
                oracle_values = [100*r['query_blind_oracle_accuracy'] for r in controls.values() if 'query_blind_oracle_accuracy' in r]
                oracle = f'{statistics.mean(oracle_values):.2f}' if oracle_values else '—'
                report += [f'| {load} | {majority:.2f} | {last:.2f} | {oracle} |']
        report += ['']
    (root/'report.md').write_text('\n'.join(report)+'\n')
    curves = defaultdict(list)
    for p in sorted(root.glob('*/*/*/metrics.jsonl')):
        stage, task = p.relative_to(root).parts[:2]
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        if rows: curves[(stage,task,rows[0]['width'])].append(rows)
    for (stage,task,width), runs in curves.items():
        loads = sorted(set(r[0]['load'] for r in runs))
        fig, axes = plt.subplots(2, len(loads), figsize=(5*len(loads), 7), squeeze=False)
        curve_groups = defaultdict(list)
        for rows in runs:
            r = rows[0]
            curve_groups[(r['family'], r['d'], r['load'])].append(rows)
        for (family, d, load), histories in curve_groups.items():
            col = loads.index(load)
            label = f'{family} d={d} n={len(histories)}'
            style = {'mamba2':'-', 'gdn':'--', 'gdn_current':':'}[family]
            by_step = defaultdict(list)
            for rows in histories:
                for row in rows: by_step[row['step']].append(row)
            steps = sorted(by_step)
            for index, key in enumerate(('train_loss', 'accuracy')):
                values = [[v[key]*(100 if key=='accuracy' else 1) for v in by_step[s]] for s in steps]
                means = [statistics.mean(v) for v in values]
                stds = [statistics.stdev(v) if len(v)>1 else 0 for v in values]
                ax = axes[index,col]
                ax.plot(steps, means, style, color=colors[d], label=label, alpha=.8)
                if len(histories)>1:
                    ax.fill_between(steps, [m-s for m,s in zip(means,stds)],
                                    [m+s for m,s in zip(means,stds)], color=colors[d], alpha=.08)
        for col, load in enumerate(loads):
            axes[0,col].set(title=f'Load {load}', ylabel='Training cross entropy', xlabel='Steps')
            axes[1,col].set(ylabel='Validation accuracy (%)', xlabel='Steps', ylim=(0,101))
            axes[1,col].axhline(6.25, color='grey', ls=':')
            for ax in axes[:,col]: ax.grid(alpha=.2)
            axes[0,col].legend(fontsize=7)
        fig.suptitle(f'{stage}: {task}, width {width}')
        fig.tight_layout()
        fig.savefig(root/f'{stage}-{task}-w{width}-curves.png', dpi=160)
        fig.savefig(root/f'{stage}-{task}-w{width}-curves.pdf')
        plt.close(fig)
    print(f'Collected {len(results)} completed runs in {root}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent/'results'/'synthetic_memory')
    p.add_argument('--snapshot')
    args = p.parse_args()
    collect(args.root)
    if args.snapshot:
        dest = args.root/'snapshots'/args.snapshot
        dest.mkdir(parents=True, exist_ok=True)
        for source in args.root.iterdir():
            if source.is_file() and source.suffix in ('.csv', '.md', '.png', '.pdf'):
                shutil.copy2(source, dest/source.name)
        print(f'Snapshot saved in {dest}', flush=True)
