"""Seeded checkpoint evaluation using config/eval and shared config/data rules.

Usage: python eval.py --config config/eval/default.yaml
Reports masked-region RGB MSE mean and standard deviation by source and square size.
"""
import argparse
import copy
import csv
from datetime import datetime
import json
import logging
import random
from statistics import fmean, pstdev
from zoneinfo import ZoneInfo

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import (build_dataset, collate_patches, data_rules, fixed_noise_time, load_data_config,
                  prepare_splits, read_config, resolve_path)
from model import NeighborInterpolationBaseline, UNetFlowMatching, make_masked_image
from randomness import seed_everything, seed_worker
from utils import save_image_rows


def load_config(path='config/eval/default.yaml'):
    config = read_config(path)
    validate_config(config)
    return config


def validate_config(config):
    if not config['checkpoint']:
        raise ValueError('Set checkpoint in config/eval to the weight file to evaluate')
    evaluation = config['evaluation']
    masks_per_image = evaluation['masks_per_image']
    if not isinstance(masks_per_image, int) or isinstance(masks_per_image, bool) or masks_per_image <= 0:
        raise ValueError('evaluation.masks_per_image must be a positive integer')
    if evaluation['split'] not in ('test', 'validation'):
        raise ValueError('evaluation.split must be test or validation')
    if evaluation['batch_size'] <= 0 or evaluation['num_workers'] < 0:
        raise ValueError('batch_size must be positive and num_workers nonnegative')
    if evaluation['sampling_steps'] <= 0:
        raise ValueError('Use positive sampling_steps')
    if config['cpu_threads'] <= 0 or config['output']['examples_per_size'] < 0:
        raise ValueError('cpu_threads must be positive and examples_per_size nonnegative')
    count = config['output']['reconstruction_examples']
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError('output.reconstruction_examples must be a nonnegative integer')


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def _write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path, report, config, counts, model_config):
    """Human-readable settings and aggregated results for the evaluated run."""
    options = config['evaluation']
    examples = report['overall']['masked_examples']
    lines = [
        'PATCH RECONSTRUCTION EVALUATION SUMMARY',
        '=' * 92,
        f'Checkpoint: {report["checkpoint"]}',
        f'Checkpoint epoch: {report["checkpoint_epoch"]}',
        f'Transformer: {model_config["use_transformer"]} | FiLM: {model_config["use_film"]}',
        f'Data config: {config["data_config"]}',
        f'Split: {report["split"]} | Seed: {report["seed"]} | Device: {report["device"]}',
        f'Photos: {counts["total"][report["split"]]} | Masked examples: {examples}',
        f'Masks per photograph: {options["masks_per_image"]} (total across mask sizes)',
        f'Sampling steps: {options["sampling_steps"]} | Solver: Euler | Batch size: {options["batch_size"]}',
        f'Baseline enabled: {config["baseline"]["enabled"]}',
        'Baseline algorithm: nearest observed pixel (Euclidean distance; row-major tie break)',
        '',
        'Metrics use RGB [-1,1] directly, without inverse mapping. Only the missing square contributes.',
        'MSE: lower is better. First average squared errors within each masked RGB patch.',
        'Mean/std describe the resulting per-image MSE values (population std, ddof=0).',
        'This is variation across test examples, not pixels, seeds or a standard error.',
        'All-source results are image-weighted. A single-example group has std=0.',
        '',
        'OVERALL (all sources and mask sizes; one score per masked example)',
    ]
    methods = [('Model', 'model')]
    if config['baseline']['enabled']:
        methods.append(('Baseline', 'baseline'))
    for label, prefix in methods:
        lines.append(f'{label:<10} MSE={report["overall"][f"{prefix}_mse"]:.6f} '
                     f'+/- {report["overall"][f"{prefix}_mse_std"]:.6f} (std)')
    lines.extend(['', 'BY DATASET AND MASK SIZE',
                  f'{"Dataset":<18} {"Mask":>9} {"Images":>8} {"Method":<10} {"MSE mean":>12} {"MSE std":>12}',
                  '-' * 92])
    for row in report['summary']:
        for label, prefix in methods:
            lines.append(f'{row["dataset"]:<18} {str(row["mask_size"]) + "x" + str(row["mask_size"]):>9} '
                         f'{row["images"]:>8} {label:<10} {row[f"{prefix}_mse"]:>12.6f} '
                         f'{row[f"{prefix}_mse_std"]:>12.6f}')
    lines.extend(['', 'All filenames, individual scores and visualizations are in this evaluation directory.',
                  'Best/worst example selection is based on model MSE.'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _region_mse(predicted, target, mask):
    observed = ~mask.bool().expand_as(target)
    if not torch.equal(predicted[observed], target[observed]):
        raise RuntimeError('Reconstruction changed observed pixels outside the mask')
    difference = predicted.float() - target.float()
    count = mask.sum((1, 2, 3)) * target.shape[1]
    mse = (difference.square() * mask).sum((1, 2, 3)) / count
    if not torch.isfinite(mse).all():
        raise FloatingPointError('Nonfinite reconstruction metrics')
    return mse.tolist()


def _save_patch_comparison(path, target, predicted, *, x, y, size, title, baseline=None):
    """Original/baseline/model full images above the corresponding enlarged patches."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    target = target.detach().cpu().clamp(0, 1)
    predicted = predicted.detach().cpu().clamp(0, 1)
    images = [target]
    labels = ['Ground truth']
    if baseline is not None:
        images.append(baseline.detach().cpu().clamp(0, 1))
        labels.append('Baseline')
    images.append(predicted)
    labels.append('Model')
    figure = Figure(figsize=(11, 7), dpi=150, layout='constrained')
    FigureCanvasAgg(figure)
    try:
        grid = figure.add_gridspec(2, len(images))
        for column, (image, label) in enumerate(zip(images, labels)):
            axis = figure.add_subplot(grid[0, column])
            axis.imshow(image.permute(1, 2, 0).numpy(), interpolation='nearest', vmin=0, vmax=1)
            axis.add_patch(Rectangle((x - 0.5, y - 0.5), size, size, fill=False,
                                     edgecolor='#ff3b30', linewidth=2, clip_on=False))
            axis.set_title(f'{label} image')
            axis.set_axis_off()
        for column, (image, label) in enumerate(zip(images, labels)):
            axis = figure.add_subplot(grid[1, column])
            patch = image[:, y:y + size, x:x + size]
            axis.imshow(patch.permute(1, 2, 0).numpy(), interpolation='nearest', vmin=0, vmax=1)
            axis.set_title(f'{label} patch')
            axis.set_axis_off()
        figure.suptitle(title)
        figure.savefig(path)
    finally:
        figure.clear()


def _highlight_panels(panels, x, y, size):
    """Red outlines for best/worst overview grids; never modify metric inputs."""
    display = torch.stack(panels).detach().cpu().clone()
    color = display.new_tensor([1.0, 0.23, 0.19]).view(1, 3, 1, 1)
    display[:, :, y:y + 1, x:x + size] = color
    display[:, :, y + size - 1:y + size, x:x + size] = color
    display[:, :, y:y + size, x:x + 1] = color
    display[:, :, y:y + size, x + size - 1:x + size] = color
    return display


def _summarize(rows, sources, sizes, use_baseline):
    summary = []
    # Per-source results keep Hong Kong and Open Images performance visible.
    for source in ['all', *sources]:
        for size in sizes:
            selected = [row for row in rows if row['mask_size'] == size
                        and (source == 'all' or row['dataset'] == source)]
            if not selected:
                continue
            record = {'dataset': source, 'mask_size': size, 'images': len(selected)}
            record.update(_mse_stats(selected, use_baseline))
            summary.append(record)
    return summary


def _mse_stats(rows, use_baseline):
    """Population std of per-image patch MSE; never average subgroup stds."""
    result = {}
    for prefix in (['model', 'baseline'] if use_baseline else ['model']):
        values = [row[f'{prefix}_mse'] for row in rows]
        result[f'{prefix}_mse'] = fmean(values)
        result[f'{prefix}_mse_std'] = pstdev(values)
    return result


@torch.no_grad()
def evaluate(config):
    """Evaluate one exact checkpoint; do not train or modify model weights."""
    config = copy.deepcopy(config)
    validate_config(config)
    seed_everything(config['seed'])
    torch.set_num_threads(config['cpu_threads'])
    requested = config['device']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    checkpoint_path = resolve_path(config['checkpoint'])
    # The current training format contains only weights and plain metadata.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    settings = load_data_config(config['data_config'])
    if data_rules(settings) != data_rules(checkpoint['data_config']):
        raise ValueError('Shared data config differs from training; use the original rules to preserve held-out sets')
    model_config = checkpoint['config']['model']
    model = UNetFlowMatching(**model_config).to(device)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    options = config['evaluation']
    split = prepare_splits(settings)
    dataset = build_dataset(settings, options['split'], splits=split,
                            masks_per_image=options['masks_per_image'])
    loader = DataLoader(dataset, batch_size=options['batch_size'], shuffle=False,
                        num_workers=options['num_workers'], pin_memory=device.type == 'cuda',
                        worker_init_fn=seed_worker, collate_fn=collate_patches,
                        generator=torch.Generator().manual_seed(config['seed']))
    use_baseline = config['baseline']['enabled']
    baseline = NeighborInterpolationBaseline().to(device).eval() if use_baseline else None
    variant = ('transformer' if model_config['use_transformer'] else 'basic') + ('_film' if model_config['use_film'] else '')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    run_dir = resolve_path(config['output']['root']) / variant / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / 'config.json', config)
    _write_json(run_dir / 'data_config.json', settings)
    _write_json(run_dir / 'split_counts.json', split['counts'])
    logger = logging.getLogger(f'comp3057.eval.{stamp}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.FileHandler(run_dir / 'eval.log', encoding='utf-8'), logging.StreamHandler()]
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    rows, examples = [], {size: {'best': [], 'worst': []} for size in settings['mask_sizes']}
    requested_examples = config['output']['reconstruction_examples']
    comparison_indices = set(random.Random(config['seed']).sample(
        range(len(dataset)), min(requested_examples, len(dataset))))
    comparisons = []
    if comparison_indices:
        (run_dir / 'patch_comparisons').mkdir()
    source_names = list(settings['datasets'])
    # Original-photo indices follow source concatenation in prepare_splits.
    source_limits, end = [], 0
    for name in source_names:
        end += split['counts'][name][options['split']]
        source_limits.append((name, end))
    try:
        logger.info('Checkpoint: %s | epoch=%s | split=%s | device=%s | seed=%s',
                    checkpoint_path, checkpoint['epoch'], options['split'], device, config['seed'])
        logger.info('Photos=%d | masked examples=%d | sampling_steps=%d | solver=Euler',
                    len(dataset.paths), len(dataset), options['sampling_steps'])
        if requested_examples > len(dataset):
            logger.warning('Requested %d patch comparisons; only %d masked examples available',
                           requested_examples, len(dataset))
        for batch in tqdm(loader, desc='Evaluation', disable=not config['output']['progress']):
            images, masks = batch['image'].to(device), batch['mask'].to(device)
            noise_seed = config['seed'] + (60000 if options['split'] == 'test' else 50000)
            noise, _ = fixed_noise_time(batch, noise_seed, device)
            model_images = model.normalize_image(images)
            masked = make_masked_image(model_images, masks, model.mask_fill_value)
            predicted = model.sample(masked, masks, num_steps=options['sampling_steps'],
                                     noise=noise)
            model_mse = _region_mse(predicted, model_images, masks)
            filled = model_images.clone() if baseline is not None else None
            if filled is not None:
                # Padding must never become a neighbouring-pixel source.
                for i in range(len(images)):
                    height, width = int(batch['height'][i]), int(batch['width'][i])
                    filled[i:i + 1, :, :height, :width] = baseline(
                        masked[i:i + 1, :, :height, :width], masks[i:i + 1, :, :height, :width])
                baseline_mse = _region_mse(filled, model_images, masks)
            # Convert only the visualization copies back to display RGB.
            predicted_display = torch.where(masks.bool(), model.denormalize_image(predicted), images)
            filled_display = torch.where(masks.bool(), model.denormalize_image(filled), images) if filled is not None else None
            for i, index in enumerate(batch['index'].tolist()):
                height, width = int(batch['height'][i]), int(batch['width'][i])
                image_index = index // dataset.masks_per_image
                source = next(name for name, limit in source_limits if image_index < limit)
                size = int(batch['mask_size'][i])
                ys, xs = batch['mask'][i, 0].nonzero(as_tuple=True)
                row = {'dataset': source, 'image_index': image_index, 'mask_size': size,
                       'mask_x': int(xs.min()), 'mask_y': int(ys.min()),
                       'model_mse': model_mse[i]}
                if filled is not None:
                    row.update(baseline_mse=baseline_mse[i])
                rows.append(row)
                if index in comparison_indices:
                    filename = f'example_{index:06d}_size_{size}.png'
                    _save_patch_comparison(run_dir / 'patch_comparisons' / filename,
                                           images[i, :, :height, :width], predicted_display[i, :, :height, :width],
                                           x=row['mask_x'], y=row['mask_y'], size=size,
                                           baseline=filled_display[i, :, :height, :width] if filled_display is not None else None,
                                           title=f'{source} | example {index} | {size}x{size} | MSE {model_mse[i]:.5f}')
                    comparisons.append({'file': f'patch_comparisons/{filename}',
                                        'example_index': index, **row})
                if config['output']['examples_per_size']:
                    panels = [images[i]] + ([filled_display[i]] if filled_display is not None else []) + [predicted_display[i]]
                    panels = [panel[:, :height, :width] for panel in panels]
                    example = (model_mse[i], index,
                               _highlight_panels(panels, row['mask_x'], row['mask_y'], size))
                    for label in ('best', 'worst'):
                        candidates = examples[size][label]
                        candidates.append(example)
                        candidates.sort(key=lambda item: (item[0], item[1]), reverse=label == 'worst')
                        del candidates[config['output']['examples_per_size']:]
        summary = _summarize(rows, source_names, settings['mask_sizes'], use_baseline)
        report = {'checkpoint': str(checkpoint_path), 'checkpoint_epoch': checkpoint['epoch'],
                  'split': options['split'], 'seed': config['seed'], 'device': str(device),
                  'masks_per_image': options['masks_per_image'],
                  'metric_range': 'RGB [-1,1]; squared errors only inside square; per-image mean, then per-size mean',
                  'baseline': {'method': 'nearest_neighbour', 'distance': 'euclidean',
                               'tie_break': 'smallest_row_major_index'} if use_baseline else None,
                  'std_definition': {'ddof': 0, 'unit': 'per-image masked-region MSE',
                                     'across': 'examples, not pixels or repeated sampling seeds'},
                  'overall': {'masked_examples': len(rows), **_mse_stats(rows, use_baseline)},
                  'summary': summary}
        _write_json(run_dir / 'metrics.json', report)
        _write_summary(run_dir / 'summary.txt', report, config, split['counts'], model_config)
        _write_csv(run_dir / 'metrics.csv', summary)
        _write_csv(run_dir / 'per_image.csv', rows)
        _write_json(run_dir / 'patch_comparisons.json', comparisons)
        selected = {}
        for size, groups in examples.items():
            for label, candidates in groups.items():
                if candidates:
                    save_image_rows([item[2] for item in candidates], run_dir / f'{label}_{size}.png')
                    selected[f'{label}_{size}'] = [{'example_index': item[1], 'model_mse': item[0]}
                                                   for item in candidates]
        _write_json(run_dir / 'examples.json', selected)
        for row in summary:
            logger.info('%s | size=%d | images=%d | model MSE=%.6f +/- %.6f (std)%s',
                        row['dataset'], row['mask_size'], row['images'], row['model_mse'], row['model_mse_std'],
                        f" | baseline MSE={row['baseline_mse']:.6f} +/- {row['baseline_mse_std']:.6f} (std)" if use_baseline else '')
        logger.info('Results: %s', run_dir)
        logger.info('Saved %d ground-truth/predicted patch comparison images', len(comparisons))
        return {'run_dir': str(run_dir), **report}
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/eval/default.yaml', help='Evaluation YAML/JSON configuration')
    args = parser.parse_args()
    evaluate(load_config(args.config))


if __name__ == '__main__':
    main()
