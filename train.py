"""Config-driven training for all four conditional U-Net FM ablations.

Usage: python train.py --config config/train/default.yaml
       python train.py --use-transformer false --use-film true
       python train.py --set training.epochs=100 training.learning_rate=5e-5
Training settings and shared data settings use separate YAML/JSON files.
Test images are never evaluated
here. Return value includes history; plots use the generic utils.line_plot.
"""
import argparse
import copy
import csv
from datetime import datetime
import json
import logging
import time
from zoneinfo import ZoneInfo

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import (DEFAULT_DATA_CONFIG, build_dataset, collate_patches, fixed_noise_time,
                  load_data_config, prepare_splits, read_config, resolve_path,
                  validate_data_config)
from model import NeighborInterpolationBaseline, UNetFlowMatching, make_masked_image
from randomness import seed_everything, seed_worker
from utils import line_plot, save_image_rows


def load_config(config_path):
    """Load data only from YAML/JSON; never import or execute config code."""
    config = read_config(config_path)
    validate_config(config)
    return config


def validate_config(config):
    training, validation = (config[k] for k in ('training', 'validation'))
    if 'data' in config:
        raise ValueError('Move data settings to config/data; pass --data-config instead')
    if training['epochs'] <= 0 or training['batch_size'] <= 0 or validation['batch_size'] <= 0:
        raise ValueError('epochs and batch sizes must be positive')
    if training['num_workers'] < 0 or validation['mse_every'] <= 0 or validation['sampling_steps'] <= 0:
        raise ValueError('workers must be nonnegative; validation intervals/steps must be positive')
    if training['learning_rate'] <= 0:
        raise ValueError('learning_rate must be positive')
    masks_per_image = training['masks_per_image']
    if not isinstance(masks_per_image, int) or isinstance(masks_per_image, bool) or masks_per_image <= 0:
        raise ValueError('training.masks_per_image must be a positive integer')
    if validation['preview_images'] < 0:
        raise ValueError('preview_images must be nonnegative')


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


@torch.no_grad()
def validate(model, loader, config, data_config, device, measure_mse=False):
    model.eval()
    total_loss, count = 0.0, 0
    errors = {size: [] for size in data_config['mask_sizes']}
    baseline_errors = {size: [] for size in data_config['mask_sizes']}
    baseline = NeighborInterpolationBaseline().to(device).eval()
    previews = []
    for batch in tqdm(loader, desc='Validation', leave=False, disable=not config['output']['progress']):
        images, masks = batch['image'].to(device), batch['mask'].to(device)
        noise, times = fixed_noise_time(batch, data_config['seed'] + 50000, device)
        model_images = model.normalize_image(images)
        loss = model.flow_matching_loss(model_images, masks, noise=noise, t=times)
        # Evaluate the deterministic baseline every epoch, even without model sampling.
        # Only the real rectangle can supply neighbouring pixels, never batch padding.
        for i, size in enumerate(batch['mask_size'].tolist()):
            height, width = int(batch['height'][i]), int(batch['width'][i])
            target = model_images[i:i + 1, :, :height, :width]
            region = masks[i:i + 1, :, :height, :width]
            filled = baseline(make_masked_image(target, region, model.mask_fill_value), region)
            error = ((filled - target).square() * region).sum() / (3 * region.sum())
            if not torch.isfinite(error):
                raise FloatingPointError('Nonfinite baseline validation reconstruction')
            baseline_errors[size].append(error.item())
        if measure_mse:
            masked = make_masked_image(model_images, masks, model.mask_fill_value)
            predicted = model.sample(masked, masks, noise=noise,
                                     num_steps=config['validation']['sampling_steps'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite validation loss')
        total_loss += loss.item() * len(images)
        count += len(images)
        if measure_mse:
            per_image = ((predicted - model_images).square() * masks).sum((1, 2, 3)) / (3 * masks.sum((1, 2, 3)))
            if not torch.isfinite(per_image).all():
                raise FloatingPointError('Nonfinite validation reconstruction')
            for size, error in zip(batch['mask_size'].tolist(), per_image.tolist()):
                errors[size].append(error)
            for i in range(len(images)):
                if len(previews) >= config['validation']['preview_images']:
                    break
                # Inverse mapping is for visualization only, never the metric.
                visible = torch.where(masks[i].bool(), 0.5, images[i])
                display_prediction = torch.where(masks[i].bool(), model.denormalize_image(predicted[i]), images[i])
                height, width = int(batch['height'][i]), int(batch['width'][i])
                previews.append(torch.stack((images[i], visible, display_prediction))[:, :, :height, :width].cpu())
    metrics = {'val_loss': total_loss / count}
    for size, values in baseline_errors.items():
        if values:
            metrics[f'val_baseline_mse_{size}'] = sum(values) / len(values)
    baseline_sizes = [size for size, values in baseline_errors.items() if values]
    metrics['val_baseline_mse'] = sum(metrics[f'val_baseline_mse_{size}'] for size in baseline_sizes) / len(baseline_sizes)
    if measure_mse:
        for size, values in errors.items():
            if values:
                metrics[f'val_mse_{size}'] = sum(values) / len(values)
        # Equal weight per represented mask size; tiny validation sets may omit a size.
        measured_sizes = [s for s, values in errors.items() if values]
        metrics['val_mse'] = sum(metrics[f'val_mse_{s}'] for s in measured_sizes) / len(measured_sizes)
    return metrics, previews


def _save_history(run_dir, history):
    _write_json(run_dir / 'history.json', history)
    temporary = run_dir / 'history.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(history)
        writer.writerows(zip(*history.values()))
    temporary.replace(run_dir / 'history.csv')
    # Reuse the generic decorator each epoch: output updates without waiting
    # for the entire training function to return. No training logic in utils.
    line_plot(run_dir / 'loss.png', x='epoch', y=['train_loss', 'val_loss'],
              title='Masked Flow Matching loss', xlabel='Epoch', ylabel='Velocity MSE')(
        lambda: history)()
    _plot_mse(run_dir, history)


def _plot_mse(run_dir, history):
    """Connect measured epochs only; retain missing values in saved history."""
    mse_keys = [key for key in history if key.startswith(('val_mse_', 'val_baseline_mse_'))]
    measured = [index for index in range(len(history['epoch']))
                if any(history[key][index] is not None for key in mse_keys)]
    if measured:
        plot_history = {key: [history[key][index] for index in measured]
                        for key in ['epoch', *mse_keys]}
        labels = {key: f'{"Baseline" if key.startswith("val_baseline_") else "Model"} | {key.rsplit("_", 1)[1]} px'
                  for key in mse_keys}
        line_plot(run_dir / 'mse.png', x='epoch', y=mse_keys, marker='o', skip_missing=True, labels=labels,
                  title='Validation reconstruction by mask size', xlabel='Epoch', ylabel='Masked MSE (RGB [-1,1])')(
            lambda: plot_history)()


def _save_weights(path, payload):
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def train(config, data_config=None):
    """Train one configured variant; return run directory and recorded history."""
    config = copy.deepcopy(config)
    data_config = copy.deepcopy(data_config) if data_config is not None else load_data_config()
    validate_config(config)
    validate_data_config(data_config)
    seed = config['seed']
    seed_everything(seed)
    torch.set_num_threads(config['cpu_threads'])
    requested = config['device']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    split = prepare_splits(data_config)
    data, training = data_config, config['training']
    training_data = build_dataset(data, 'train', splits=split, training_seed=seed,
                                  masks_per_image=training['masks_per_image'])
    validation_data = build_dataset(data, 'validation', splits=split,
                                    masks_per_image=training['masks_per_image'])
    loader_generator = torch.Generator().manual_seed(seed)
    val_generator = torch.Generator().manual_seed(seed + 1)
    flow_generator = torch.Generator(device=device).manual_seed(seed + 30000)
    train_loader = DataLoader(training_data, batch_size=training['batch_size'], shuffle=True,
                              num_workers=training['num_workers'], pin_memory=device.type == 'cuda',
                              generator=loader_generator, worker_init_fn=seed_worker, persistent_workers=False,
                              collate_fn=collate_patches)
    val_loader = DataLoader(validation_data, batch_size=config['validation']['batch_size'], shuffle=False,
                            num_workers=training['num_workers'], pin_memory=device.type == 'cuda',
                            generator=val_generator, worker_init_fn=seed_worker, collate_fn=collate_patches)
    model = UNetFlowMatching(**config['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=training['learning_rate'])
    keys = ['epoch', 'train_loss', 'val_loss', 'val_mse',
            *[f'val_mse_{s}' for s in data['mask_sizes']], 'val_baseline_mse',
            *[f'val_baseline_mse_{s}' for s in data['mask_sizes']], 'learning_rate', 'seconds']
    history = {key: [] for key in keys}
    best, best_epoch = float('inf'), 0
    output_root = resolve_path(config['output']['root'])
    variant = ('transformer' if config['model']['use_transformer'] else 'basic') + ('_film' if config['model']['use_film'] else '')
    name = config['output']['name'] or variant
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
    run_dir = (output_root / name / stamp).resolve()
    if not run_dir.is_relative_to(output_root):
        raise ValueError('output.name must remain inside output.root')
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / 'config.json', config)
    _write_json(run_dir / 'data_config.json', data)
    _write_json(run_dir / 'split_counts.json', split['counts'])
    logger = logging.getLogger(f'comp3057.{run_dir.name}')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    file_handler = logging.FileHandler(run_dir / 'train.log', encoding='utf-8')
    console = logging.StreamHandler()
    for handler in (file_handler, console):
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    writer = None
    try:
        if config['output']['tensorboard']:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(run_dir / 'tensorboard'))
        logger.info('Run: %s | device=%s | parameters=%d | splits=%s', run_dir, device,
                    sum(p.numel() for p in model.parameters()), split['counts'])
        logger.info('Training masks per image=%d | masked samples per epoch=%d',
                    training['masks_per_image'], len(training_data))
        logger.info('Validation masks per image=%d | fixed masked samples=%d',
                    training['masks_per_image'], len(validation_data))
        for epoch in range(1, training['epochs'] + 1):
            begin = time.perf_counter()
            model.train()
            training_data.epoch = epoch
            total, count = 0.0, 0
            learning_rate = optimizer.param_groups[0]['lr']
            batches = tqdm(train_loader, desc=f'Epoch {epoch}/{training["epochs"]}', disable=not config['output']['progress'])
            for batch in batches:
                image, mask = batch['image'].to(device), batch['mask'].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = model.flow_matching_loss(model.normalize_image(image), mask, generator=flow_generator)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                optimizer.step()
                total += loss.item() * len(image)
                count += len(image)
                batches.set_postfix(loss=f'{loss.item():.5f}')
            measure_mse = epoch == 1 or epoch == training['epochs'] or epoch % config['validation']['mse_every'] == 0
            metrics, previews = validate(model, val_loader, config, data, device, measure_mse)
            metrics.update(epoch=epoch, train_loss=total / count, learning_rate=learning_rate,
                           seconds=time.perf_counter() - begin)
            for key in keys:
                history[key].append(metrics.get(key))
            improved = measure_mse and metrics['val_mse'] < best
            if improved:
                best, best_epoch = metrics['val_mse'], epoch
            payload = {'epoch': epoch, 'model': model.state_dict(),
                       'config': config, 'data_config': data,
                       'best_mse': best, 'best_epoch': best_epoch}
            _save_weights(run_dir / 'last.pt', payload)
            if improved:
                _save_weights(run_dir / 'best.pt', payload)
            _save_history(run_dir, history)
            if previews:
                save_image_rows(previews, run_dir / f'preview_{epoch:04d}.png')
            if writer:
                for key, value in metrics.items():
                    if key != 'epoch':
                        writer.add_scalar(key, value, epoch)
                writer.flush()
            mse_message = ', '.join(f'{k}={v:.5f}' for k, v in metrics.items()
                                    if k.startswith(('val_mse', 'val_baseline_mse')))
            logger.info('epoch=%d train_loss=%.6f val_loss=%.6f lr=%.3g seconds=%.1f %s',
                        epoch, metrics['train_loss'], metrics['val_loss'], learning_rate, metrics['seconds'], mse_message)
        logger.info('Finished: best validation MSE=%.6f at epoch %d; test set not evaluated', best, best_epoch)
        return {'run_dir': str(run_dir), 'history': history, 'best_epoch': best_epoch, 'best_mse': best}
    finally:
        if writer:
            writer.close()
        for handler in (file_handler, console):
            logger.removeHandler(handler)
            handler.close()


def apply_overrides(config, overrides):
    """Return a copied config with existing dotted keys replaced by YAML values."""
    config = copy.deepcopy(config)
    for override in overrides:
        key, separator, raw = override.partition('=')
        if not separator or not key or not raw.strip():
            raise ValueError(f'Expected KEY=VALUE, got {override!r}')
        parts = key.split('.')
        target = config
        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                raise ValueError(f'Unknown configuration key: {key}')
            target = target[part]
        if not isinstance(target, dict) or parts[-1] not in target:
            raise ValueError(f'Unknown configuration key: {key}')
        previous = target[parts[-1]]
        value = yaml.safe_load(raw)
        # YAML 1.1 can parse scientific notation such as 5e-5 as a string.
        if isinstance(previous, float) and isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        if previous is not None:
            valid = (isinstance(value, (int, float)) and not isinstance(value, bool)
                     if isinstance(previous, float) else type(value) is type(previous))
            if not valid:
                raise ValueError(f'{key} must have type {type(previous).__name__}')
        target[parts[-1]] = value
    return config


def _boolean(value):
    if value.lower() not in ('true', 'false'):
        raise argparse.ArgumentTypeError('Use true or false')
    return value.lower() == 'true'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/default.yaml', help='YAML/JSON configuration file')
    parser.add_argument('--data-config', default=DEFAULT_DATA_CONFIG, help='Shared train/eval data configuration')
    parser.add_argument('--set', dest='overrides', action='extend', nargs='+', default=[],
                        metavar='KEY=VALUE', help='Override existing training config keys, e.g. training.epochs=100')
    parser.add_argument('--use-transformer', type=_boolean, default=None,
                        help='Override model.use_transformer (true/false); takes precedence over --set')
    parser.add_argument('--use-film', type=_boolean, default=None,
                        help='Override model.use_film (true/false); takes precedence over --set')
    args = parser.parse_args(argv)
    try:
        config = apply_overrides(read_config(args.config), args.overrides)
        for name in ('use_transformer', 'use_film'):
            value = getattr(args, name)
            if value is not None:
                config['model'][name] = value
        validate_config(config)
    except (ValueError, TypeError, KeyError, yaml.YAMLError) as error:
        parser.error(str(error))
    train(config, load_data_config(args.data_config))


if __name__ == '__main__':
    main()
