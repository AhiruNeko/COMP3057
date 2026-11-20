"""English Gradio demo with UI settings and shared config/data rules.

Run from the project root: bash scripts/ui.sh
No training/evaluation configuration files are imported or modified.
"""
import argparse
import copy
from datetime import datetime
import json
import math
import os
import secrets
import shutil
import tempfile
from pathlib import Path
from threading import Lock
from zoneinfo import ZoneInfo

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml
from torchvision.transforms import functional as TF

from data import load_data_config, pad_to_square, prepare_splits, read_config, resize_photograph, resolve_path
from model import NeighborInterpolationBaseline, UNetFlowMatching, make_masked_image, square_mask
from randomness import seed_everything

UI_OUTPUT_ROOT = resolve_path('outputs/ui')


def save_settings(path, config, side, method, x, y, size, steps, seed, random_seed):
    """Update only editable UI defaults; preserve other on-disk settings."""
    size = integer(size, 'Mask size', 1, side - 1)
    x = min(integer(x, 'Mask X', 0, side - 1), side - size)
    y = min(integer(y, 'Mask Y', 0, side - 1), side - size)
    steps = integer(steps, 'Sampling steps', 1, config['sampling']['max_steps'])
    if method not in config['methods'] or not isinstance(random_seed, bool):
        raise ValueError('Invalid UI settings.')
    seed = integer(seed, 'Fixed seed', 0, 2**32 - 1)
    latest = read_config(path)
    latest['default_method'] = method
    latest['mask'].update(x=x, y=y, size=size)
    latest['sampling'].update(steps=steps)
    latest['runtime'].update(seed=seed, random_seed=random_seed)
    target = resolve_path(path)
    content = (json.dumps(latest, indent=2) + '\n' if target.suffix.lower() == '.json'
               else yaml.safe_dump(latest, sort_keys=False, allow_unicode=True))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent,
                                         prefix=target.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    for key in ('default_method', 'mask', 'sampling', 'runtime'):
        config[key] = copy.deepcopy(latest[key])
    return f'Settings saved to {target}'


def clear_ui_outputs(output_root):
    """Empty only the explicitly authorized outputs/ui directory at startup."""
    root = Path(output_root)
    if root.is_symlink() or root.resolve() != UI_OUTPUT_ROOT.resolve():
        raise ValueError('UI output root must be the project outputs/ui directory, not a symlink.')
    root.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        if child.is_symlink():
            child.unlink()
        elif child.is_dir():
            if not child.resolve().is_relative_to(root.resolve()):
                raise ValueError('Refusing to remove a directory outside outputs/ui.')
            shutil.rmtree(child)
        else:
            child.unlink()


def integer(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or value is None:
        raise ValueError(f'{name} must be an integer.')
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f'{name} must be an integer.')
    number = int(number)
    if number < minimum or (maximum is not None and number > maximum):
        raise ValueError(f'{name} is outside its allowed range.')
    return number


def load_config(path='config/ui/default.yaml'):
    config = read_config(path)
    shared_data = load_data_config(config['data_config'])
    side = shared_data['image_size']
    mask = config['mask']
    size = integer(mask['size'], 'Mask size', 1, side - 1)
    integer(mask['x'], 'Mask X', 0, side - size)
    integer(mask['y'], 'Mask Y', 0, side - size)
    integer(config['runtime']['seed'], 'Seed', 0, 2**32 - 1)
    if not isinstance(config['runtime']['random_seed'], bool):
        raise ValueError('random_seed must be true or false.')
    for field in ('image_display_size', 'patch_display_size'):
        integer(config['visualization'][field], field, 1, 2048)
    integer(config['runtime']['cpu_threads'], 'CPU threads', 1)
    integer(config['sampling']['steps'], 'Sampling steps', 1, config['sampling']['max_steps'])
    integer(config['app']['port'], 'Port', 1, 65535)
    if config['default_method'] not in config['methods']:
        raise ValueError('default_method must name a configured method.')
    for method in config['methods'].values():
        if method['kind'] not in ('baseline', 'model'):
            raise ValueError('Method kind must be baseline or model.')
    for key in ('display_color', 'outline_color'):
        if len(mask[key]) != 3:
            raise ValueError('Mask colors must contain three RGB values.')
        for value in mask[key]:
            integer(value, 'RGB value', 0, 255)
    return config


class ReconstructionDemo:
    """Shared inference backend; Gradio state keeps each user's input separate."""
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.data = load_data_config(config['data_config'])
        self.side = self.data['image_size']
        seed_everything(config['runtime']['seed'])
        torch.set_num_threads(config['runtime']['cpu_threads'])
        requested = config['runtime']['device']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else torch.device(requested)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is not available.')
        self.baseline = NeighborInterpolationBaseline().to(self.device).eval()
        self._model, self._model_key, self._checkpoint_data = None, None, None
        self._lock = Lock()
        self.test_images, self.dataset_errors = {}, {}
        for name, source in self.data['datasets'].items():
            rules = copy.deepcopy(self.data)
            rules['datasets'] = {name: source}
            try:
                self.test_images[name] = prepare_splits(rules)['test']
            except (ValueError, FileNotFoundError) as exc:
                self.test_images[name] = []
                self.dataset_errors[name] = str(exc)

    def photograph(self, image):
        if image is None:
            raise ValueError('Upload a photograph first.')
        image = resize_photograph(image, self.side)
        if min(image.size) < 2:
            raise ValueError('Photograph is too narrow after resizing; increase data.image_size.')
        return image

    def dataset_choices(self, name):
        return [(f'{index + 1:04d} | {Path(path).name}', str(index))
                for index, path in enumerate(self.test_images.get(name, []))]

    def dataset_photograph(self, name, index):
        paths = self.test_images.get(name, [])
        if not paths:
            raise ValueError(f'No test photographs available for {name}. {self.dataset_errors.get(name, "")}')
        index = integer(index, 'Test image index', 0, len(paths) - 1)
        with Image.open(paths[index]) as source:
            image = self.photograph(source)
        return image, {'kind': 'dataset', 'dataset': name, 'test_index': index, 'filename': Path(paths[index]).name}

    def mask_values(self, x, y, size, image=None):
        width, height = image.size if image is not None else (self.side, self.side)
        size = integer(size, 'Mask size', 1, min(width, height) - 1)
        return (integer(x, 'Mask X', 0, width - size),
                integer(y, 'Mask Y', 0, height - size), size)

    def clamp_mask(self, image, x, y, size):
        width, height = image.size if image is not None else (self.side, self.side)
        size = max(1, min(int(size), min(width, height) - 1))
        return max(0, min(int(x), width - size)), max(0, min(int(y), height - size)), size

    def preview(self, image, x, y, size):
        if image is None:
            return None
        x, y, size = self.clamp_mask(image, x, y, size)
        preview = image.copy()
        ImageDraw.Draw(preview).rectangle((x, y, x + size - 1, y + size - 1),
                                          outline=tuple(self.config['mask']['outline_color']), width=2)
        return self.display(preview)

    def display(self, image, patch=False):
        side = self.config['visualization']['patch_display_size' if patch else 'image_display_size']
        scale = side / max(image.size)
        dimensions = tuple(max(1, round(value * scale)) for value in image.size)
        return image.resize(dimensions, Image.Resampling.NEAREST)

    def trajectory_preview(self, step, trajectory):
        """Display one cached sampling state; never run the model again."""
        if not trajectory:
            return None, 'No Flow Matching trajectory. Run a neural reconstruction first.'
        step = integer(step, 'Trajectory step', 0, trajectory['steps'])
        with Image.open(trajectory['original']) as source:
            frame = source.convert('RGB')
        with Image.open(trajectory['patches'][step]) as patch:
            frame.paste(patch.convert('RGB'), (trajectory['x'], trajectory['y']))
        caption = f'Step {step}/{trajectory["steps"]} | t = {step / trajectory["steps"]:.3f}'
        if step == 0:
            caption += ' | Initial noise'
        return self.display(frame), caption

    def method_status(self, key):
        method = self.config['methods'][key]
        if method['kind'] == 'baseline':
            return 'Ready. Deterministic nearest-neighbour interpolation; no checkpoint required.'
        if not method.get('checkpoint'):
            return f'Not configured. Set methods.{key}.checkpoint in the UI configuration.'
        if not resolve_path(method['checkpoint']).is_file():
            return 'Checkpoint not found. Update its path in the UI configuration.'
        return 'Checkpoint available. Loaded on first reconstruction.'

    def _load_model(self, key):
        method = self.config['methods'][key]
        path_value = method.get('checkpoint')
        if not path_value:
            raise ValueError(f'No checkpoint configured for {method["label"]}. Set methods.{key}.checkpoint.')
        path = resolve_path(path_value)
        if not path.is_file():
            raise ValueError(f'Checkpoint not found: {path_value}')
        if self._model_key != (key, str(path)):
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            model_config = checkpoint['config']['model']
            for flag in ('use_transformer', 'use_film'):
                if model_config[flag] != method[flag]:
                    raise ValueError(f'Checkpoint module flags do not match {method["label"]}. Use its separately trained weights.')
            if checkpoint['data_config']['image_size'] != self.side:
                raise ValueError('UI image_size differs from the checkpoint working resolution.')
            model = UNetFlowMatching(**model_config)
            model.load_state_dict(checkpoint['model'], strict=True)
            self._model = None  # keep at most one neural method resident on the GPU
            self._model = model.to(self.device).eval()
            self._model_key = (key, str(path))
            self._checkpoint_data = checkpoint['data_config']
        return self._model

    def _check_held_out(self, source):
        if source.get('kind') != 'dataset':
            return
        name = source['dataset']
        original = self._checkpoint_data
        selected = self.data
        if original['seed'] != selected['seed'] or name not in original['datasets']:
            raise ValueError('Shared data test split rules do not match this checkpoint. Review config/data.')
        for key in ('train_fraction', 'val_fraction', 'test_fraction'):
            if original['datasets'][name][key] != selected['datasets'][name][key]:
                raise ValueError('UI test split fractions differ from the checkpoint training data.')
        if resolve_path(original['datasets'][name]['path']) != resolve_path(selected['datasets'][name]['path']):
            raise ValueError('UI dataset path differs from this checkpoint training dataset.')

    @torch.no_grad()
    def reconstruct(self, image, source, key, x, y, size, steps, seed, random_seed=False):
        if image is None:
            raise ValueError('Upload a photograph or load a dataset test image first.')
        if max(image.size) != self.side:
            raise ValueError('Load the photograph before reconstructing it.')
        if key not in self.config['methods']:
            raise ValueError('Choose a configured reconstruction method.')
        x, y, size = self.mask_values(x, y, size, image)
        steps = integer(steps, 'Sampling steps', 1, self.config['sampling']['max_steps'])
        if not isinstance(random_seed, bool):
            raise ValueError('random_seed must be true or false.')
        seed = secrets.randbits(32) if random_seed else integer(seed, 'Seed', 0, 2**32 - 1)
        method = self.config['methods'][key]
        source = source or {'kind': 'upload'}
        trajectory = []

        def capture(step, state):
            # Cache every actual step, but only the patch, keeping memory/storage bounded.
            patch = model.denormalize_image(state[0, :, y:y + size, x:x + size])
            pixels = patch.cpu().permute(1, 2, 0).numpy()
            trajectory.append((step, np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)))

        with self._lock:
            seed_everything(seed)
            tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)
            model_tensor = tensor * 2 - 1
            mask = square_mask(tensor, x, y, size)
            if method['kind'] == 'baseline':
                prediction = self.baseline(make_masked_image(model_tensor, mask, 0.0), mask)
            else:
                model = self._load_model(key)
                self._check_held_out(source)
                # Match training batch padding, then discard it before display/metrics.
                padded_tensor = pad_to_square(model_tensor, value=-1.0)
                padded_mask = pad_to_square(mask)
                noise = torch.randn(padded_tensor.shape, generator=torch.Generator().manual_seed(seed)).to(self.device)
                prediction = model.sample(make_masked_image(padded_tensor, padded_mask, model.mask_fill_value), padded_mask,
                                          noise=noise, num_steps=steps, step_callback=capture)
                prediction = prediction[:, :, :image.height, :image.width]
            observed = ~mask.bool().expand_as(tensor)
            if not torch.equal(prediction[observed], model_tensor[observed]):
                raise RuntimeError('Reconstruction modified observed pixels.')
            mse = float(((prediction - model_tensor).square() * mask).sum() / (3 * mask.sum()))
            pixels = ((prediction[0] + 1) / 2).cpu().permute(1, 2, 0).numpy()
        original = np.array(image, dtype=np.uint8, copy=True)
        reconstructed = original.copy()
        # Composite only the hole after quantization, preserving all observed RGB bytes.
        reconstructed[y:y + size, x:x + size] = np.rint(np.clip(pixels[y:y + size, x:x + size], 0, 1) * 255).astype(np.uint8)
        visible = original.copy()
        visible[y:y + size, x:x + size] = self.config['mask']['display_color']
        masked_image, result = Image.fromarray(visible), Image.fromarray(reconstructed)
        stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        directory = resolve_path(self.config['output']['root']) / stamp
        directory.mkdir(parents=True, exist_ok=False)
        image.save(directory / 'original.png')
        masked_image.save(directory / 'masked_input.png')
        result.save(directory / 'reconstructed.png')
        box = (x, y, x + size, y + size)
        original_patch, result_patch = image.crop(box), result.crop(box)
        original_patch.save(directory / 'original_patch.png')
        result_patch.save(directory / 'reconstructed_patch.png')
        frames = None
        if trajectory:
            patch_paths = []
            for step, patch_pixels in trajectory:
                path = directory / f'flow_patch_step_{step:04d}.png'
                Image.fromarray(patch_pixels).save(path)
                patch_paths.append(str(path))
            frames = {'original': str(directory / 'original.png'), 'patches': patch_paths,
                      'x': x, 'y': y, 'steps': steps}
        record = {'method': key, 'label': method['label'], 'source': source,
                  'checkpoint': method.get('checkpoint'), 'seed': seed, 'random_seed': random_seed,
                  'image_size': self.side, 'width': image.width, 'height': image.height,
                  'trajectory_steps': [step for step, _ in trajectory],
                  'mask': {'x': x, 'y': y, 'size': size}, 'steps': steps if method['kind'] == 'model' else None,
                  'masked_mse': mse,
                  'device': str(self.device)}
        (directory / 'request.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        (directory / 'ui_config.json').write_text(json.dumps(self.config, indent=2), encoding='utf-8')
        (directory / 'data_config.json').write_text(json.dumps(self.data, indent=2), encoding='utf-8')
        return {'masked': masked_image, 'reconstructed': result, 'file': str(directory / 'reconstructed.png'),
                'status': f'{method["label"]} | Masked MSE: {mse:.6f} | Seed: {seed} | Saved to {directory}',
                'record': record, 'original_patch': self.display(original_patch, patch=True),
                'result_patch': self.display(result_patch, patch=True), 'trajectory': frames}


def build_app(config, config_path=None):
    backend = ReconstructionDemo(config)
    initial = config['mask']
    datasets = list(backend.data['datasets'])
    first_choices = backend.dataset_choices(datasets[0])
    with gr.Blocks(title=config['app']['title'], analytics_enabled=False) as app:
        gr.Markdown(f'# {config["app"]["title"]}')
        gr.Markdown(f'Upload a photograph or choose a test image. The longest side is resized to '
                    f'**{backend.side} pixels**, preserving the aspect ratio without cropping. '
                    'Mask coordinates refer to this working image. '
                    'Observed pixels outside the square are preserved exactly at this resolution.')
        current = gr.State(None)
        source_state = gr.State({'kind': 'upload'})
        trajectory_state = gr.State(None)
        with gr.Row():
            with gr.Column():
                with gr.Tabs():
                    with gr.Tab('Upload photograph'):
                        upload = gr.Image(label='Upload photograph', sources=['upload'], type='pil', format='png')
                    with gr.Tab('Dataset test images'):
                        dataset_name = gr.Dropdown(datasets, value=datasets[0], label='Dataset')
                        photo_choice = gr.Dropdown(first_choices, value=first_choices[0][1] if first_choices else None,
                                                   label='Test photograph')
                        load_photo = gr.Button('Load selected test photograph')
                source_status = gr.Markdown('No photograph loaded.')
            with gr.Column():
                preview = gr.Image(label='Working image and square mask location', type='pil', format='png',
                                   interactive=False, buttons=['fullscreen'], height=config['visualization']['image_display_size'])
                gr.Markdown('Click the working image to place the square top-left corner, or use the controls below.')
        with gr.Row():
            x = gr.Slider(0, backend.side - initial['size'], value=initial['x'], step=1, label='Mask X (left column)')
            y = gr.Slider(0, backend.side - initial['size'], value=initial['y'], step=1, label='Mask Y (top row)')
            size = gr.Slider(1, backend.side - 1, value=initial['size'], step=1, label='Square size (pixels)')
        gr.Markdown(f'Configured training-size reference: {backend.data["mask_sizes"]} pixels. '
                    'Other square sizes are allowed, but reconstruction quality may vary.')
        with gr.Row():
            method = gr.Dropdown([(value['label'], key) for key, value in config['methods'].items()],
                                 value=config['default_method'], label='Reconstruction method', scale=2)
            steps = gr.Slider(1, config['sampling']['max_steps'], value=config['sampling']['steps'], step=1,
                              label='Flow sampling steps (model only)')
            seed = gr.Number(value=config['runtime']['seed'], precision=0, minimum=0, maximum=2**32 - 1,
                             label='Noise seed (model only)')
            random_seed = gr.Checkbox(value=config['runtime']['random_seed'], label='Random seed each run',
                                      min_width=140)
        method_status = gr.Markdown(backend.method_status(config['default_method']))
        run = gr.Button('Reconstruct patch', variant='primary')
        settings_status = gr.Markdown('Control changes are saved automatically to the UI configuration.'
                                      if config_path else 'Settings persistence is disabled for this embedded app.')
        with gr.Row():
            masked_output = gr.Image(label='Masked input', type='pil', format='png', interactive=False,
                                     buttons=['fullscreen'], height=config['visualization']['image_display_size'])
            reconstructed_output = gr.Image(label='Reconstructed photograph', type='pil', format='png', interactive=False,
                                            buttons=['fullscreen'], height=config['visualization']['image_display_size'])
        status = gr.Markdown()
        with gr.Tabs():
            with gr.Tab('Enlarged patch comparison'):
                with gr.Row():
                    original_patch = gr.Image(label='Original patch (ground truth)', type='pil', interactive=False,
                                              height=config['visualization']['patch_display_size'])
                    result_patch = gr.Image(label='Reconstructed patch', type='pil', interactive=False,
                                            height=config['visualization']['patch_display_size'])
            with gr.Tab('Flow Matching trajectory'):
                gr.Markdown('Drag the step slider to inspect any state from the same reconstruction, '
                            'including initial noise (step 0) and the final result. No new inference is performed. '
                            'Display values are clipped to [0, 1]; sampling states are not. '
                            'The baseline has no Flow Matching trajectory.')
                flow_step = gr.Slider(0, 1, value=0, step=1, precision=0, interactive=False,
                                      label='Trajectory step')
                flow_image = gr.Image(label='Sampling state', type='pil', format='png', interactive=False,
                                      buttons=['fullscreen'], height=config['visualization']['image_display_size'])
                flow_status = gr.Markdown('Run a neural reconstruction to inspect its sampling trajectory.')
        gr.Markdown('Models are loaded lazily from separate checkpoints. This demo never trains models. '
                    'GPU inference may compete with a running training job; use runtime.device: cpu if needed.')

        def loaded_result(image, source, message, mask_x, mask_y, mask_size):
            mask_x, mask_y, mask_size = backend.clamp_mask(image, mask_x, mask_y, mask_size)
            return (image, source, backend.preview(image, mask_x, mask_y, mask_size),
                    f'{message} | Working image: {image.width} x {image.height}.',
                    None, None, '', None, None, None,
                    gr.Slider(minimum=0, maximum=1, value=0, interactive=False), None,
                    'Run a neural reconstruction to inspect its sampling trajectory.',
                    gr.Slider(maximum=image.width - mask_size, value=mask_x),
                    gr.Slider(maximum=image.height - mask_size, value=mask_y),
                    gr.Slider(maximum=min(image.size) - 1, value=mask_size))

        def use_upload(image, mask_x, mask_y, mask_size):
            if image is None:
                return (None, {'kind': 'upload'}, None, 'No photograph loaded.', None, None, '', None, None, None,
                        gr.Slider(minimum=0, maximum=1, value=0, interactive=False), None,
                        'Run a neural reconstruction to inspect its sampling trajectory.',
                        gr.skip(), gr.skip(), gr.skip())
            image = backend.photograph(image)
            return loaded_result(image, {'kind': 'upload'}, 'Active input: uploaded photograph.', mask_x, mask_y, mask_size)

        def choices(name):
            items = backend.dataset_choices(name)
            return gr.Dropdown(choices=items, value=items[0][1] if items else None)

        def use_dataset(name, index, mask_x, mask_y, mask_size):
            try:
                image, source = backend.dataset_photograph(name, index)
                return loaded_result(image, source, f'Active input: {name}, test photograph {int(index) + 1}.',
                                     mask_x, mask_y, mask_size)
            except (ValueError, OSError) as exc:
                raise gr.Error(str(exc)) from exc

        def resize_mask(mask_size, mask_x, mask_y, image):
            width, height = image.size if image is not None else (backend.side, backend.side)
            mask_x, mask_y, mask_size = backend.clamp_mask(image, mask_x, mask_y, mask_size)
            return (gr.Slider(maximum=width - mask_size, value=mask_x),
                    gr.Slider(maximum=height - mask_size, value=mask_y),
                    gr.Slider(maximum=min(width, height) - 1, value=mask_size),
                    backend.preview(image, mask_x, mask_y, mask_size))

        def click_mask(image, mask_size, event: gr.SelectData):
            if image is None:
                raise gr.Error('Load a photograph first.')
            displayed = backend.display(image)
            mask_x, mask_y, mask_size = backend.clamp_mask(
                image, int(event.index[0] * image.width / displayed.width),
                int(event.index[1] * image.height / displayed.height), mask_size)
            return mask_x, mask_y, backend.preview(image, mask_x, mask_y, mask_size)

        def predict(image, source, chosen_method, mask_x, mask_y, mask_size, sample_steps, noise_seed, randomize):
            try:
                result = backend.reconstruct(image, source, chosen_method, mask_x, mask_y, mask_size,
                                             sample_steps, noise_seed, randomize)
                actual_seed = result['record']['seed']
                if config_path is not None:
                    with backend._lock:
                        save_settings(config_path, backend.config, backend.side, chosen_method,
                                      mask_x, mask_y, mask_size, sample_steps, actual_seed, randomize)
                trajectory = result['trajectory']
                last_step = trajectory['steps'] if trajectory else 0
                frame, caption = backend.trajectory_preview(last_step, trajectory)
                return (backend.display(result['masked']), backend.display(result['reconstructed']),
                        result['status'], result['original_patch'], result['result_patch'], trajectory,
                        gr.Slider(minimum=0, maximum=max(1, last_step), value=last_step, interactive=bool(trajectory)),
                        frame, caption, actual_seed)
            except (ValueError, OSError, RuntimeError, KeyError) as exc:
                raise gr.Error(str(exc)) from exc

        def persist(chosen_method, mask_x, mask_y, mask_size, sample_steps, noise_seed, randomize):
            try:
                with backend._lock:
                    message = save_settings(config_path, backend.config, backend.side, chosen_method,
                                            mask_x, mask_y, mask_size, sample_steps, noise_seed, randomize)
                return message
            except (ValueError, OSError) as exc:
                raise gr.Error(f'Could not save UI settings: {exc}') from exc

        load_outputs = [current, source_state, preview, source_status, masked_output, reconstructed_output, status,
                        original_patch, result_patch, trajectory_state, flow_step, flow_image, flow_status, x, y, size]
        upload.change(use_upload, [upload, x, y, size], load_outputs)
        dataset_name.change(choices, dataset_name, photo_choice)
        load_photo.click(use_dataset, [dataset_name, photo_choice, x, y, size], load_outputs)
        method.change(backend.method_status, method, method_status)
        x.change(backend.preview, [current, x, y, size], preview)
        y.change(backend.preview, [current, x, y, size], preview)
        size.input(resize_mask, [size, x, y, current], [x, y, size, preview])
        preview.select(click_mask, [current, size], [x, y, preview])
        flow_step.input(backend.trajectory_preview, [flow_step, trajectory_state], [flow_image, flow_status],
                        queue=False, show_progress='hidden', trigger_mode='always_last')
        if config_path is not None:
            settings_inputs = [method, x, y, size, steps, seed, random_seed]
            for control in settings_inputs:
                control.change(persist, settings_inputs, settings_status, concurrency_limit=1,
                               concurrency_id='save_ui_settings')
        run.click(predict, [current, source_state, method, x, y, size, steps, seed, random_seed],
                  [masked_output, reconstructed_output, status, original_patch, result_patch,
                   trajectory_state, flow_step, flow_image, flow_status, seed],
                  concurrency_limit=1)
    app.queue(default_concurrency_limit=1)
    return app, backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/ui/default.yaml', help='Independent UI YAML/JSON configuration')
    args = parser.parse_args()
    config = load_config(args.config)
    app, _ = build_app(config, args.config)
    output_root = resolve_path(config['output']['root'])
    clear_ui_outputs(output_root)
    print(f'Cleared previous UI outputs: {output_root}', flush=True)
    app.launch(server_name=config['app']['host'], server_port=config['app']['port'],
               share=config['app']['share'], inbrowser=config['app']['open_browser'],
               max_file_size=config['app']['max_file_size'], allowed_paths=[str(output_root)],
               footer_links=[], show_error=False)


if __name__ == '__main__':
    main()
