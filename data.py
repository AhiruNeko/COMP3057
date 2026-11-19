"""Shared, seed-based data configuration, splitting and masks for train/eval.

No manifests are written: each source is sorted and independently shuffled,
then its three partitions are appended to the corresponding combined sets.
"""
import copy
import json
import math
from pathlib import Path
import random

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torch.utils.data import default_collate
from torch.nn import functional as F
from torchvision.transforms import functional as TF
import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_CONFIG = 'config/data/default.yaml'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read_config(config_path):
    path = resolve_path(config_path)
    with path.open(encoding='utf-8') as stream:
        if path.suffix.lower() == '.json':
            config = json.load(stream)
        elif path.suffix.lower() in ('.yaml', '.yml'):
            config = yaml.safe_load(stream)
        else:
            raise ValueError('Configuration must be YAML or JSON')
    if not isinstance(config, dict):
        raise ValueError('Configuration must contain a mapping')
    return config


def validate_data_config(config):
    if not isinstance(config['seed'], int) or isinstance(config['seed'], bool) or not 0 <= config['seed'] < 2**32:
        raise ValueError('data seed must be an integer in [0, 2**32)')
    if not isinstance(config['image_size'], int) or config['image_size'] < 16:
        raise ValueError('image_size must be an integer >= 16')
    sizes = config['mask_sizes']
    if len(set(sizes)) < 2 or any(not isinstance(s, int) or isinstance(s, bool)
                                or not 0 < s <= config['image_size'] for s in sizes):
        raise ValueError('Provide at least two distinct square sizes fitting the image')
    datasets = config['datasets']
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError('datasets must be a nonempty mapping of names to settings')
    paths = []
    for name, source in datasets.items():
        if name == 'total':
            raise ValueError('Dataset name "total" is reserved for aggregate counts')
        ratios = [source[key] for key in ('train_fraction', 'val_fraction', 'test_fraction')]
        if any(isinstance(r, bool) or not isinstance(r, (int, float)) or not 0 < r < 1 for r in ratios):
            raise ValueError(f'{name}: all three fractions must be positive and below 1')
        if not math.isclose(sum(ratios), 1.0, abs_tol=1e-9):
            raise ValueError(f'{name}: train/val/test fractions must sum to 1')
        path = resolve_path(source['path'])
        if any(path == other or path in other.parents or other in path.parents for other in paths):
            raise ValueError('Dataset roots must not overlap (each file should be split once)')
        paths.append(path)


def load_data_config(path=DEFAULT_DATA_CONFIG):
    config = read_config(path)
    validate_data_config(config)
    return config


def data_rules(config):
    """Only settings affecting model inputs/splits, excluding download metadata."""
    return {'seed': config['seed'], 'image_size': config['image_size'],
            'mask_sizes': list(config['mask_sizes']),
            'datasets': [{'name': name, **{key: source[key] for key in
                         ('path', 'train_fraction', 'val_fraction', 'test_fraction')}}
                         for name, source in config['datasets'].items()]}


def prepare_splits(config):
    """Build in-memory partitions from rules only, never a saved list of names.

    All sources use independent Random(seed) instances. Selecting/reordering
    another source does not change any existing source's partition. Ratios
    apply per source: round validation/test counts; training gets the remainder.
    """
    validate_data_config(config)
    split = {'train': [], 'validation': [], 'test': [], 'counts': {},
             'specification': copy.deepcopy(config)}
    for name, source in config['datasets'].items():
        directory = resolve_path(source['path'])
        if not directory.is_dir():
            raise FileNotFoundError(f'Image directory not found: {directory}')
        paths = [str(path) for path in sorted(directory.rglob('*'))
                 if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        random.Random(config['seed']).shuffle(paths)
        n_val = round(len(paths) * source['val_fraction'])
        n_test = round(len(paths) * source['test_fraction'])
        n_train = len(paths) - n_val - n_test
        if min(n_train, n_val, n_test) < 1:
            raise ValueError(f'{name}: too few images for nonempty train/validation/test sets')
        partitions = {'validation': paths[:n_val], 'test': paths[n_val:n_val + n_test],
                      'train': paths[n_val + n_test:]}
        for key, values in partitions.items():
            split[key].extend(values)
        split['counts'][name] = {key: len(values) for key, values in partitions.items()}
    split['counts']['total'] = {key: len(split[key]) for key in ('train', 'validation', 'test')}
    return split


def resize_photograph(image, image_size):
    """EXIF-correct RGB; scale longest side to image_size without stretching/cropping."""
    image = ImageOps.exif_transpose(image).convert('RGB')
    scale = image_size / max(image.size)
    width, height = (max(1, round(dimension * scale)) for dimension in image.size)
    return TF.resize(image, [height, width], interpolation=TF.InterpolationMode.BILINEAR)


def pad_to_square(tensor, value=0.0):
    """Temporary bottom/right batch padding; never rescale image content."""
    height, width = tensor.shape[-2:]
    side = max(height, width)
    return F.pad(tensor, (0, side - width, 0, side - height), value=value)


def collate_patches(samples):
    """Stack variable-aspect inputs; masks remain zero in temporary padding."""
    return default_collate([{**sample, 'image': pad_to_square(sample['image']),
                             'mask': pad_to_square(sample['mask'])} for sample in samples])


class PatchDataset(Dataset):
    """RGB resize, random epoch-dependent training masks, fixed val/test masks."""
    def __init__(self, paths, image_size, mask_sizes, seed, training=False, masks_per_image=1):
        if not isinstance(masks_per_image, int) or isinstance(masks_per_image, bool) or masks_per_image <= 0:
            raise ValueError('masks_per_image must be a positive integer')
        self.paths = list(paths)
        self.image_size = image_size
        self.mask_sizes = list(mask_sizes)
        self.seed = seed
        self.training = training
        self.masks_per_image = masks_per_image
        self.epoch = 0

    def __len__(self):
        return len(self.paths) * self.masks_per_image

    def __getitem__(self, index):
        image_index = index // self.masks_per_image
        rng = random.Random(self.seed + index * 1000003 + (self.epoch * 9176 if self.training else 0))
        with Image.open(self.paths[image_index]) as source:
            image = resize_photograph(source, self.image_size)
            tensor = TF.to_tensor(image)
        width, height = image.size
        fitting_sizes = [size for size in self.mask_sizes if size <= min(width, height)]
        if not fitting_sizes:
            raise ValueError(f'No configured mask fits {self.paths[image_index]} after resizing to {width}x{height}; '
                             'increase data.image_size or reduce mask_sizes.')
        size = rng.choice(fitting_sizes) if self.training else fitting_sizes[index % len(fitting_sizes)]
        x = rng.randrange(width - size + 1)
        y = rng.randrange(height - size + 1)
        mask = torch.zeros(1, height, width)
        mask[:, y:y + size, x:x + size] = 1
        return {'image': tensor, 'mask': mask, 'mask_size': size, 'index': index,
                'height': height, 'width': width}


def build_dataset(config, split_name, *, splits=None, training_seed=None, masks_per_image=1):
    """Shared train/eval entry point; validation/test ignore training_seed."""
    if split_name not in ('train', 'validation', 'test'):
        raise ValueError('split_name must be train, validation or test')
    splits = prepare_splits(config) if splits is None else splits
    training = split_name == 'train'
    seed = (config['seed'] if training_seed is None else training_seed) if training else config['seed'] + (
        10000 if split_name == 'validation' else 20000)
    return PatchDataset(splits[split_name], config['image_size'], config['mask_sizes'], seed, training,
                        masks_per_image=masks_per_image)


def fixed_noise_time(batch, seed, device):
    """Item-seeded CPU noise/time, independent of loader batches and device."""
    noises, times = [], []
    for image, index in zip(batch['image'], batch['index'].tolist()):
        generator = torch.Generator().manual_seed(seed + index * 1000033)
        noises.append(torch.randn(image.shape, generator=generator))
        times.append(torch.rand((), generator=generator))
    return torch.stack(noises).to(device), torch.stack(times).to(device)
