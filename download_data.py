"""Check configured dataset inventories and download only missing files.

Usage: python download_data.py --config config/data/default.yaml
Download manifests are separate from training/evaluation split rules.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import re
import shutil
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from tqdm.auto import tqdm

from data import DEFAULT_DATA_CONFIG, IMAGE_EXTENSIONS, load_data_config, resolve_path


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def _open(url, options):
    if urlparse(url).scheme not in ('http', 'https'):
        raise ValueError('Download addresses must use http or https')
    request = Request(url, headers={'User-Agent': options['user_agent']})
    return urlopen(request, timeout=options['timeout'])


def _retry(operation, options):
    for attempt in range(options['retries']):
        try:
            return operation()
        except (OSError, HTTPError, URLError, ValueError) as exc:
            if attempt + 1 == options['retries']:
                raise
            delay = min(2 ** attempt, 30)
            if isinstance(exc, HTTPError) and exc.headers.get('Retry-After', '').isdigit():
                delay = min(int(exc.headers['Retry-After']), 60)
            time.sleep(delay)


def _filename(value):
    # A remote manifest must never write outside the selected dataset directory.
    if not value or value in ('.', '..') or '/' in value or '\\' in value or Path(value).is_absolute():
        raise ValueError(f'Unsafe download filename: {value!r}')
    if Path(value).suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f'Unsupported image extension: {value}')
    return value


def _commons_inventory(directory, source, options):
    """Reuse the original source CSV when present; otherwise query the category."""
    old_csv = directory / 'sources.csv'
    if old_csv.exists():
        with old_csv.open(encoding='utf-8-sig', newline='') as stream:
            return [{'filename': row['filename'], 'url': row['download_url'] or row['original_url'],
                     'source_page': row['source_page'], 'license': row.get('license', ''),
                     'license_url': row.get('license_url', ''), 'artist': row.get('artist', ''),
                     'original_download_date': row.get('download_date', '')}
                    for row in csv.DictReader(stream)]
    recipe = source['download']
    params = {'action': 'query', 'format': 'json', 'formatversion': 2,
              'generator': 'categorymembers', 'gcmtitle': recipe['category'],
              'gcmtype': 'file', 'gcmlimit': 50, 'prop': 'imageinfo',
              'iiprop': 'url|mime|extmetadata', 'iiurlwidth': recipe['thumbnail_width'], 'maxlag': 5}
    records = []
    while True:
        def fetch():
            with _open(recipe['api_url'] + '?' + urlencode(params), options) as response:
                payload = json.load(response)
            if 'error' in payload:
                raise ValueError(f'Wikimedia API error: {payload["error"]}')
            return payload
        payload = _retry(fetch, options)
        for page in payload.get('query', {}).get('pages', []):
            info = page.get('imageinfo', [{}])[0]
            title = page['title']
            extension = Path(title).suffix.lower()
            if extension not in IMAGE_EXTENSIONS or not info.get('mime', '').startswith('image/'):
                continue
            # Stable names for fresh downloads; existing CSV names stay unchanged.
            stem = re.sub(r'[\\/:*?"<>|]', '_', title.removeprefix('File:'))
            filename = f'{Path(stem).stem[:100]}_{hashlib.sha1(title.encode()).hexdigest()[:10]}{extension}'
            metadata = info.get('extmetadata', {})
            records.append({'filename': filename, 'url': info.get('thumburl', info['url']),
                            'original_url': info['url'], 'source_page': info.get('descriptionurl', ''),
                            'license': metadata.get('LicenseShortName', {}).get('value', ''),
                            'license_url': metadata.get('LicenseUrl', {}).get('value', ''),
                            'artist': metadata.get('Artist', {}).get('value', '')})
        if 'continue' not in payload:
            break
        params.update(payload['continue'])
    records.sort(key=lambda record: record['filename'])
    return records[:recipe['expected_count']]


def _openimages_inventory(directory, source, options):
    recipe = source['download']
    expected = recipe['expected_count']
    existing = sorted({path.stem.lower() for path in directory.glob('*')
                       if path.is_file() and path.suffix.lower() == '.jpg'
                       and re.fullmatch(r'[0-9a-fA-F]{16}', path.stem)})
    if len(existing) > expected:
        raise ValueError('More existing Open Images IDs than expected_count; review the configured subset')
    selected = set(existing)
    if len(selected) < expected:
        # Bootstrap only. Persist the final IDs BEFORE downloading any images.
        # Existing IDs are retained; the remaining subset is seeded, not class-filtered.
        print('No complete saved inventory: preserving existing IDs and selecting additional official IDs.', flush=True)
        needed = expected - len(selected)
        def select():
            rng, reservoir, seen = random.Random(recipe['selection_seed']), [], 0
            with _open(recipe['image_ids_url'], options) as response:
                for row in csv.DictReader(io.TextIOWrapper(response, encoding='utf-8-sig', newline='')):
                    image_id = row['ImageID'].lower()
                    if row.get('Subset', recipe['source_split']) != recipe['source_split']:
                        continue
                    if image_id in selected or not re.fullmatch(r'[0-9a-f]{16}', image_id):
                        continue
                    seen += 1
                    if len(reservoir) < needed:
                        reservoir.append(image_id)
                    else:
                        index = rng.randrange(seen)
                        if index < needed:
                            reservoir[index] = image_id
            if len(reservoir) != needed:
                raise ValueError('Official ID index has too few eligible images')
            return reservoir
        selected.update(_retry(select, options))
    return [{'filename': image_id + '.jpg', 'image_id': image_id,
             'url': recipe['image_url_template'].format(split=recipe['source_split'], image_id=image_id)}
            for image_id in sorted(selected)]


def _inventory(directory, source, options):
    recipe = source['download']
    manifest_path = resolve_path(recipe['manifest'])
    specification = {'source_url': source['source_url'], 'download': recipe}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['specification'] != specification:
            raise ValueError(f'Download recipe changed: review {manifest_path}; do not silently replace its subset')
    else:
        if recipe['provider'] == 'wikimedia':
            images = _commons_inventory(directory, source, options)
        elif recipe['provider'] == 'openimages':
            images = _openimages_inventory(directory, source, options)
        else:
            raise ValueError(f'Unknown download provider: {recipe["provider"]}')
        manifest = {'created_at': _now(), 'specification': specification, 'images': images}
    records = manifest['images']
    filenames = [_filename(record['filename']) for record in records]
    if len(records) != recipe['expected_count'] or len(set(filenames)) != len(records):
        raise ValueError(f'Inventory does not contain exactly {recipe["expected_count"]} distinct images')
    for record in records:
        if urlparse(record['url']).scheme not in ('http', 'https'):
            raise ValueError('Invalid image URL in download inventory')
    if not manifest_path.exists():
        _write_json(manifest_path, manifest)
    return records


def _download_one(record, directory, options):
    path = directory / _filename(record['filename'])
    if path.is_file():
        return {'filename': path.name, 'status': 'existing'}
    if path.exists() or path.is_symlink():
        raise ValueError(f'Download target exists but is not a regular file: {path}')
    def transfer():
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, suffix='.part', delete=False) as temporary:
                temporary_path = Path(temporary.name)
                with _open(record['url'], options) as response:
                    shutil.copyfileobj(response, temporary)
            # Existence only: do not decode images or inspect file contents/size.
            if path.exists():
                if path.is_file():
                    return {'filename': path.name, 'status': 'existing'}
                raise ValueError(f'Download target is not a regular file: {path}')
            temporary_path.replace(path)
            return {'filename': path.name, 'status': 'downloaded', 'url': record['url'],
                    'downloaded_at': _now(), 'bytes': path.stat().st_size}
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return _retry(transfer, options)


def ensure_datasets(config):
    """Check exact recorded filenames, then download missing files only."""
    options = config['download']
    if min(options['timeout'], options['retries'], options['workers']) <= 0:
        raise ValueError('download timeout, retries and workers must be positive')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    report_dir = resolve_path(options['output_root']) / stamp
    report, failures = {'started_at': _now(), 'datasets': {}}, []
    for name, source in config['datasets'].items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('Download dataset names must contain only letters, digits, _ or -')
        directory = resolve_path(source['path'])
        directory.mkdir(parents=True, exist_ok=True)
        try:
            records = _inventory(directory, source, options)
            results = []
            def process(record):
                try:
                    return _download_one(record, directory, options)
                except Exception as exc:
                    return {'filename': record['filename'], 'status': 'failed', 'error': str(exc)}
            # Wikimedia is intentionally serial to avoid overloading the service.
            workers = 1 if source['download']['provider'] == 'wikimedia' else options['workers']
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for result in tqdm(pool.map(process, records), total=len(records), desc=name):
                    results.append(result)
            failed = sum(result['status'] == 'failed' for result in results)
            summary = {'expected': len(records), 'existing': sum(r['status'] == 'existing' for r in results),
                       'downloaded': sum(r['status'] == 'downloaded' for r in results),
                       'failed': failed, 'complete': failed == 0, 'files': results}
            if failed:
                failures.append(name)
            print(f'{name}: existing={summary["existing"]}, downloaded={summary["downloaded"]}, failed={failed}', flush=True)
            report['datasets'][name] = summary
        except Exception as exc:
            failures.append(name)
            report['datasets'][name] = {'complete': False, 'error': str(exc)}
            print(f'{name}: {exc}', flush=True)
        _write_json(report_dir / 'report.json', report)
    report['finished_at'] = _now()
    _write_json(report_dir / 'report.json', report)
    if failures:
        raise RuntimeError(f'Incomplete datasets: {", ".join(failures)}; see {report_dir / "report.json"}')
    return {'report_dir': str(report_dir), **report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_DATA_CONFIG, help='Shared dataset YAML/JSON configuration')
    args = parser.parse_args()
    result = ensure_datasets(load_data_config(args.config))
    print(f'All configured images are available. Report: {result["report_dir"]}')


if __name__ == '__main__':
    main()
