"""Create 25 stored 7z release assets without changing the trained models."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


GIB = 1024 ** 3
RELEASE_LIMIT = 2 * GIB


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def snapshot(path):
    info = path.stat()
    return {'bytes': info.st_size, 'mtime_ns': info.st_mtime_ns,
            'file_id': str(info.st_ino)}


def build_plan(models_root):
    plan = []
    for branch, parent, directory in (
        ('ESM2', 'ESM2_BiGating', True),
        ('ProtT5', 'ProtT5_Fusion', True),
        ('SaProt', 'Saprot_DAGAT', False),
    ):
        for fold in range(1, 6):
            name = f'{branch}_branch_fold_{fold}'
            source = models_root / parent / (name if directory else name + '.pth')
            if directory:
                if not source.is_dir():
                    raise FileNotFoundError(source)
                required = 'model.safetensors' if branch == 'ESM2' else 'pytorch_model.bin'
                if not (source / required).is_file():
                    raise FileNotFoundError(source / required)
                files = sorted(p for p in source.rglob('*') if p.is_file())
            else:
                if not source.is_file():
                    raise FileNotFoundError(source)
                files = [source]
            if not files or any(p.is_symlink() for p in files):
                raise ValueError(f'Empty model or symbolic link: {source}')
            members = {p.relative_to(models_root.parent).as_posix(): snapshot(p) for p in files}
            plan.append({'branch': branch, 'fold': fold, 'archive': name + '.7z',
                         'source': source.relative_to(models_root.parent).as_posix(),
                         'source_bytes': sum(v['bytes'] for v in members.values()),
                         'members': members, 'expected_assets': 3 if branch == 'ProtT5' else 1,
                         'status': 'pending'})
    return plan


def run_7z(executable, arguments, cwd, log_path):
    command = [str(executable), *arguments, '-sccUTF-8', '-bd', '-bsp0']
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                            encoding='utf-8', errors='replace')
    with log_path.open('a', encoding='utf-8') as handle:
        handle.write('\nCOMMAND: ' + subprocess.list2cmdline(command) + '\n')
        handle.write(result.stdout + result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f'7-Zip returned {result.returncode}; see {log_path}\n'
                           + (result.stdout + result.stderr)[-3000:])
    return result.stdout


def list_members(text):
    members = {}
    for block in text.strip().split('\n\n'):
        fields = {}
        for line in block.splitlines():
            key, separator, value = line.partition(' = ')
            if separator:
                fields[key] = value
        if 'Path' not in fields or 'Size' not in fields:
            continue
        if 'D' in fields.get('Attributes', '') or fields.get('Folder') == '+':
            continue
        members[fields['Path'].replace('\\', '/')] = {
            'bytes': int(fields['Size']), 'crc32': fields.get('CRC'),
            'method': fields.get('Method'),
        }
    return members


def write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-root', type=Path,
                        default=Path(__file__).resolve().parents[3] / 'trained_models')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seven-zip', type=Path, required=True)
    parser.add_argument('--volume-mib', type=int, default=1800)
    args = parser.parse_args()
    models_root = args.models_root.resolve()
    output = args.output_dir.resolve()
    seven_zip = args.seven_zip.resolve()
    volume_bytes = args.volume_mib * 1024 ** 2
    if not seven_zip.is_file():
        raise FileNotFoundError(seven_zip)
    if not 0 < volume_bytes < RELEASE_LIMIT:
        raise ValueError('Volume size must be positive and less than 2 GiB.')
    if output == models_root or output.is_relative_to(models_root):
        raise ValueError('Output must be outside the source model directory.')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to modify a nonempty output directory: {output}')
    plan = build_plan(models_root)
    total_bytes = sum(item['source_bytes'] for item in plan)
    space_parent = output
    while not space_parent.exists():
        space_parent = space_parent.parent
    if shutil.disk_usage(space_parent).free < total_bytes + 256 * 1024 ** 2:
        raise OSError('Insufficient output disk space for stored archives and safety margin.')
    output.mkdir(parents=True, exist_ok=True)
    assets_dir = output / 'assets'
    assets_dir.mkdir()
    manifest_path = output / 'packaging_manifest.json'
    log_path = output / 'PACKAGING_LOG.txt'
    manifest = {'status': 'running', 'mode': '7z Copy (-mx=0)',
                'source_models_root': str(models_root), 'output_dir': str(output),
                'volume_bytes': volume_bytes, 'release_asset_limit_bytes': RELEASE_LIMIT,
                'source_bytes': total_bytes, 'expected_model_asset_count': 25,
                'archives': plan}
    write_manifest(manifest_path, manifest)
    started = time.monotonic()
    print(f'Starting 15 archives / 25 model assets; source {total_bytes / GIB:.2f} GiB.', flush=True)
    all_assets = []
    try:
        for index, item in enumerate(plan, start=1):
            archive = assets_dir / item['archive']
            print(f'[{index}/15] Creating {archive.name} ({item["source_bytes"] / GIB:.2f} GiB).', flush=True)
            item['status'] = 'creating'
            write_manifest(manifest_path, manifest)
            switches = ['a', '-t7z', '-mx=0', '-y']
            if item['branch'] == 'ProtT5':
                switches.append(f'-v{volume_bytes}b')
            run_7z(seven_zip, [*switches, str(archive), item['source']],
                   models_root.parent, log_path)
            if item['branch'] == 'ProtT5':
                parts = sorted(assets_dir.glob(archive.name + '.[0-9][0-9][0-9]'))
                expected_names = [archive.name + f'.{part:03d}' for part in range(1, 4)]
                if [p.name for p in parts] != expected_names:
                    raise ValueError(f'Expected three volumes for {archive.name}; got {[p.name for p in parts]}')
            else:
                parts = [archive]
            if any(not p.is_file() or p.stat().st_size >= RELEASE_LIMIT for p in parts):
                raise ValueError(f'Missing archive or asset reaches the 2 GiB limit: {archive.name}')
            item['status'] = 'testing'
            write_manifest(manifest_path, manifest)
            print(f'[{index}/15] Testing {parts[0].name}.', flush=True)
            run_7z(seven_zip, ['t', str(parts[0])], models_root.parent, log_path)
            listed = run_7z(seven_zip, ['l', '-slt', '-ba', str(parts[0])],
                            models_root.parent, log_path)
            members = list_members(listed)
            expected = {name: info['bytes'] for name, info in item['members'].items()}
            if {name: info['bytes'] for name, info in members.items()} != expected:
                raise ValueError(f'Archive paths or member sizes differ: {archive.name}')
            if any(info['method'] != 'Copy' for info in members.values()):
                raise ValueError(f'Archive does not use storage mode: {archive.name}')
            for name, info in item['members'].items():
                if snapshot(models_root.parent / name) != info:
                    raise ValueError(f'Source model changed during packaging: {name}')
            print(f'[{index}/15] Computing SHA256 for {len(parts)} asset(s).', flush=True)
            item['assets'] = [{'name': p.name, 'bytes': p.stat().st_size,
                               'sha256': file_hash(p)} for p in parts]
            item['verified_members'] = members
            item['status'] = 'verified'
            all_assets.extend(item['assets'])
            write_manifest(manifest_path, manifest)
            print(f'[{index}/15] Verified {archive.name}; {len(all_assets)}/25 assets complete.', flush=True)
        actual_names = {p.name for p in assets_dir.iterdir() if p.is_file()}
        if len(all_assets) != 25 or actual_names != {a['name'] for a in all_assets}:
            raise ValueError('Final asset count or inventory is not exactly 25.')
        (output / 'SHA256SUMS.txt').write_text(''.join(
            f'{asset["sha256"]}  {asset["name"]}\n' for asset in all_assets), encoding='utf-8')
        (output / 'DOWNLOAD_MODELS.md').write_text(
            '# SSG-Kla model downloads\n\n'
            'The assets directory contains exactly 25 model attachments: five ESM2 archives, '
            '15 ProtT5 volumes, and five SaProt archives. All use uncompressed 7z storage.\n\n'
            'Download all volumes of each ProtT5 fold into the same directory and start '
            'extraction from its .7z.001 file. Extract each ESM2/SaProt .7z archive directly. '
            'Use the same destination for every archive: the parent of the SSG-Kla code folder. '
            'This restores trained_models/ESM2_BiGating, trained_models/ProtT5_Fusion, and '
            'trained_models/Saprot_DAGAT, including the existing fold model names.\n\n'
            'Check downloaded attachments against SHA256SUMS.txt. There is no need to modify '
            'the model loading code after restoring this layout.\n', encoding='utf-8')
        manifest['status'] = 'complete'
        manifest['asset_count'] = len(all_assets)
        manifest['asset_bytes'] = sum(a['bytes'] for a in all_assets)
        manifest['elapsed_seconds'] = round(time.monotonic() - started, 2)
        write_manifest(manifest_path, manifest)
        print(f'COMPLETE: 25 verified model assets in {assets_dir}', flush=True)
    except BaseException as error:
        manifest['status'] = 'failed'
        manifest['error'] = f'{type(error).__name__}: {error}'
        write_manifest(manifest_path, manifest)
        raise


if __name__ == '__main__':
    main()
