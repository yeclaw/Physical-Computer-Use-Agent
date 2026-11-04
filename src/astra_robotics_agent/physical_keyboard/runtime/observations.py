"""Immutable camera snapshot validation and inline transfer decoding."""
from pathlib import Path, PurePosixPath
import base64
from astra_robotics_agent.physical_keyboard.runtime.remote_host import digest

VIEWS = ("head", "left_wrist", "right_wrist")

def validate_snapshot(manifest, run_id, remote_run):
    if (not manifest.get('complete') or manifest.get('run_id') != run_id
            or set(manifest.get('views', {})) != set(VIEWS)):
        raise ValueError('Incomplete/cross-run camera evidence')
    root = PurePosixPath(manifest['output'])
    if not root.is_relative_to(PurePosixPath(remote_run) / 'snapshots'):
        raise ValueError('Snapshot path outside remote host run')
    for view in VIEWS:
        if (manifest['views'][view]['copied_image'] != view + '.jpg'
                or manifest['files'][view + '.jpg']['sha256'] != manifest['views'][view]['source']['sha256']):
            raise ValueError('Camera identity/hash mismatch')
    return root

def save_inline_snapshot_images(bundle, manifest, folder):
    """Verify all fixed-view bytes before writing; no file transfer subprocess."""
    if (type(bundle) is not dict or set(bundle) != {'manifest', 'images_base64'}
            or bundle['manifest'] != manifest
            or type(bundle['images_base64']) is not dict
            or set(bundle['images_base64']) != set(VIEWS)):
        raise ValueError('Malformed inline snapshot bundle')
    maximum = 4 * 1024 * 1024
    verified = {}
    for view in VIEWS:
        value = bundle['images_base64'][view]
        if type(value) is not str or len(value) > 4 * ((maximum + 2) // 3):
            raise ValueError('Inline image exceeds size limit or is not base64 text')
        try:
            raw = base64.b64decode(value, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError('Invalid inline image base64') from exc
        if len(raw) > maximum or digest(raw) != manifest['files'][view + '.jpg']['sha256']:
            raise ValueError('Inline image size/source SHA mismatch')
        if (folder / (view + '.jpg')).exists():
            raise ValueError('Refusing to replace an existing observation image')
        verified[view] = raw
    for view, raw in verified.items():
        with (folder / (view + '.jpg')).open('xb') as stream:
            stream.write(raw)
