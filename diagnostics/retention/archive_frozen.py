"""Archive an explicit verified frozen-file manifest with bounded transient space.

Compatible with research-history schema1. Only the coordinator may admit frozen
closed outputs. This does not establish writer ownership or authorize GPU work.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

SKIP = re.compile(r'(^\.git$|^\.env|credential|api.key|token|secret|^id_rsa|^id_ed25519|^ssh-key|^known.hosts$|^node_modules$|^__pycache__$)', re.I)


def identity(s):
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def regular(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Regular nonsymlink file required')
    return path.stat()


def hash_stable(path, cache, counters):
    before = regular(path)
    key = identity(before)
    if key not in cache:
        h = hashlib.sha256()
        with path.open('rb') as f:
            while block := f.read(1024 * 1024):
                h.update(block)
                counters['hashed_bytes'] += len(block)
        if identity(regular(path)) != key:
            raise ValueError('File changed while hashing')
        cache[key] = h.hexdigest()
    return cache[key], before


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def snapshot_frozen(label, manifest, archive):
    if not re.fullmatch(r'[\w.-]{1,120}', label):
        raise ValueError('Short research ID required')
    archive = Path(archive).resolve()
    objects, records = archive / 'objects', archive / 'records'
    objects.mkdir(parents=True, exist_ok=True, mode=0o700)
    records.mkdir(exist_ok=True, mode=0o700)
    entries = manifest['files']
    if manifest.get('frozen_closed_outputs') is not True:
        raise ValueError('Explicit frozen closed-output admission required')
    # Validate every path/expected digest before any source mutation.
    paths = set()
    for entry in entries:
        p = Path(entry['path']).absolute()
        if p in paths:
            raise ValueError('Duplicate source path')
        paths.add(p)
        s = regular(p)
        if p.resolve().is_relative_to(archive) or any(SKIP.search(x) for x in p.parts):
            raise ValueError('Archive or sensitive path rejected')
        if s.st_mode & 0o111:
            raise ValueError('Executable artifact rejected; archive separately preserving mode')
        if s.st_dev != objects.stat().st_dev:
            raise ValueError('Same-filesystem path-preserving linking required')
        if not re.fullmatch('[0-9a-f]{64}', entry['sha256']) or s.st_size != entry['bytes']:
            raise ValueError('Expected digest and size required')
        if s.st_nlink != 1 and not (objects / entry['sha256']).exists():
            raise ValueError('Unarchived multiply-linked source has unbounded retained-copy space')
    cache, files = {}, []
    counters = dict(hashed_bytes=0, copied_bytes=0, peak_temporary_bytes=0,
                    reclaimed_allocated_bytes=0, already_shared=0)
    for entry in entries:
        p = Path(entry['path']).absolute()
        expected = entry['sha256']
        obj = objects / expected
        before = regular(p)
        original = dict(mtime_ns=before.st_mtime_ns, mode=stat.S_IMODE(before.st_mode))
        if obj.exists():
            actual, archived = hash_stable(obj, cache, counters)
            if actual != expected or archived.st_size != entry['bytes']:
                raise ValueError('Existing archive object corrupt; never overwrite')
            actual, before = hash_stable(p, cache, counters)
            if actual != expected:
                raise ValueError('Source differs from expected digest')
        else:
            # One source pass, one temporary file at a time. Prior files have
            # already been linked, so peak extra disk is at most the largest file.
            fd, name = tempfile.mkstemp(prefix='.frozen-', dir=objects)
            tmp = Path(name)
            try:
                h = hashlib.sha256()
                with os.fdopen(fd, 'wb') as out, p.open('rb') as inp:
                    while block := inp.read(1024 * 1024):
                        h.update(block)
                        out.write(block)
                        counters['hashed_bytes'] += len(block)
                        counters['copied_bytes'] += len(block)
                    out.flush()
                    os.fsync(out.fileno())
                counters['peak_temporary_bytes'] = max(counters['peak_temporary_bytes'], tmp.stat().st_size)
                if identity(regular(p)) != identity(before) or h.hexdigest() != expected:
                    raise ValueError('Source changed or differs from expected digest')
                tmp.chmod(0o400)
                try:
                    os.link(tmp, obj)
                except FileExistsError:
                    actual, _ = hash_stable(obj, cache, counters)
                    if actual != expected:
                        raise ValueError('Concurrent archive object differs')
                sync_dir(objects)
            finally:
                tmp.unlink(missing_ok=True)
            archived = regular(obj)
            cache[identity(archived)] = expected
        if identity(regular(p)) != identity(before):
            raise ValueError('Source changed before replacement')
        if (before.st_dev, before.st_ino) == (archived.st_dev, archived.st_ino):
            counters['already_shared'] += 1
        else:
            obj.chmod(stat.S_IMODE(archived.st_mode) & ~0o222)
            # Changing mode changes ctime; carry verified hash to the new exact
            # identity only after this own operation and recheck digest identity.
            cache[identity(regular(obj))] = expected
            fd, name = tempfile.mkstemp(prefix='.archive-link-', dir=p.parent)
            os.close(fd)
            Path(name).unlink()
            try:
                os.link(obj, name)
                if identity(regular(p)) != identity(before):
                    raise ValueError('Source changed before atomic replacement')
                os.replace(name, p)
                sync_dir(p.parent)
            finally:
                Path(name).unlink(missing_ok=True)
            if before.st_nlink == 1:
                counters['reclaimed_allocated_bytes'] += before.st_blocks * 512
        # Our hardlink operation changes ctime/nlink without changing bytes.
        # Cache only this post-operation identity, never reuse across invocations.
        cache[identity(regular(obj))] = expected
        files.append(dict(path=str(p), sha256=expected, bytes=entry['bytes'], original=original))
    data = dict(schema=1, id=label, files=files, omitted=[])
    revision = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    record = records / f'{label}-{revision}.json'
    if not record.exists():
        data['archived_at'] = datetime.now(timezone.utc).isoformat()
        fd, name = tempfile.mkstemp(prefix='.record-', dir=records)
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2)
                f.write('\n')
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(name, record)
            except FileExistsError:
                pass
            sync_dir(records)
        finally:
            Path(name).unlink(missing_ok=True)
    return dict(record=str(record), files=len(files), **counters)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--id', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(snapshot_frozen(a.id, json.loads(a.manifest.read_text()), a.archive)))


if __name__ == '__main__':
    main()
