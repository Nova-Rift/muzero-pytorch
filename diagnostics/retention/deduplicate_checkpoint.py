"""Preserve frozen checkpoint paths/bytes while sharing an immutable archive inode."""
import hashlib
import os
from pathlib import Path
import stat
import tempfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as source:
        while block:=source.read(1048576):h.update(block)
    return h.hexdigest()


def deduplicate(source, archive_object, expected_sha256):
    source,archive_object=Path(source),Path(archive_object)
    for path in (source,archive_object):
        if path.is_symlink() or not path.is_file():raise ValueError('Regular nonsymlink file required')
    before,archived=source.stat(),archive_object.stat()
    if before.st_dev!=archived.st_dev:raise ValueError('Hardlink requires same filesystem')
    if before.st_size!=archived.st_size:raise ValueError('Different sizes')
    if digest(source)!=expected_sha256 or digest(archive_object)!=expected_sha256:
        raise ValueError('Checkpoint or archive content differs')
    # Only frozen, closed outputs may be admitted by the caller. Never use on
    # mutable latest states, active writers, or historical model source files.
    # Reading may legitimately advance atime; only content/identity metadata
    # belongs in this guard.
    stable=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if stable(source.stat())!=stable(before) or stable(archive_object.stat())!=stable(archived):
        raise ValueError('File metadata changed during validation')
    if before.st_ino==archived.st_ino:
        return dict(already_shared=True,sha256=expected_sha256,bytes=before.st_size,reclaimed_allocated_bytes=0)
    os.chmod(archive_object, stat.S_IMODE(archived.st_mode) & ~0o222)
    fd,name=tempfile.mkstemp(prefix='.dedup-',dir=source.parent);os.close(fd);Path(name).unlink()
    try:
        os.link(archive_object,name)
        os.replace(name,source)
        directory=os.open(source.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:Path(name).unlink(missing_ok=True)
    assert source.stat().st_ino==archive_object.stat().st_ino
    return dict(already_shared=False,sha256=expected_sha256,bytes=before.st_size,
                reclaimed_allocated_bytes=before.st_blocks*512 if before.st_nlink==1 else 0,
                source_path_preserved=True,archive_path_preserved=True,read_only=True)
