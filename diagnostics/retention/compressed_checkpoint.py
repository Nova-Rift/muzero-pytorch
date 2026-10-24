"""Lossless transport encoding for immutable checkpoints; never deserializes data."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

CHUNK = 1024 * 1024


def verify(path, expected_sha256, expected_bytes):
    digest = hashlib.sha256()
    count = 0
    with gzip.open(path, 'rb') as source:
        while chunk := source.read(CHUNK):
            count += len(chunk)
            if count > expected_bytes:
                raise ValueError('Decoded checkpoint exceeds expected size')
            digest.update(chunk)
    if count != expected_bytes or digest.hexdigest() != expected_sha256:
        raise ValueError('Decoded checkpoint does not match expected content')
    return {'raw_bytes': count, 'raw_sha256': digest.hexdigest()}


def restore(source, destination, expected_sha256, expected_bytes):
    """Publish raw bytes only after bounded decoding and exact digest validation."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    fd, name = tempfile.mkstemp(prefix='.restore-', dir=destination.parent)
    try:
        digest, count = hashlib.sha256(), 0
        with os.fdopen(fd, 'wb') as raw, gzip.open(source, 'rb') as incoming:
            while chunk := incoming.read(CHUNK):
                count += len(chunk)
                if count > expected_bytes:
                    raise ValueError('Decoded checkpoint exceeds expected size')
                digest.update(chunk)
                raw.write(chunk)
            raw.flush()
            os.fsync(raw.fileno())
        if count != expected_bytes or digest.hexdigest() != expected_sha256:
            raise ValueError('Decoded checkpoint does not match expected content')
        os.link(name, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(name).unlink(missing_ok=True)


def compress(source, destination, expected_sha256, level=1):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    started = time.monotonic()
    digest = hashlib.sha256()
    count = 0
    fd, name = tempfile.mkstemp(prefix='.checkpoint-', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', compresslevel=level,
                               fileobj=raw, mtime=0) as encoded:
                with source.open('rb') as incoming:
                    while chunk := incoming.read(CHUNK):
                        digest.update(chunk)
                        count += len(chunk)
                        encoded.write(chunk)
            raw.flush()
            os.fsync(raw.fileno())
        if digest.hexdigest() != expected_sha256:
            raise ValueError('Input differs from registered checkpoint hash')
        compression_seconds = time.monotonic() - started
        verified = verify(name, expected_sha256, count)
        encoded_hash = hashlib.sha256()
        with open(name, 'rb') as incoming:
            while chunk := incoming.read(CHUNK):
                encoded_hash.update(chunk)
        os.link(name, destination)  # Atomic and refuses concurrent replacement.
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return dict(verified, compressed_bytes=destination.stat().st_size,
                    compressed_sha256=encoded_hash.hexdigest(), level=level,
                    compression_seconds=compression_seconds,
                    total_seconds=time.monotonic() - started,
                    scope='Exact byte recovery only; no numerical resume or strength claim')
    finally:
        Path(name).unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    print(json.dumps(compress(args.source, args.destination, args.sha256), indent=2))
