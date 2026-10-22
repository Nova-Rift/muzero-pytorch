"""Immutable full-state milestones with an atomically refreshed latest alias."""
import os
import tempfile
from pathlib import Path


def save_milestone(state, output, save):
    """Single-writer output directory; save is the checkpoint serializer.

    Never overwrite a milestone. A failed alias update may leave a valid new
    milestone, but never removes the old latest state. Recovery must copy the
    entire full-state directory, not just the convenience alias.
    """
    output = Path(output)
    history = output / 'full-state'
    history.mkdir(exist_ok=True)
    games = state['totals']['games']
    updates = state['totals']['optimizer_updates']
    if type(games) is not int or type(updates) is not int or min(games, updates) < 0:
        raise ValueError('Invalid completed counters')
    destination = history / f'games-{games}-updates-{updates}.pt'
    fd, name = tempfile.mkstemp(prefix='.saving-', suffix='.tmp', dir=history)
    os.close(fd)
    temporary = Path(name)
    alias = output / '.latest-state.tmp'
    try:
        save(state, temporary)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.link(temporary, destination)  # exclusive publication, no replacement
        os.link(destination, alias)
        os.replace(alias, output / 'resume-state.pt')
        for directory in (history, output):
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return destination
    finally:
        temporary.unlink(missing_ok=True)
        alias.unlink(missing_ok=True)
