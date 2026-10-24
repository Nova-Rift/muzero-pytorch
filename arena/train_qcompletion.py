"""New experiment: reference unvisited Q completion; unchanged trainer/loss.

Start fresh, or resume only this exact source/config/runtime contract. Historical
checkpoints from the original search are not exact continuations of this variant.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena import train_selfplay
from batched_search_qcompletion import search_batch


def main():
    train_selfplay.search_batch = search_batch
    train_selfplay.main()


if __name__ == '__main__':
    main()
