"""Runtime-gate entrypoint with explicit serialized root5 loss contract."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arena.train_loss import main

if __name__ == '__main__':
    if any(x == '--loss-contract' or x.startswith('--loss-contract=') for x in sys.argv[1:]):
        raise ValueError('This entrypoint fixes root5-mass-v1')
    sys.argv.extend(['--loss-contract', 'root5-mass-v1'])
    main()
