# Independent-seed initialization gate

The source contract remains frozen to trainer `14fece1209067004f4c3936a6171bc40da55577d`, seed 73142017 and its exact source map. This is a conditional preparation tool; it neither chooses EGO/MIR nor authorizes training.

The initializer checks the template hash and complete source map/commit before model imports. It rejects cached `model` or `arena` modules from another checkout, then imports only the frozen `model.Config` and `arena.rootweighted.model_class`. No trainer main, optimizer, Pool, collector or Search is invoked.

Before producing a frozen contract it compares its explicit constructor against the AST of the reviewed trainer initialization prefix: single-thread setting, Torch seed, NumPy generator seed, TF32 setting, baseline factory, exact Config and device transfer. The source prefix and initial export statements must AST-match reviewed literals; unrelated trainer statements cannot enter this executable prefix. Both paths must produce identical tensor names/order, dtype, shape and bytes, compressed NPZ bytes, and post-construction Torch/NumPy/CUDA RNG states. The full source binding prevents an alternative trainer from passing by retaining just this prefix.

The trainer itself builds Adam and Pool before saving initial tensors. The no-update gate deliberately does not invoke either: neither changes model tensors in the reviewed source, and no search or update happens before the initial export. This prefix comparison is not a claim that the complete trainer ran. The sidecar states `actual_trainer_export_verified: false` explicitly.

Production `generate()` still requires Python 3.12.3, Torch 2.8.0+cu128 and available CUDA. It writes the unchanged schema-1 initialization receipt and contract, plus `initial-parity.json` evidence. Existing consumers and receipt fields are unchanged. CPU execution of `initialize_and_check()` is diagnostic only and cannot write an admitted initialization receipt through `generate()`.

Before admission: run the no-update CLI in the frozen CUDA runtime after storage/compute admission, retain the exact source/template/generator hashes and all output artifacts, and require both actual trainers' `initial.npz` hashes to equal the frozen contract (the existing replication pair validator already requires this). A CPU hash must never populate the contract. The real CUDA runtime gate remains unexecuted here. No strength or correctness claim about complete training follows from these checks.

Run source-bound checks with explicit read-only fixtures:

```sh
REPLICATION_FROZEN_SOURCE=/path/to/clean/14fece1-checkout \
REPLICATION_TEMPLATE=/path/to/contract-ego.template.json \
python3 -m unittest discover -s diagnostics/retention -p test_replication_initialization.py -v
```

Without those fixture variables the initializer-specific test class explicitly skips. Broader replication contract/integration tests remain independently runnable. CPU validation used Python 3.10.12/Torch 2.2.0+cu121: all 20 initial tensors, compressed bytes and RNG state matched locally; this is not frozen-runtime CUDA evidence.
