import argparse
import time
from pathlib import Path
import yaml
import numpy as np
import torch
from tqdm import tqdm

from model import BoaConstrictor, ByteDataloader, make_splits
from boa import BOA
from train import train


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def resolve_config_path(config_arg: str, experiments_root: Path = Path('experiments')) -> Path:
    """Resolve a --config argument which may be a path or an experiment name.

    Order:
      1. If the argument is an existing file path, return it.
      2. If it's a simple experiment name (no existing path), look for
         experiments/<name>/<name>.yaml and return if exists.
      3. Fallback to configs/<name>.yaml if present.
      4. Raise FileNotFoundError.
    """
    if config_arg is None:
        return None
    p = Path(config_arg)
    # Direct file path provided
    if p.exists():
        return p

    # Try experiments/<name>/<name>.yaml
    name = p.stem
    exp_cfg = experiments_root / name / f"{name}.yaml"
    if exp_cfg.exists():
        return exp_cfg

    # Try configs/<name>.yaml
    cfg_cfg = Path('configs') / f"{name}.yaml"
    if cfg_cfg.exists():
        return cfg_cfg

    raise FileNotFoundError(f"Could not resolve config argument '{config_arg}' to a config file")


def parse_args():
    p = argparse.ArgumentParser(description="Run BoaConstrictor experiments from a config file")
    p.add_argument('--config', '-c', type=Path, required=False, help='Path to YAML experiment config')
    p.add_argument('--no-progress', action='store_true', help='Disable progress bars')
    p.add_argument('--device', type=str, default="cuda", help='Torch device override (cpu|cuda)')
    p.add_argument('--precision', type=str, default="fp32", choices=['fp32','fp16', 'fp8'], help='Precision override')
    p.add_argument('--new-experiment', action='store_true', help='Create a new experiment config interactively and run it')
    p.add_argument('--train-only', action='store_true', help='Only run training')
    p.add_argument('--compress-only', action='store_true', help='Only run compression')
    p.add_argument('--decompress-only', action='store_true', help='Only run decompression')
    p.add_argument('--show-timings', action='store_true', help='Print timings for each major operation')
    p.add_argument('--verify', action='store_true', help='After decompression, verify bytes match the input file used for compression')
    p.add_argument('--model-path', type=str, default=None, help='Path to a pre-trained model .pt file (state_dict or full model). If provided, training is skipped and the model is loaded')
    return p.parse_args()


def main():
    args = parse_args()

    # If user requests a new experiment, run interactive creator and obtain a config path
    if args.new_experiment:
        def _prompt(prompt, default=None, cast=str):
            if default is None:
                resp = input(f"{prompt}: ").strip()
            else:
                resp = input(f"{prompt} [{default}]: ").strip()
                if resp == "":
                    resp = str(default)
            try:
                return cast(resp)
            except Exception:
                return resp

        print("Creating a new experiment config interactively. Press enter to accept the default shown in brackets.")
        name = _prompt("Experiment name", "example_experiment")
        file_path = _prompt("Path to dataset file (binary)", "/path/to/dataset.bin")
        progress = _prompt("Show progress bars (true/false)", "true", lambda s: s.lower() in ("1","true","yes"))
        device = _prompt("Device (cpu|cuda)", "cuda")
        precision = _prompt("Precision (fp32|fp16|fp8)", "fp32")
        seq_len = _prompt("Sequence length (seq_len)", 32768, int)
        batch_size = _prompt("Batch size", 3, int)
        d_model = _prompt("Model d_model", 256, int)
        num_layers = _prompt("Model num_layers", 2, int)
        lr = _prompt("Learning rate", 5e-4, float)
        epochs = _prompt("Epochs", 10, int)
        chunks_count = _prompt("Compression chunks_count", 1000, int)
        compress_file = _prompt("File to compress (leave blank to use dataset file)", "", lambda s: s if s != "" else "")
        splits_in = _prompt("Data splits as comma-separated (train,val,test)", "0.8,0.1,0.1")
        try:
            splits = [float(x.strip()) for x in splits_in.split(',')]
            if len(splits) != 3 or abs(sum(splits) - 1.0) > 1e-6:
                print("Warning: splits do not sum to 1. Using default [0.8,0.1,0.1].")
                splits = [0.8, 0.1, 0.1]
        except Exception:
            splits = [0.8, 0.1, 0.1]

        cfg = {
            'name': name,
            'file_path': file_path,
            'progress': bool(progress),
            'device': device,
            'precision': precision,
            'dataloader': {'seq_len': int(seq_len), 'batch_size': int(batch_size)},
            'model': {'d_model': int(d_model), 'num_layers': int(num_layers)},
            'training': {'lr': float(lr), 'epochs': int(epochs)},
            'compression': {'chunks_count': int(chunks_count), 'file_to_compress': compress_file},
            'splits': splits
        }

        # Decide where to save the config: store it under experiments/<name>/<name>.yaml
        cfg_path = Path('experiments') / name / f"{name}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, 'w') as f:
            yaml.safe_dump(cfg, f)
        print(f"Wrote new experiment config to: {cfg_path}")

        # Use the newly created config for the rest of the run
        args.config = str(cfg_path)

    if args.config is None:
        raise ValueError('Either --config must be provided or use --new-experiment to create one interactively')

    # Resolve the config argument: allow passing an experiment name which maps
    # to experiments/<name>/<name>.yaml, or a direct path.
    args.config = resolve_config_path(str(args.config))
    config = load_config(args.config)

    # Apply CLI overrides
    progress = not args.no_progress and config.get('progress', True)
    device =  config.get('device', 'cuda' if torch.cuda.is_available() else 'cuda') or args.device
    
    print(device)
    precision = args.precision or config.get('precision', 'fp32')
    verify = args.verify or bool(config.get('verify', False))
    # Model path can be provided via CLI or config (either top-level 'model_path' or under 'model.path')
    model_path_cfg = config.get('model_path') or config.get('model', {}).get('path')
    model_path = Path(args.model_path).expanduser() if args.model_path else (Path(model_path_cfg).expanduser() if model_path_cfg else None)
    if model_path is not None:
        try:
            cfg_dir = Path(args.config).parent if args.config is not None else Path.cwd()
        except Exception:
            cfg_dir = Path.cwd()
        if not model_path.is_absolute():
            model_path = (cfg_dir / model_path).resolve()

    # Experiment parameters (with sensible defaults)
    name = config.get('name', 'experiment')
    file_path = config.get('file_path', '')
    # Resolve file_path: if it's absolute, use as-is; if relative, interpret
    # it relative to the directory of the resolved config file (so passing
    # --config <experiment_name> works and paths inside the YAML are relative
    # to that YAML file).
    if file_path:
        file_path = Path(file_path)
        try:
            cfg_dir = Path(args.config).parent if args.config is not None else Path.cwd()
        except Exception:
            cfg_dir = Path.cwd()
        if not file_path.is_absolute():
            file_path = (cfg_dir / file_path).resolve()
    seq_len = config.get('dataloader', {}).get('seq_len', 32768)
    batch_size = config.get('dataloader', {}).get('batch_size', 3)
    d_model = config.get('model', {}).get('d_model', 256)
    num_layers = config.get('model', {}).get('num_layers', 8)
    lr = float(config.get('training', {}).get('lr', 5e-4))
    num_epochs = config.get('training', {}).get('epochs', 50)

    timings = {}

    # Read file
    t0 = time.perf_counter()
    if not file_path:
        raise ValueError('file_path must be set in the config or passed via CLI')

    # file_path is already a Path (resolved above when possible)
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, 'rb') as f:
        data_bytes = f.read()

    timings['read_bytes'] = time.perf_counter() - t0
    print(f"Read {len(data_bytes)} bytes from {file_path} in {timings['read_bytes']:.2f}s")

    # Prepare experiment output directory and filenames (needed before optional training)
    experiments_root = Path(config.get('experiments_root', 'experiments'))
    exp_dir = experiments_root / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup model, dataloaders, optimizer, loss
    model = BoaConstrictor(d_model=d_model, num_layers=num_layers, device=device)

    dataloader = ByteDataloader(data_bytes, seq_len=seq_len, batch_size=batch_size, device=device)

    train_b, val_b, test_b = make_splits(data_bytes, dataloader.seq_len, dataloader.batch_size,
                                         splits=tuple(config.get('splits', (0.8, 0.1, 0.1))))

    train_loader = ByteDataloader(train_b, seq_len=dataloader.seq_len, batch_size=dataloader.batch_size, device=device)
    val_loader = ByteDataloader(val_b, seq_len=dataloader.seq_len, batch_size=dataloader.batch_size, device=device)
    test_loader = ByteDataloader(test_b, seq_len=dataloader.seq_len, batch_size=dataloader.batch_size, device=device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # If a model path is provided and exists, load it and skip training
    def _load_model_from_path(model, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        obj = torch.load(path, map_location='cpu')
        try:
            # Try state_dict first
            if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
                model.load_state_dict(obj, strict=False)
                return model
            # If whole model was saved
            if hasattr(obj, 'state_dict') and hasattr(obj, 'parameters'):
                return obj
        except Exception:
            pass
        # Fallback: if torch.save was used with state_dict under a key
        if isinstance(obj, dict) and 'state_dict' in obj:
            model.load_state_dict(obj['state_dict'], strict=False)
            return model
        raise ValueError(f"Unrecognized checkpoint format at {path}")

    # Training or loading
    if model_path is not None and Path(model_path).exists():
        print(f"Loading pre-trained model from {model_path} and skipping training")
        t_start = time.perf_counter()
        model = _load_model_from_path(model, Path(model_path))
        model = model.to(device)
        timings['load_model'] = time.perf_counter() - t_start
        print(f"Model loaded in {timings['load_model']:.2f}s")
    elif not args.compress_only and not args.decompress_only:
        print(f"Starting training on device=={device}, precision={precision}, epochs={num_epochs}")
        t_start = time.perf_counter()
        train(model, train_loader, val_loader, test_loader, optimizer, criterion,
              device=device, name=str(exp_dir / name), NUM_EPOCHS=num_epochs, PRECISION=precision, progress=progress)
        timings['training'] = time.perf_counter() - t_start
        print(f"Training complete in {timings['training']:.2f}s")

    compress_file_cfg = config.get('compression', {}).get('file_to_compress', '')
    # If blank, use the original dataset file we already loaded
    if not compress_file_cfg:
        compress_file_path = file_path
    else:
        # Resolve compress_file relative to config dir when relative
        cfp = Path(compress_file_cfg)
        cfg_dir = Path(args.config).parent if args.config is not None else Path.cwd()
        if not cfp.is_absolute():
            cfp = (cfg_dir / cfp).resolve()
        if not cfp.exists():
            raise FileNotFoundError(f"Compression input file not found: {cfp}")
        compress_file_path = cfp

    boa = BOA(device, str(exp_dir / f"{name}.boa"), model)
    file_format = compress_file_path.suffix.lstrip('.') or 'bin'
    # Compression
    if not args.train_only and not args.decompress_only:
        print("Starting compression...")
        t_start = time.perf_counter()
        # Create BOA that writes into the experiment directory
        boa.compress(
            data_path=str(compress_file_path),
            chunks_count=config.get('compression', {}).get('chunks_count', 1000),
            progress=progress,
        )
        timings['compression'] = time.perf_counter() - t_start
        print(f"Compression complete in {timings['compression']:.2f}s")

    # Decompression (write decompressed bytes into the experiment directory)
    if not args.train_only and not args.compress_only:
        print("Starting decompression...")
        t_start = time.perf_counter()
        # BoaFile.decompress() returns the original bytes
        decompressed_bytes = boa.decompress(progress=progress)
        out_path = exp_dir / f"{name}_decompressed.{file_format}"
        with open(out_path, 'wb') as outf:
            outf.write(decompressed_bytes)
        timings['decompression'] = time.perf_counter() - t_start
        print(f"Decompression complete in {timings['decompression']:.2f}s")

        # Optional verification: compare decompressed bytes with original compression input
        if verify:
            # Compare against the bytes we actually compressed (compress_file_path)
            with open(compress_file_path, 'rb') as rf:
                ref_bytes = rf.read()
            same = decompressed_bytes == ref_bytes
            if same:
                print(f"VERIFY: OK — decompressed output matches input ({len(decompressed_bytes)} bytes)")
            else:
                # Provide small diagnostic: print sizes and first mismatch position (bounded)
                print("VERIFY: MISMATCH — decompressed output differs from input")
                if len(decompressed_bytes) != len(ref_bytes):
                    print(f"  Sizes differ: decompressed={len(decompressed_bytes)} vs input={len(ref_bytes)}")
                else:
                    # Find first mismatch up to a cap
                    cap = min(len(decompressed_bytes), 1_000_000)
                    for i in range(cap):
                        if decompressed_bytes[i] != ref_bytes[i]:
                            print(f"  First differing byte at offset {i}: dec={decompressed_bytes[i]} input={ref_bytes[i]}")
                            break

    # Note: configs are stored under experiments/<name>/<name>.yaml when created
    # and can be referenced by experiment name via --config <name>. No copy is necessary.

    if args.show_timings:
        print('\nTimings:')
        for k, v in timings.items():
            print(f"  {k}: {v:.2f}s")


if __name__ == '__main__':
    main()


