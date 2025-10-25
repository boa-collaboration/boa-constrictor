import argparse
import time
from pathlib import Path
import yaml
import numpy as np
import torch
from tqdm import tqdm

from model import BoaConstrictor, ByteDataloader, make_splits
from boa import BoaFile
from train import train


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


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
        precision = _prompt("Precision (fp32|fp16|fp8)", "fp16")
        seq_len = _prompt("Sequence length (seq_len)", 32768, int)
        batch_size = _prompt("Batch size", 3, int)
        d_model = _prompt("Model d_model", 256, int)
        num_layers = _prompt("Model num_layers", 8, int)
        lr = _prompt("Learning rate", 5e-4, float)
        epochs = _prompt("Epochs", 50, int)
        chunks_count = _prompt("Compression chunks_count", 1000, int)
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
            'compression': {'chunks_count': int(chunks_count)},
            'splits': splits
        }

        # Decide where to save the config
        cfg_path = args.config if args.config is not None else Path('configs') / f"{name}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg_path, 'w') as f:
            yaml.safe_dump(cfg, f)
        print(f"Wrote new experiment config to: {cfg_path}")

        # Use the newly created config for the rest of the run
        args.config = cfg_path

    if args.config is None:
        raise ValueError('Either --config must be provided or use --new-experiment to create one interactively')

    config = load_config(args.config)

    # Apply CLI overrides
    progress = not args.no_progress and config.get('progress', True)
    device = args.device or config.get('device', 'cuda' if torch.cuda.is_available() else 'cuda')
    precision = args.precision or config.get('precision', 'fp32')

    # Experiment parameters (with sensible defaults)
    name = config.get('name', 'experiment')
    file_path = config.get('file_path', '')
    seq_len = config.get('dataloader', {}).get('seq_len', 32768)
    batch_size = config.get('dataloader', {}).get('batch_size', 3)
    d_model = config.get('model', {}).get('d_model', 256)
    num_layers = config.get('model', {}).get('num_layers', 8)
    lr = config.get('training', {}).get('lr', 5e-4)
    num_epochs = config.get('training', {}).get('epochs', 50)

    timings = {}

    # Read file
    t0 = time.perf_counter()
    if not file_path:
        raise ValueError('file_path must be set in the config or passed via CLI')

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, 'rb') as f:
        data_bytes = f.read()

    timings['read_bytes'] = time.perf_counter() - t0
    print(f"Read {len(data_bytes)} bytes from {file_path} in {timings['read_bytes']:.2f}s")

    # Setup model, dataloaders, optimizer, loss
    model = BoaConstrictor(d_model=d_model, num_layers=num_layers)

    dataloader = ByteDataloader(data_bytes, seq_len=seq_len, batch_size=batch_size)

    train_b, val_b, test_b = make_splits(data_bytes, dataloader.seq_len, dataloader.batch_size,
                                         splits=tuple(config.get('splits', (0.8, 0.1, 0.1))))

    train_loader = ByteDataloader(train_b, seq_len=dataloader.seq_len, batch_size=dataloader.batch_size)
    val_loader = ByteDataloader(val_b, seq_len=dataloader.seq_len, batch_size=dataloader.batch_size)
    test_loader = ByteDataloader(test_b, seq_len=dataloader.seq_len, batch_size=dataloader.batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training
    if not args.compress_only and not args.decompress_only:
        print(f"Starting training on device={device}, precision={precision}, epochs={num_epochs}")
        t_start = time.perf_counter()
        train(model, train_loader, val_loader, test_loader, optimizer, criterion,
              device=device, name=name, NUM_EPOCHS=num_epochs, PRECISION=precision, progress=progress)
        timings['training'] = time.perf_counter() - t_start
        print(f"Training complete in {timings['training']:.2f}s")

    # Compression
    if not args.train_only and not args.decompress_only:
        print("Starting compression...")
        t_start = time.perf_counter()
        boa = BoaFile(f"{name}.boa", model)
        boa.compress(npz_path=f"{name}.npz",
                     chunks_count=config.get('compression', {}).get('chunks_count', 1000),
                     progress=progress)
        timings['compression'] = time.perf_counter() - t_start
        print(f"Compression complete in {timings['compression']:.2f}s")

    # Decompression
    if not args.train_only and not args.compress_only:
        print("Starting decompression...")
        t_start = time.perf_counter()
        boa.decompress(npz_path=f"{name}.npz", output_path=f"{name}_output.boa", progress=progress)
        timings['decompression'] = time.perf_counter() - t_start
        print(f"Decompression complete in {timings['decompression']:.2f}s")

    if args.show_timings:
        print('\nTimings:')
        for k, v in timings.items():
            print(f"  {k}: {v:.2f}s")


if __name__ == '__main__':
    main()


