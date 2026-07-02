#!/usr/bin/env python3
"""
BOA Constrictor Benchmark Suite
Runs all backbones (Python and C++), records compression ratio, speed, model size,
and generates a Pareto plot (compression ratio vs. model size).
"""

import os
import sys
import time
import json
import subprocess
import argparse

# Ensure repo root is on the path so backbone modules can be found
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuration
CMS_DATA_PATH = "experiments/cms_experiment/CMS_DATA_float32.bin"
CONFIG_PATH = "experiments/cms_experiment/cms_experiment.yaml"
RESULTS_FILE = "benchmark_results.json"
BACKBONES = ["GRU", "LSTM", "MinGRU", "Transformer"]

def get_param_count(backbone_name):
    """Get parameter count from Python implementation."""
    import importlib
    mod = importlib.import_module(f"{backbone_name.lower()}_backbone")
    model = mod.BoaConstrictor(d_model=256, num_layers=1)
    return sum(p.numel() for p in model.parameters())

def run_training(backbone_name):
    """Train a backbone using the Python pipeline."""
    print(f"\n{'='*50}")
    print(f"Training: {backbone_name}")
    print(f"{'='*50}\n")

    # Patch model.py
    with open("model.py", "r") as f:
        lines = f.readlines()
    with open("model.py", "w") as f:
        for line in lines:
            if "_backbone import BoaConstrictor" in line or "from gru_backbone" in line:
                f.write(f'from {backbone_name.lower()}_backbone import BoaConstrictor\n')
            else:
                f.write(line)

    # Clean config
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    config.pop("model_path", None)
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f)

    # Remove old checkpoints
    for fname in os.listdir("experiments/cms_experiment/"):
        if fname.endswith(".pt"):
            os.remove(f"experiments/cms_experiment/{fname}")

    start = time.time()
    result = subprocess.run(
        ["python", "main.py", "--config", "cms_experiment",
         "--show-timings", "--device", "cuda"],
        capture_output=True, text=True
    )
    elapsed = time.time() - start

    # Extract ratio from output
    import re
    match = re.search(r"\[TEST\] bpp=([\d.]+)\s+ratio ~ ([\d.]+)x", result.stdout)
    if match:
        bpp = float(match.group(1))
        ratio = float(match.group(2))
        params = get_param_count(backbone_name)
        return {
            "backbone": backbone_name,
            "ratio": ratio,
            "bpp": bpp,
            "time": int(elapsed),
            "params": params,
            "model_size_mb": params * 4 / 1e6,
            "status": "success"
        }
    else:
        return {
            "backbone": backbone_name,
            "status": "failed",
            "error": "Could not extract ratio from output"
        }

def load_baseline_results():
    """Load baseline results from the paper."""
    return [
        {"backbone": "LZMA", "ratio": 3.22, "bpp": 2.490, "time": 26, "params": 0, "model_size_mb": 0},
        {"backbone": "ZLIB", "ratio": 2.56, "bpp": 3.130, "time": 10, "params": 0, "model_size_mb": 0},
        {"backbone": "Mamba", "ratio": 4.03, "bpp": 1.990, "time": 260, "params": 1100000, "model_size_mb": 4.4},
    ]

def generate_pareto_plot(results, output_file="pareto_plot.png"):
    """Generate Pareto plot: Compression Ratio vs Model Size."""
    df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    colors = {"GRU": "blue", "LSTM": "green", "MinGRU": "orange", "Transformer": "red", "Mamba": "purple"}

    for backbone in df["backbone"].unique():
        data = df[df["backbone"] == backbone]
        if not data.empty and "ratio" in data.columns:
            plt.scatter(
                data["model_size_mb"], data["ratio"],
                label=backbone, s=100, color=colors.get(backbone, "gray")
            )
            for _, row in data.iterrows():
                plt.annotate(
                    f"{row['ratio']:.2f}x",
                    (row["model_size_mb"], row["ratio"]),
                    xytext=(5, 5), textcoords="offset points"
                )

    # Baseline markers
    baselines = load_baseline_results()
    for b in baselines:
        plt.axhline(y=b["ratio"], linestyle="--", alpha=0.5, label=f"{b['backbone']} baseline")

    plt.xlabel("Model Size (MB)")
    plt.ylabel("Compression Ratio (x)")
    plt.title("Pareto Frontier: Compression Ratio vs Model Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Pareto plot saved to: {output_file}")
    plt.show()

def run_smoke_tests():
    """Run smoke tests for Python backbones."""
    print("\n" + "="*50)
    print("Running Python Smoke Tests")
    print("="*50 + "\n")

    for backbone in BACKBONES:
        try:
            import importlib
            mod = importlib.import_module(f"{backbone.lower()}_backbone")
            model = mod.BoaConstrictor(d_model=256, num_layers=1)
            dummy_input = torch.randint(0, 256, (1, 100))
            output = model(dummy_input)
            assert output.shape == (1, 100, 256), f"Shape mismatch: {output.shape}"
            probs = model.get_probabilities(dummy_input)
            assert probs.shape == (1, 100, 256)
            assert torch.allclose(probs.sum(dim=-1), torch.ones(1, 100), atol=1e-4)
            print(f"✅ {backbone} smoke test passed")
        except Exception as e:
            print(f"❌ {backbone} smoke test failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true", help="Only run smoke tests")
    parser.add_argument("--benchmark-all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--plot-only", action="store_true", help="Generate Pareto plot from existing results")
    parser.add_argument("--backbone", type=str, help="Run only this backbone")
    args = parser.parse_args()

    if args.smoke_only:
        run_smoke_tests()
        return

    if args.plot_only:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r") as f:
                results = json.load(f)
            generate_pareto_plot(results)
        else:
            print("No results file found. Run benchmarks first.")
        return

    print("="*50)
    print("BOA Constrictor Benchmark Suite")
    print("="*50 + "\n")

    # Run smoke tests
    run_smoke_tests()

    # Determine which backbones to run
    to_run = BACKBONES if not args.backbone else [args.backbone]

    results = []
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results from {RESULTS_FILE}")

    existing = [r["backbone"] for r in results]
    for backbone in to_run:
        if backbone not in existing:
            result = run_training(backbone)
            results.append(result)
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved result for {backbone}")

    # Generate Pareto plot
    generate_pareto_plot(results)

    print("\n" + "="*50)
    print("Final Results Summary")
    print("="*50 + "\n")
    df = pd.DataFrame(results)
    print(df[["backbone", "ratio", "bpp", "time", "params", "model_size_mb"]])

if __name__ == "__main__":
    main()
