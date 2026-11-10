# Boa-Constrictor Baseline Metrics

## Hardware & Environment
* **GPU**: NVIDIA GeForce RTX 5070 Ti Laptop GPU (Blackwell, Compute 12.0)
* **CUDA**: 13.0
* **Model**: Mamba-1 (Untrained, ~7.9M parameters)

## Dataset
* **Name**: Enwik8 (First 1MB chunk)
* **Size**: 1,048,576 bytes

## Baseline Performance (Pre-Optimization)
* **Execution Time**: 7.14 seconds
* **Throughput**: ~145.94 KB/s (0.14 MB/s)
* **Compression Ratio**: 0.99 (Expected behavior for randomly initialized weights)

## Goal
This establishes the starting metrics for the GSoC optimizations targeting the `selective_scan_fn` and parallel GPU streams.
