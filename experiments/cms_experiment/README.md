# CMS experiment: plotting original vs decompressed jet features

This folder contains a small script (`plotting.py`) to compare the original float data with a decompressed version and generate per-feature plots.

## Requirements

- Python 3.8+
- NumPy, Matplotlib
- Optional: `mplhep` for nicer HEP-style plots

Install the minimal deps (from the repo root or this folder):

```bash
pip install numpy matplotlib mplhep
```

You can also use the project-wide `requirements.txt` if you plan to run other parts of the repo.

## Data expected

Two flat binary files containing IEEE-754 floats of the same dtype and length:

- `--original`: the original float binary (e.g. `CMS_DATA_float32.bin`)
- `--decompressed`: the decompressed float binary (e.g. `cms_experiment_decompressed.bin`)

Both files must contain the same number of floats and be reshaped into N rows × 24 columns, where the 24 columns correspond to the following features:

1. pt (GeV)
2. eta
3. phi (rad)
4. mass (GeV)
5. jet area
6. charged hadron energy (GeV)
7. neutral hadron energy (GeV)
8. photon energy (GeV)
9. electron energy (GeV)
10. muon energy (GeV)
11. HF hadron energy (GeV)
12. HF EM energy (GeV)
13. charged hadron multiplicity (count)
14. neutral hadron multiplicity (count)
15. photon multiplicity (count)
16. electron multiplicity (count)
17. muon multiplicity (count)
18. HF hadron multiplicity (count)
19. HF EM multiplicity (count)
20. charged EM energy (GeV)
21. charged muon energy (GeV)
22. neutral EM energy (GeV)
23. charged multiplicity (count)
24. neutral multiplicity (count)

## How to run

From this folder:

```bash
python plotting.py \
  --original CMS_DATA_float32.bin \
  --decompressed cms_experiment_decompressed.bin \
  --nrows 500000 \
  --out-dir jet-features
```

Useful flags:

- `--dtype` Float dtype of the binary files (default: `float32`)
- `--nrows` Number of rows to read and plot (default: 200)
- `--out-dir` Output directory for plots (default: `./plots`)
- `--bins` Number of bins for histograms (default: 100)
- `--hist-log` Use log scale for histogram counts
- `--no-hist` Skip histograms and only save residual plots
- `--alpha-original`, `--alpha-decompressed` Transparency for overlaid histograms
- `--style` Plot style: `none`, `atlas`, `hep`, `cms`, or `mplhep` (default: `hep`). If `mplhep` is installed, the selected style will be applied.

## Outputs

- For each column, a PNG is written under `--out-dir`.
  - With histograms enabled: `col_XX_<name>_hist.png` (top: overlaid histograms, bottom: residuals)
  - With `--no-hist`: `col_XX_<name>_residual.png` (residuals only)

## Troubleshooting

- "Original shape != decompressed shape": make sure both files contain the same number of floats and you used the correct `--dtype`.
- If you see many NaNs in residuals, check the dtype and file integrity.
- Large `--nrows` can be memory intensive; try a smaller value to test.
