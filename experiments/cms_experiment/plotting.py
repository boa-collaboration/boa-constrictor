import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import warnings


COLUMN_NAMES = [
 'pt (GeV)',
 'eta',
 'phi (rad)',
 'mass (GeV)',
 'jet area',
 'charged hadron energy (GeV)',
 'neutral hadron energy (GeV)',
 'photon energy (GeV)',
 'electron energy (GeV)',
 'muon energy (GeV)',
 'HF hadron energy (GeV)',
 'HF EM energy (GeV)',
 'charged hadron multiplicity (count)',
 'neutral hadron multiplicity (count)',
 'photon multiplicity (count)',
 'electron multiplicity (count)',
 'muon multiplicity (count)',
 'HF hadron multiplicity (count)',
 'HF EM multiplicity (count)',
 'charged EM energy (GeV)',
 'charged muon energy (GeV)',
 'neutral EM energy (GeV)',
 'charged multiplicity (count)',
 'neutral multiplicity (count)',
]


def load_flat_float_array(path: Path, dtype=np.float32):
    b = path.read_bytes()
    arr = np.frombuffer(b, dtype=dtype)
    return arr


def reshape_to_records(arr: np.ndarray, ncols: int):
    if arr.size % ncols != 0:
        raise ValueError(f"Array length ({arr.size}) not divisible by ncols ({ncols})")
    return arr.reshape(-1, ncols)


def plot_columns(orig_rec, decomp_rec, names, out_dir: Path, nrows_to_plot=200,
                 alpha_orig=0.6, alpha_decomp=0.9,
                 create_hist=True, bins=50, hist_log=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    ncols = orig_rec.shape[1]
    # limit rows
    rows = min(nrows_to_plot, orig_rec.shape[0])

    for i in range(ncols):
        col_name = names[i] if i < len(names) else f"col_{i}"
        y_orig = orig_rec[:rows, i]
        y_decomp = decomp_rec[:rows, i]

        # filter NaNs (if any)
        y1 = y_orig[~np.isnan(y_orig)]
        y2 = y_decomp[~np.isnan(y_decomp)]
        residual = y_orig - y_decomp

        if create_hist:
            # Histogram plot (counts vs feature value) + residuals panel
            fig = plt.figure(figsize=(7, 4.5))
            gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.5)
            ax_h = fig.add_subplot(gs[0])
            ax_res = fig.add_subplot(gs[1])

            # use same bin edges for both so counts are comparable
            try:
                _, bin_edges = np.histogram(y1, bins=bins)
                ax_h.hist(y1, bins=bin_edges, alpha=alpha_orig, label='original')
                ax_h.hist(y2, bins=bin_edges, alpha=alpha_decomp, label='decompressed')
            except ValueError:
                # fallback: let matplotlib choose bins
                ax_h.hist(y1, bins=bins, alpha=alpha_orig, label='original')
                ax_h.hist(y2, bins=bins, alpha=alpha_decomp, label='decompressed')

            ax_h.set_xlabel(col_name, fontsize=9)
            ax_h.set_ylabel('count', fontsize=9)
            if hist_log:
                ax_h.set_yscale('log')
            ax_h.legend(fontsize='small')

            ax_res.plot(range(rows), residual, color='black', lw=0.5, alpha=0.8)
            ax_res.set_xlabel('row index (first N)', fontsize=8)
            ax_res.set_ylabel('$\\Delta$ = original - decompressed', fontsize=8)
            ax_res.tick_params(axis='both', labelsize=8)

            out_file = out_dir / f"col_{i:02d}_{sanitize_filename(col_name)}_hist.png"
            fig.savefig(out_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            # Only residual plot
            fig, ax = plt.subplots(figsize=(7, 3.0))
            ax.plot(range(rows), residual, color='black', lw=0.6, alpha=0.9)
            ax.set_title(col_name, fontsize=10)
            ax.set_xlabel('row index (first N)', fontsize=9)
            ax.set_ylabel('$\\Delta$ = original - decompressed', fontsize=9)
            ax.tick_params(axis='both', labelsize=9)

            out_file = out_dir / f"col_{i:02d}_{sanitize_filename(col_name)}_residual.png"
            fig.savefig(out_file, dpi=150, bbox_inches='tight')
            plt.close(fig)


def apply_hep_style(style: str):
    """Apply a HEP plotting style.

    style: one of 'none' (do nothing), 'atlas', 'hep', 'cms', or 'mplhep'.
    Tries to use mplhep if installed; otherwise falls back to a compact matplotlib rcParams
    that approximates common HEP plotting aesthetics (sans logos).
    """
    style = (style or "").lower()
    if style in ("", "none", "no"):
        return

    try:
        import mplhep as hep
        # mplhep supports a style.use API that accepts common experiment strings
        try:
            # prefer explicit names
            if style in ("atlas", "mplhep"):
                hep.style.use("ATLAS")
            elif style == "cms":
                hep.style.use("CMS")
            else:
                # generic 'hep'
                hep.style.use("hep")
        except Exception:
            # fallback to generic style application
            try:
                hep.style.use(style.upper())
            except Exception:
                plt.style.use("seaborn-whitegrid")
        return
    except Exception:
        warnings.warn("mplhep not available; using fallback Matplotlib settings for HEP-like style")

    # Fallback rcParams setting (small, clean, grid, ticks outward)
    plt.rcParams.update({
        "figure.figsize": (7, 4.5),
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.4,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
    })

def sanitize_filename(s: str) -> str:
    # produce a short safe filename segment from the column name
    import re

    s2 = re.sub(r"[^0-9A-Za-z._-]", "_", s)
    # condense multiple underscores
    s2 = re.sub(r"_+", "_", s2)
    # limit length
    return s2[:64]


def main():
    p = argparse.ArgumentParser(description="Plot decompressed.bin vs original float data for cms_experiment")
    p.add_argument("--original", "-o", required=True, type=Path, help="Path to original float binary file")
    p.add_argument("--decompressed", "-d", required=True, type=Path, help="Path to decompressed binary file")
    p.add_argument("--dtype", default="float32", help="NumPy dtype used when interpreting bytes (default: float32)")
    p.add_argument("--nrows", type=int, default=200, help="Number of rows to plot (default: 200)")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "plots", help="Output directory for plots")
    p.add_argument("--alpha-original", type=float, default=0.6, help="Alpha/transparency for original histogram (default: 0.6)")
    p.add_argument("--alpha-decompressed", type=float, default=0.9, help="Alpha/transparency for decompressed histogram (default: 0.9)")
    # histogram options
    p.add_argument("--bins", type=int, default=100, help="Number of bins for histograms (default: 100)")
    p.add_argument("--no-hist", dest='hist', action='store_false', help="Do not create histogram plots")
    p.add_argument("--hist-log", dest='hist_log', action='store_true', help="Use log scale for histogram counts")
    p.add_argument("--style", choices=["none", "atlas", "hep", "cms", "mplhep"], default="hep",
                   help="Plotting style to apply. Uses mplhep when available (choices: none, atlas, hep, cms, mplhep).")
    p.set_defaults(hist=True)
    args = p.parse_args()

    dtype = np.dtype(args.dtype)
    ncols = len(COLUMN_NAMES)

    orig_arr = load_flat_float_array(args.original, dtype=dtype)
    decomp_arr = load_flat_float_array(args.decompressed, dtype=dtype)

    orig_rec = reshape_to_records(orig_arr, ncols)
    decomp_rec = reshape_to_records(decomp_arr, ncols)

    if orig_rec.shape != decomp_rec.shape:
        raise ValueError(f"Original shape {orig_rec.shape} != decompressed shape {decomp_rec.shape}")

    # Apply chosen plotting style (mplhep preferred, otherwise use fallback)
    apply_hep_style(args.style)

    plot_columns(orig_rec, decomp_rec, COLUMN_NAMES, args.out_dir, nrows_to_plot=args.nrows,
                 alpha_orig=args.alpha_original, alpha_decomp=args.alpha_decompressed,
                 create_hist=args.hist, bins=args.bins, hist_log=args.hist_log)
    print(f"Plots written to {args.out_dir}")


if __name__ == "__main__":
    main()
