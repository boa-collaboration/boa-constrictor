# ATLAS jets round-trip helper

This experiment demonstrates a reversible pipeline for the ATLAS FTAG HDF5 "jets" dataset:

1) Download an ATLAS HDF5 file (via XRootD or HTTPS)
2) Extract the `jets` dataset into a compact, compressor-friendly binary format:
   - `atlas.bin`: raw row-major bytes for the entire dataset
   - `atlas.meta.json`: minimal metadata required to reconstruct (shape + dtype)
3) Reconstruct an HDF5 file from `atlas.bin` + `atlas.meta.json`
4) Compare the original and reconstructed HDF5 files for equality of `jets`

The pipeline streams data to avoid excessive memory use and faithfully preserves structured dtypes (named fields) by round-tripping `dtype.descr`.

## Requirements

- Python 3.9+
- numpy
- h5py

Install (if needed):

```bash
pip install -r ../../requirements.txt
```

## Data source

Default source (small sample from EOS):

- root://eospublic.cern.ch//eos/opendata/atlas/datascience/ATLAS-FTAG-2023-05/mc-flavtag-ttbar-small.h5

You can also pass a different `--src` (supports `root://`, `https://`, or a local path).

## Quick start

Run all steps end-to-end (download → extract → reconstruct → compare):

```bash
python download.py --all-steps \
  --src root://eospublic.cern.ch//eos/opendata/atlas/datascience/ATLAS-FTAG-2023-05/mc-flavtag-ttbar-small.h5
```

This will produce:

- `atlas.h5` — the downloaded source file (or copy if `--src` was local)
- `atlas.bin` — raw bytes of the `jets` dataset (row-major)
- `atlas.meta.json` — metadata sidecar (shape + dtype and full `dtype.descr` for structured arrays)
- `atlas_reconstructed.h5` — reconstructed HDF5 containing a `jets` dataset
- Console report for the field-by-field comparison (chunked)

Exit code is `0` on success, non-zero on mismatch or error.

## Individual steps

- Download only:

```bash
python download.py --download --src <root-or-https-or-local> --h5 atlas.h5
```

- Extract `jets` to BIN + META (streams data in chunks):

```bash
python download.py --extract --h5 atlas.h5 --bin atlas.bin --meta atlas.meta.json
```

- Reconstruct HDF5 from BIN + META:

```bash
python download.py --reconstruct --bin atlas.bin --meta atlas.meta.json --recon-h5 atlas_reconstructed.h5
```

- Compare original versus reconstructed:

```bash
python download.py --compare --h5 atlas.h5 --recon-h5 atlas_reconstructed.h5
```

## CLI options (summary)

- `--src`         Source HDF5 (root://, https://, or local path)
- `--h5`          Local HDF5 filename (default: atlas.h5)
- `--bin`         Output/input BIN path (default: atlas.bin)
- `--meta`        Output/input metadata JSON (default: atlas.meta.json)
- `--recon-h5`    Output reconstructed HDF5 (default: atlas_reconstructed.h5)
- `--download`    Download the HDF5 if missing
- `--extract`     Extract `jets` to BIN + META
- `--reconstruct` Reconstruct HDF5 from BIN + META
- `--compare`     Compare original `jets` vs reconstructed `jets`
- `--all-steps`   Convenience flag enabling download, extract, reconstruct, compare

## Format details

- `atlas.bin` stores the `jets` dataset as contiguous row-major bytes.
- `atlas.meta.json` stores:
  - `shape`: full dataset shape
  - `dtype`: NumPy dtype string (fallback for simple dtypes)
  - `dtype_descr`: full structured dtype description (`dtype.descr`) when applicable
  - `order`: memory order used (`"C"`)

On reconstruction, the script prefers `dtype_descr` (if present) to faithfully rebuild structured (fielded) dtypes. Otherwise, it uses `dtype`.

## Comparison method

- The comparison loads chunks of rows to limit memory usage.
- If the dtype is structured, it compares each field independently:
  - Floating fields: `np.allclose` with `atol=0`, `rtol=0` (configurable in code)
  - Non-floating fields: exact equality via `np.array_equal`
- For non-structured arrays, `np.allclose` is used directly.

## Notes and troubleshooting

- XRootD: If `xrdcp` is available and `--src` is `root://...`, the script will use `xrdcp`. Otherwise, it falls back to an HTTPS mirror.
- Large files: `save_bin` writes in chunks; reconstruction uses `np.fromfile`. Ensure free disk space is sufficient for the BIN and reconstructed HDF5.
- Structured dtype mismatch: The script embeds `dtype.descr` in the metadata to avoid void-type ambiguities; if you edit META manually, ensure it remains valid JSON convertible to a list-of-tuples for `np.dtype`.

## License

This experiment uses open ATLAS data hosted on CERN EOS. Code is provided as-is under the repository's license.
