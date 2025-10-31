# HEPMC Experiment

This folder contains a helper script to fetch a HEPMC sample from CERN EOS Open Data, extract it, and produce two binary outputs for downstream processing.

## What it does
- Downloads the archive from EOS:
  - `root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc16_13TeV/HEPMC.43646133._000001.tar.gz.1`
- Safely extracts the archive to a temporary directory
- Locates the contained HEPMC payload
- Writes:
  - `hepmc.hepmc` — full HEPMC byte stream
  - `hepmc_200m.hepmc` — first 200 MiB of the byte stream

## Requirements
- Python 3.8+
- Packages (already listed in the repo `requirements.txt`):
  - `requests`, `tqdm`
- Optional fallback tool:
  - `xrdcp` (from XRootD) — used only if HTTPS streaming fails

Install dependencies at the repo root:

```bash
pip install -r requirements.txt
```

## Usage
Run from this directory (or pass the path):

```bash
# default: downloads the ATLAS Open Data HEPMC sample, extracts, and writes outputs
python download.py

# force re-download and overwrite outputs
python download.py --force

# use a custom URL (supports root:// or https://)
python download.py --url <your-url>
```

Outputs will be created in this directory:
- `hepmc.hepmc`
- `hepmc_200m.hepmc`

## Notes
- 200 MiB means `200 * 1024 * 1024` bytes.
- If the archive contains `*.hepmc.gz`, the script transparently gunzips it.
- If neither `.hepmc` nor `.hepmc.gz` is found, the largest file in the archive is used as a heuristic fallback.
- If HTTPS fails and `xrdcp` is available, the script automatically falls back to `xrdcp`.
