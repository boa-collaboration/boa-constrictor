# CAMEL Experiment

This experiment uses data from the CAMEL (Cosmology and Astrophysics with MachinE Learning) simulations project.

## About CAMEL

CAMEL is a suite of state-of-the-art cosmological hydrodynamical simulations designed to:
- Train machine learning algorithms
- Quantify their potential impact on cosmological analyses
- Study galaxy formation and cosmology

Website: https://www.camel-simulations.org/

## Downloading Real CAMEL Data

### Option 1: Direct Download (Requires Registration)

1. Visit https://www.camel-simulations.org/
2. Register for data access
3. Navigate to the data portal: https://camels.flatironinstitute.org/
4. Download IllustrisTNG or SIMBA simulation snapshots
5. Place the HDF5 snapshot files in this directory
6. Run the download script to extract and convert to binary format

### Option 2: Globus Transfer

CAMEL data can be accessed via Globus for large transfers:
- Endpoint: `camels_PUBLIC`
- Path: `/SIMBA/` or `/IllustrisTNG/`

### Option 3: Use Our Download Script

```bash
# Install requirements
pip install numpy h5py requests tqdm

# Run download script (may require manual file placement)
python download_camel.py
```

## Data Format

The converted binary files contain particle data from cosmological simulations:

**Fields (in order):**
- Position X, Y, Z (comoving coordinates)
- Velocity X, Y, Z (peculiar velocities in km/s)
- Density (gas density)
- Mass (particle mass)
- Internal Energy (thermal energy)
- Electron Abundance (ionization state)
- Metallicity (metal enrichment)

**Files:**
- `camel.bin` - Full dataset (~1GB, float32)
- `camel_200m.bin` - Training subset (200MB, float32)

## Running Compression

```bash
# Train the model
python ../../main.py --config camel_experiment --train

# Compress the data
python ../../main.py --config camel_experiment --compress

# Decompress and verify
python ../../main.py --config camel_experiment --decompress
python ../../main.py --config camel_experiment --verify
```

## Alternative: Synthetic Data

If you cannot access real CAMEL data, the download script can generate synthetic cosmological data with similar statistical properties for testing purposes.

