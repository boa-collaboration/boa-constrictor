"""
Download and prepare CAMEL (Cosmology and Astrophysics with MachinE Learning) 
simulation data for training.

CAMEL simulations: https://www.camel-simulations.org/
This script downloads real CAMEL simulation data from their public repository.

Requirements:
    pip install numpy h5py requests tqdm

CAMEL data is stored in HDF5 format on their public servers.
"""

import numpy as np
import sys
import os
import requests
from tqdm import tqdm
import h5py
import warnings

# CAMEL data repository
# Data is available at: https://users.flatironinstitute.org/~camels/
BASE_URL = "https://users.flatironinstitute.org/~camels/Sims/"

# Configuration - downloading IllustrisTNG simulation suite
SIMULATION_SUITE = "IllustrisTNG"  # Options: IllustrisTNG, SIMBA
SIMULATION_SET = "CV"  # CV = Cosmo Variations, 1P = 1-parameter variations, LH = Latin Hypercube
SIMULATION_NUMBER = 0  # Simulation index (0-26 for CV)
SNAPSHOT = 24  # Snapshot number (33 = z=0, present day)

# Alternative: Try different snapshot formats
SNAPSHOT_FORMATS = [
    "snap_{:03d}.hdf5",
    "snapshot_{:03d}.hdf5", 
    "snapdir_{:03d}/snap_{:03d}.0.hdf5"
]

# Target ~1GB of data
TARGET_SIZE_GB = 1.0

def download_file(url, local_filename, expected_size=None):
    """
    Download a file from URL with progress bar
    
    Args:
        url: URL to download from
        local_filename: Local path to save file
        expected_size: Expected file size in bytes (for progress bar)
    """
    print(f"Downloading from: {url}")
    print(f"Saving to: {local_filename}")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        if total_size == 0 and expected_size:
            total_size = expected_size
            
        block_size = 8192
        
        with open(local_filename, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(local_filename)) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✓ Downloaded successfully: {os.path.getsize(local_filename) / (1024**2):.2f} MB")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Download failed: {e}")
        if os.path.exists(local_filename):
            os.remove(local_filename)
        return False

def extract_camel_data_from_hdf5(hdf5_file, output_bin, target_size_mb=1024):
    """
    Extract particle data from CAMEL HDF5 snapshot and save as binary
    
    Args:
        hdf5_file: Path to HDF5 snapshot file
        output_bin: Output binary file path
        target_size_mb: Target size in MB
    """
    print(f"\nExtracting data from {hdf5_file}...")
    
    with h5py.File(hdf5_file, 'r') as f:
        print("\nAvailable particle types:")
        for key in f.keys():
            print(f"  - {key}")
        
        # CAMEL snapshots contain different particle types:
        # PartType0: Gas particles
        # PartType1: Dark matter
        # PartType4: Stars
        # PartType5: Black holes
        
        # Focus on gas particles (PartType0) for astrophysical data
        if 'PartType0' not in f:
            print("Error: No gas particles (PartType0) found in snapshot")
            return None
            
        gas = f['PartType0']
        print(f"\nGas particle datasets:")
        for key in gas.keys():
            if isinstance(gas[key], h5py.Dataset):
                print(f"  - {key}: shape={gas[key].shape}, dtype={gas[key].dtype}")
        
        # Key fields to extract:
        # - Coordinates: 3D positions
        # - Velocities: 3D velocities  
        # - Density: gas density
        # - InternalEnergy: thermal energy
        # - Masses: particle masses
        # - ElectronAbundance: ionization state
        # - Metallicity: metal content
        
        fields_to_extract = []
        field_names = []
        
        # Coordinates (3 components)
        if 'Coordinates' in gas:
            coords = gas['Coordinates'][:]
            fields_to_extract.append(coords)
            field_names.extend(['x', 'y', 'z'])
            print(f"✓ Loaded Coordinates: {coords.shape}")
        
        # Velocities (3 components)
        if 'Velocities' in gas:
            vels = gas['Velocities'][:]
            fields_to_extract.append(vels)
            field_names.extend(['vx', 'vy', 'vz'])
            print(f"✓ Loaded Velocities: {vels.shape}")
        
        # Density (scalar)
        if 'Density' in gas:
            density = gas['Density'][:].reshape(-1, 1)
            fields_to_extract.append(density)
            field_names.append('density')
            print(f"✓ Loaded Density: {density.shape}")
        
        # Masses (scalar)
        if 'Masses' in gas:
            masses = gas['Masses'][:].reshape(-1, 1)
            fields_to_extract.append(masses)
            field_names.append('mass')
            print(f"✓ Loaded Masses: {masses.shape}")
        
        # Internal Energy / Temperature (scalar)
        if 'InternalEnergy' in gas:
            energy = gas['InternalEnergy'][:].reshape(-1, 1)
            fields_to_extract.append(energy)
            field_names.append('internal_energy')
            print(f"✓ Loaded InternalEnergy: {energy.shape}")
        
        # ElectronAbundance (scalar)
        if 'ElectronAbundance' in gas:
            electron = gas['ElectronAbundance'][:].reshape(-1, 1)
            fields_to_extract.append(electron)
            field_names.append('electron_abundance')
            print(f"✓ Loaded ElectronAbundance: {electron.shape}")
        
        # Metallicity (can be scalar or array)
        if 'Metallicity' in gas:
            metallicity = gas['Metallicity'][:]
            if len(metallicity.shape) == 1:
                metallicity = metallicity.reshape(-1, 1)
            # If multi-element, take total metallicity (first column or sum)
            if metallicity.shape[1] > 1:
                metallicity = metallicity[:, 0:1]  # Take first element
            fields_to_extract.append(metallicity)
            field_names.append('metallicity')
            print(f"✓ Loaded Metallicity: {metallicity.shape}")
        
        # Combine all fields
        if not fields_to_extract:
            print("Error: No valid fields found to extract")
            return None
            
        combined_data = np.concatenate(fields_to_extract, axis=1).astype(np.float32)
        
        print(f"\nCombined data shape: {combined_data.shape}")
        print(f"Fields: {field_names}")
        print(f"Total size: {combined_data.nbytes / (1024**2):.2f} MB")
        
        # If data is larger than target, subsample
        target_bytes = target_size_mb * 1024 * 1024
        if combined_data.nbytes > target_bytes:
            n_particles_target = target_bytes // (combined_data.shape[1] * 4)
            print(f"\nSubsampling from {combined_data.shape[0]} to {n_particles_target} particles...")
            indices = np.random.choice(combined_data.shape[0], n_particles_target, replace=False)
            combined_data = combined_data[indices]
            print(f"New size: {combined_data.nbytes / (1024**2):.2f} MB")
    
    return combined_data, field_names

def save_binary_file(data, filename):
    """Save data as binary file in float32 format"""
    data_flat = data.flatten().astype(np.float32)
    data_flat.tofile(filename)
    
    file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"\n✓ Saved {filename}: {file_size_mb:.2f} MB")
    print(f"  Shape: {data.shape}")
    print(f"  Total elements: {data_flat.size:,}")
    
    return file_size_mb

def create_subset_file(input_file, output_file, target_size_mb=200):
    """Create a subset file of specified size"""
    bytes_per_float = 4
    target_bytes = target_size_mb * 1024 * 1024
    num_floats = target_bytes // bytes_per_float
    
    print(f"\nCreating {target_size_mb}MB subset...")
    
    data = np.fromfile(input_file, dtype=np.float32, count=num_floats)
    data.tofile(output_file)
    
    actual_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ Saved {output_file}: {actual_size_mb:.2f} MB")
    print(f"  Elements: {data.size:,}")

def main():
    print("=" * 70)
    print("CAMEL Simulation Data Downloader")
    print("=" * 70)
    print()
    print("CAMEL: Cosmology and Astrophysics with MachinE Learning")
    print("Website: https://www.camel-simulations.org/")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Simulation Suite: {SIMULATION_SUITE}")
    print(f"  Simulation Set: {SIMULATION_SET}")
    print(f"  Simulation Number: {SIMULATION_NUMBER}")
    print(f"  Snapshot: {SNAPSHOT} (z=0)")
    print("=" * 70)
    print()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct URL for CAMEL snapshot
    # Try different URL patterns as CAMEL structure varies
    sim_name = f"{SIMULATION_SET}_{SIMULATION_NUMBER}"
    
    # Try snapdir format first (most common)
    snapshot_filename = f"snapdir_{SNAPSHOT:03d}"
    snap_file = f"snapshot_024.hdf5"
    download_url = f"{BASE_URL}{SIMULATION_SUITE}/{SIMULATION_SET}/{sim_name}/{snap_file}"
    
    local_hdf5 = os.path.join(output_dir, snap_file)
    camel_bin_path = os.path.join(output_dir, "camel.bin")
    camel_200m_path = os.path.join(output_dir, "camel_200m.bin")
    
    try:
        # Download HDF5 snapshot if not already present
        if not os.path.exists(local_hdf5):
            print(f"Downloading CAMEL snapshot...")
            success = download_file(download_url, local_hdf5)
            if not success:
                print("\n" + "=" * 70)
                print("Alternative: Manual Download")
                print("=" * 70)
                print("\nIf automatic download fails, you can:")
                print("1. Visit: https://www.camel-simulations.org/")
                print("2. Navigate to the data repository")
                print("3. Download a snapshot file manually")
                print("4. Place it in this directory as 'snap_033.hdf5'")
                print("5. Run this script again")
                print("=" * 70)
                return
        else:
            print(f"✓ HDF5 file already exists: {local_hdf5}")
        
        # Extract data from HDF5
        data, field_names = extract_camel_data_from_hdf5(
            local_hdf5, 
            camel_bin_path, 
            target_size_mb=int(TARGET_SIZE_GB * 1024)
        )
        
        if data is None:
            print("Failed to extract data from HDF5 file")
            return
        
        # Save full dataset
        print("\n" + "=" * 70)
        print("Saving binary files...")
        print("=" * 70)
        save_binary_file(data, camel_bin_path)
        
        # Create 200MB subset
        create_subset_file(camel_bin_path, camel_200m_path, target_size_mb=200)
        
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"✓ Full dataset: {camel_bin_path}")
        print(f"✓ Training subset: {camel_200m_path}")
        print(f"✓ Original HDF5: {local_hdf5}")
        print("\nFields in dataset (in order):")
        for i, name in enumerate(field_names):
            print(f"  {i}: {name}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Install required packages: pip install numpy h5py requests tqdm")
        print("2. Check internet connection")
        print("3. Try manual download from: https://www.camel-simulations.org/")
        sys.exit(1)

if __name__ == "__main__":
    main()
