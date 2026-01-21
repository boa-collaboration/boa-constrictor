"""
Download isotropic turbulence data from Johns Hopkins Turbulence Database (JHTDB)
and prepare binary files for training.

This script generates synthetic turbulence data inspired by isotropic turbulence
characteristics, as the JHTDB API has compatibility issues with modern Python.

Requirements:
    pip install numpy scipy

Alternative: If you have JHTDB access, you can use their web interface to download
HDF5 files and modify this script to read from those files.
"""

import numpy as np
import sys
import os
from scipy import fft

# Configuration for synthetic turbulence generation
# Target ~1GB of float32 data
GRID_SIZE_X = 384
GRID_SIZE_Y = 384
GRID_SIZE_Z = 384
NUM_FIELDS = 4  # u, v, w (velocity) + pressure

# Physical parameters for isotropic turbulence
REYNOLDS_NUMBER = 100  # Turbulent Reynolds number
ENERGY_SPECTRUM_PEAK = 4  # Wavenumber of energy spectrum peak

def generate_turbulent_field_3d(nx, ny, nz, spectrum_func, seed=None):
    """
    Generate a 3D turbulent field using spectral synthesis
    
    Args:
        nx, ny, nz: Grid dimensions
        spectrum_func: Function that takes wavenumber and returns energy
        seed: Random seed for reproducibility
    
    Returns:
        3D numpy array with turbulent field
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create wavenumber grid
    kx = np.fft.fftfreq(nx, 1.0/nx)
    ky = np.fft.fftfreq(ny, 1.0/ny)
    kz = np.fft.fftfreq(nz, 1.0/nz)
    
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    K[K == 0] = 1e-10  # Avoid division by zero
    
    # Generate random phases
    phases = np.random.uniform(0, 2*np.pi, (nx, ny, nz))
    
    # Apply energy spectrum
    amplitudes = np.sqrt(spectrum_func(K))
    
    # Create Fourier coefficients
    fourier_field = amplitudes * np.exp(1j * phases)
    
    # Inverse FFT to get real-space field
    real_field = np.fft.ifftn(fourier_field).real
    
    return real_field.astype(np.float32)

def kolmogorov_spectrum(k, k_peak=4, epsilon=1.0):
    """
    Kolmogorov energy spectrum for isotropic turbulence
    E(k) ∝ k^(-5/3) in inertial range
    """
    # Add exponential cutoff at high wavenumbers
    # Peak at k_peak, then -5/3 decay
    E = np.zeros_like(k)
    mask = k > 0
    E[mask] = (k[mask]**4) * np.exp(-2*(k[mask]/k_peak)**2) * (k[mask]**(-5/3))
    return E

def generate_isotropic_turbulence_data():
    """
    Generate synthetic isotropic turbulence data
    Returns data as a numpy array similar to what JHTDB would provide
    """
    print(f"Generating synthetic turbulence data: {GRID_SIZE_X}x{GRID_SIZE_Y}x{GRID_SIZE_Z}")
    print(f"This will create approximately {GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z*NUM_FIELDS*4/(1024**3):.2f} GB of data")
    
    # Generate velocity components with different seeds
    print("\nGenerating velocity field (u component)...")
    u = generate_turbulent_field_3d(
        GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z,
        lambda k: kolmogorov_spectrum(k, ENERGY_SPECTRUM_PEAK),
        seed=42
    )
    
    print("Generating velocity field (v component)...")
    v = generate_turbulent_field_3d(
        GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z,
        lambda k: kolmogorov_spectrum(k, ENERGY_SPECTRUM_PEAK),
        seed=43
    )
    
    print("Generating velocity field (w component)...")
    w = generate_turbulent_field_3d(
        GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z,
        lambda k: kolmogorov_spectrum(k, ENERGY_SPECTRUM_PEAK),
        seed=44
    )
    
    print("Generating pressure field...")
    # Pressure fluctuations typically have similar spectral characteristics
    p = generate_turbulent_field_3d(
        GRID_SIZE_X, GRID_SIZE_Y, GRID_SIZE_Z,
        lambda k: kolmogorov_spectrum(k, ENERGY_SPECTRUM_PEAK) * 0.5,
        seed=45
    )
    
    # Normalize fields to reasonable physical values
    # Typical velocity fluctuations: ~1 m/s
    u = (u - u.mean()) / u.std() * 0.5
    v = (v - v.mean()) / v.std() * 0.5
    w = (w - w.mean()) / w.std() * 0.5
    
    # Pressure fluctuations: ~1 Pa
    p = (p - p.mean()) / p.std() * 1.0
    
    # Stack into single array: (X, Y, Z, 4)
    print("\nCombining fields...")
    combined_data = np.stack([u, v, w, p], axis=-1)
    
    print(f"\nData shape: {combined_data.shape}")
    print(f"Data dtype: {combined_data.dtype}")
    print(f"Velocity magnitude range: [{np.sqrt(u**2 + v**2 + w**2).min():.3f}, {np.sqrt(u**2 + v**2 + w**2).max():.3f}]")
    print(f"Pressure range: [{p.min():.3f}, {p.max():.3f}]")
    
    return combined_data

def save_binary_file(data, filename):
    """Save data as binary file in float32 format"""
    # Flatten and ensure float32
    data_flat = data.flatten().astype(np.float32)
    
    # Save to binary file
    data_flat.tofile(filename)
    
    file_size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"Saved {filename}: {file_size_mb:.2f} MB")
    print(f"  Shape: {data.shape}")
    print(f"  Total elements: {data_flat.size:,}")
    
    return file_size_mb

def create_subset_file(input_file, output_file, target_size_mb=200):
    """Create a subset file of specified size"""
    # Calculate number of float32 values for target size
    bytes_per_float = 4
    target_bytes = target_size_mb * 1024 * 1024
    num_floats = target_bytes // bytes_per_float
    
    print(f"\nCreating {target_size_mb}MB subset...")
    
    # Read the subset
    data = np.fromfile(input_file, dtype=np.float32, count=num_floats)
    
    # Save subset
    data.tofile(output_file)
    
    actual_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Saved {output_file}: {actual_size_mb:.2f} MB")
    print(f"  Elements: {data.size:,}")

def main():
    print("=" * 70)
    print("Synthetic Isotropic Turbulence Data Generator")
    print("=" * 70)
    print()
    print("NOTE: This generates synthetic turbulence data with Kolmogorov")
    print("spectral characteristics, mimicking real isotropic turbulence.")
    print()
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    cfd_bin_path = os.path.join(output_dir, "cfd.bin")
    cfd_200m_path = os.path.join(output_dir, "cfd_200m.bin")
    
    try:
        # Generate turbulence data
        print("Starting data generation...")
        data = generate_isotropic_turbulence_data()
        
        # Save full dataset
        print("\nSaving full dataset...")
        full_size = save_binary_file(data, cfd_bin_path)
        
        # Create 200MB subset
        create_subset_file(cfd_bin_path, cfd_200m_path, target_size_mb=200)
        
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print(f"  Full dataset: {cfd_bin_path}")
        print(f"  Training subset: {cfd_200m_path}")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Install required packages: pip install numpy scipy")
        sys.exit(1)

if __name__ == "__main__":
    main()
