#!/usr/bin/env python3
"""
Download a HEPMC tarball from CERN EOS, extract it, and materialize:
  - hepmc.hepmc       : full binary byte stream of the extracted HEPMC file
  - hepmc_200m.hepmc  : first 200 MiB of the HEPMC byte stream

Source:
  root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc16_13TeV/HEPMC.43646133._000001.tar.gz.1

This script streams the download over HTTPS (with xrdcp fallback if available),
extracts the tar.gz safely, locates the contained HEPMC file, and writes the outputs
in the current directory (i.e., experiments/hepmc_experiment).

Usage:
  python download.py

Notes:
  - 200 MiB means 200 * 1024 * 1024 bytes. Adjust BYTES_200M if you need decimal MB.
  - If the outputs already exist, they won't be overwritten unless --force is used.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import requests
from tqdm import tqdm


ROOT_URL = (
	"root://eospublic.cern.ch//eos/opendata/atlas/rucio/mc16_13TeV/HEPMC.43646133._000001.tar.gz.1"
)
BYTES_200M = 200 * 1024 * 1024  # 200 MiB


def root_to_https(root_url: str) -> str:
	"""Convert an xrootd URL to an HTTPS URL for eospublic.cern.ch.

	Example:
	  root://eospublic.cern.ch//eos/opendata/... -> https://eospublic.cern.ch/eos/opendata/...
	"""
	prefix = "root://eospublic.cern.ch/"
	if not root_url.startswith(prefix):
		return root_url  # fallback: return as-is
	path = root_url[len(prefix) :]
	# Ensure single leading slash after host in https
	if not path.startswith("/"):
		path = "/" + path
	return "https://eospublic.cern.ch" + path


def has_xrdcp() -> bool:
	try:
		subprocess.run(["xrdcp", "--version"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		return True
	except FileNotFoundError:
		return False


def download_file(url: str, dest: Path, force: bool = False) -> Path:
	"""Download a file from HTTPS or via xrdcp fallback into dest path.

	Returns the path to the downloaded file.
	"""
	dest.parent.mkdir(parents=True, exist_ok=True)
	if dest.exists() and not force:
		print(f"File already exists: {dest} (use --force to re-download)")
		return dest

	https_url = root_to_https(url)
	print(f"Downloading from: {https_url}")

	try:
		with requests.get(https_url, stream=True, timeout=60) as r:
			r.raise_for_status()
			total = int(r.headers.get("Content-Length", 0))
			with open(dest, "wb") as f, tqdm(
				total=total if total > 0 else None,
				unit="B",
				unit_scale=True,
				desc=f"Downloading {dest.name}",
			) as pbar:
				for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MiB chunks
					if chunk:
						f.write(chunk)
						if total > 0:
							pbar.update(len(chunk))
		return dest
	except Exception as https_err:
		print(f"HTTPS download failed: {https_err}")

	# Fallback to xrdcp if available
	if has_xrdcp():
		print("Falling back to xrdcp…")
		cmd = [
			"xrdcp",
			"-f",
			url,
			str(dest),
		]
		try:
			subprocess.run(cmd, check=True)
			return dest
		except subprocess.CalledProcessError as e:
			raise RuntimeError(f"xrdcp failed with exit code {e.returncode}") from e
	else:
		raise RuntimeError("Both HTTPS and xrdcp download attempts failed.")


def safe_extract_tar(tar_path: Path, extract_dir: Path) -> None:
	"""Safely extract a tar(.gz) archive to extract_dir, preventing path traversal."""
	extract_dir.mkdir(parents=True, exist_ok=True)

	def is_within_directory(directory: Path, target: Path) -> bool:
		try:
			directory = directory.resolve(strict=False)
			target = target.resolve(strict=False)
		except Exception:
			# resolve(strict=False) can still raise in some edge cases; fallback to simple check
			pass
		return os.path.commonpath([str(directory), str(target)]) == str(directory)

	with tarfile.open(tar_path, mode="r:*") as tf:
		members = tf.getmembers()
		for m in members:
			target_path = extract_dir / m.name
			if not is_within_directory(extract_dir, target_path):
				raise RuntimeError(f"Unsafe path in tar archive: {m.name}")
		tf.extractall(path=extract_dir)


def find_hepmc_file(search_dir: Path) -> Path:
	"""Find a HEPMC payload file after extraction.

	Preference order:
	  1) First file with .hepmc (case-insensitive)
	  2) First file with .hepmc.gz (will be gunzipped on-the-fly)
	  3) If none found, choose the largest regular file
	"""
	hepmc_candidates = []
	gz_candidates = []
	other_files = []
	for root, _dirs, files in os.walk(search_dir):
		for fname in files:
			p = Path(root) / fname
			if p.is_file():
				lname = fname.lower()
				if lname.endswith(".hepmc"):
					hepmc_candidates.append(p)
				elif lname.endswith(".hepmc.gz"):
					gz_candidates.append(p)
				else:
					other_files.append(p)

	if hepmc_candidates:
		return hepmc_candidates[0]
	if gz_candidates:
		# On-the-fly gunzip into a temp file
		import gzip

		gz_path = gz_candidates[0]
		tmp_out = gz_path.with_suffix("")  # drop .gz
		with gzip.open(gz_path, "rb") as fin, open(tmp_out, "wb") as fout:
			shutil.copyfileobj(fin, fout)
		return tmp_out

	if other_files:
		# Choose the largest file as a fallback heuristic
		return max(other_files, key=lambda p: p.stat().st_size)

	raise FileNotFoundError("No files found in the extracted archive")


def write_truncated_copy(src: Path, dest: Path, limit_bytes: int) -> None:
	"""Write the first limit_bytes from src into dest."""
	with open(src, "rb") as fin, open(dest, "wb") as fout:
		remaining = limit_bytes
		bufsize = 4 * 1024 * 1024  # 4 MiB chunks
		total_written = 0
		while remaining > 0:
			to_read = min(bufsize, remaining)
			chunk = fin.read(to_read)
			if not chunk:
				break
			fout.write(chunk)
			total_written += len(chunk)
			remaining -= len(chunk)
	size = dest.stat().st_size if dest.exists() else 0
	if size < limit_bytes:
		print(
			f"Warning: source smaller than {limit_bytes} bytes. Wrote only {size} bytes to {dest.name}."
		)


def main(argv: Optional[list[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Download and prepare HEPMC files")
	parser.add_argument("--force", action="store_true", help="Re-download and overwrite outputs if present")
	parser.add_argument(
		"--url",
		default=ROOT_URL,
		help="Source URL (root:// or https://). Default is the ATLAS Open Data sample.",
	)
	args = parser.parse_args(argv)

	workdir = Path(__file__).parent.resolve()
	tar_dest = workdir / Path(args.url.split("/")[-1])  # keep original filename
	full_out = workdir / "hepmc.hepmc"
	trunc_out = workdir / "hepmc_200m.hepmc"

	# Step 1: Download
	downloaded = download_file(args.url, tar_dest, force=args.force)
	print(f"Downloaded to: {downloaded}")

	# Step 2: Extract safely to temp dir
	with tempfile.TemporaryDirectory(prefix="hepmc_extract_") as tmpdir:
		tmpdir_p = Path(tmpdir)
		print(f"Extracting archive to: {tmpdir_p}")
		safe_extract_tar(downloaded, tmpdir_p)

		# Step 3: Locate HEPMC payload
		hepmc_payload = find_hepmc_file(tmpdir_p)
		print(f"Found HEPMC payload: {hepmc_payload}")

		# Step 4: Copy full payload to hepmc.hepmc
		if full_out.exists() and not args.force:
			print(f"Skipping write; already exists: {full_out} (use --force to overwrite)")
		else:
			shutil.copyfile(hepmc_payload, full_out)
			print(f"Wrote full payload to: {full_out}")

		# Step 5: Write truncated 200 MiB copy
		if trunc_out.exists() and not args.force:
			print(f"Skipping write; already exists: {trunc_out} (use --force to overwrite)")
		else:
			write_truncated_copy(full_out, trunc_out, BYTES_200M)
			print(f"Wrote 200 MiB truncated copy to: {trunc_out}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

