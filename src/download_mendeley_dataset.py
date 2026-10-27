"""
src/download_mendeley_dataset.py - Step 1: High-Speed Selective Range Downloader & Extractor
Dataset: "Static gestures of Indian Sign Language (ISL) for English Alphabet, Hindi Vowels and Numerals"
URL: https://data.mendeley.com/datasets/7tsw22y96w/1

Features:
  1. Selective HTTP Range extraction: Downloads ONLY the blocks containing Adults & Teenagers
     English Alphabets ('A'-'Z') and Numerals ('0'-'9').
  2. STRICT Filter: 100% of Kids/Child gestures (23,056 files) and Hindi Vowels are skipped.
  3. Resumable chunked multi-threaded streaming (8 workers, 16 MB chunks).
  4. In-flight Deflate decompression (zlib) directly extracting verified JPEGs to data/mendeley_isl/<Class>/.
  5. Class canonicalization: Maps variant folders E1 and E2 into standard class 'E', and 9a/9b into '9'.
"""

import os
import sys
import time
import json
import zlib
import struct
import argparse
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DOWNLOAD_URL = "https://data.mendeley.com/public-api/zip/7tsw22y96w/download/1"

BLOCKS = [
    # Alphabets (A-Z)
    {
        "id": 1,
        "name": "Teenagers_Half_Sleeves_Alphabet",
        "start": 0,
        "end": 1094028987,
        "size_mb": 1043.35,
        "category": "alphabets"
    },
    {
        "id": 2,
        "name": "Teenagers_Full_Sleeves_Alphabet",
        "start": 2185009220,
        "end": 3255863369,
        "size_mb": 1021.25,
        "category": "alphabets"
    },
    {
        "id": 3,
        "name": "Adults_Half_Sleeves_Alphabet",
        "start": 5333900061,
        "end": 5794102301,
        "size_mb": 438.88,
        "category": "alphabets"
    },
    {
        "id": 4,
        "name": "Adults_Full_Sleeves_Alphabet",
        "start": 6069332173,
        "end": 6518846750,
        "size_mb": 428.69,
        "category": "alphabets"
    },
    # Numerals (0-9)
    {
        "id": 5,
        "name": "Teenagers_Half_Sleeves_Numerals",
        "start": 1094029029,
        "end": 1543980399,
        "size_mb": 429.11,
        "category": "numerals"
    },
    {
        "id": 6,
        "name": "Teenagers_Full_Sleeves_Numerals",
        "start": 1896137129,
        "end": 2185009183,
        "size_mb": 275.49,
        "category": "numerals"
    },
    {
        "id": 7,
        "name": "Adults_Half_Sleeves_Numerals",
        "start": 5184602442,
        "end": 5333900030,
        "size_mb": 142.38,
        "category": "numerals"
    },
    {
        "id": 8,
        "name": "Adults_Full_Sleeves_Numerals",
        "start": 5901580429,
        "end": 6069332143,
        "size_mb": 159.98,
        "category": "numerals"
    }
]


def download_chunk(url, start_byte, end_byte, chunk_idx, temp_dir, retries=5):
    """Downloads a single chunk of bytes to a temporary chunk file."""
    chunk_file = temp_dir / f"chunk_{chunk_idx:05d}.part"
    expected_size = end_byte - start_byte + 1
    
    if chunk_file.exists() and chunk_file.stat().st_size == expected_size:
        return chunk_idx, expected_size

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "Range": f"bytes={start_byte}-{end_byte}"
                }
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                if len(data) != expected_size:
                    raise IOError(f"Truncated chunk: expected {expected_size}, got {len(data)}")
                with open(chunk_file, "wb") as f:
                    f.write(data)
                return chunk_idx, len(data)
        except Exception as e:
            if attempt == retries:
                raise RuntimeError(f"Failed chunk {chunk_idx} ({start_byte}-{end_byte}) after {retries} retries: {e}")
            time.sleep(2 * attempt)


def download_block_file(block_info, out_file: Path, chunk_size=16*1024*1024, max_workers=8):
    """Downloads a contiguous block into out_file using parallel range chunks."""
    b_id = block_info["id"]
    start = block_info["start"]
    end = block_info["end"]
    total_bytes = end - start + 1
    
    if out_file.exists() and out_file.stat().st_size == total_bytes:
        print(f"  [CACHE] Block {b_id} already completely downloaded ({out_file.stat().st_size / (1024*1024):.2f} MB).")
        return out_file

    temp_chunks_dir = out_file.parent / f"block_{b_id}_chunks"
    temp_chunks_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    c_start = start
    c_idx = 0
    while c_start <= end:
        c_end = min(c_start + chunk_size - 1, end)
        chunks.append((c_start, c_end, c_idx))
        c_start = c_end + 1
        c_idx += 1

    print(f"\n>>> Downloading Block {b_id}: {block_info['name']} ({total_bytes / (1024*1024):.2f} MB in {len(chunks)} chunks)...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_chunk, DOWNLOAD_URL, cs, ce, ci, temp_chunks_dir): ci
            for cs, ce, ci in chunks
        }
        with tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"Block {b_id}") as pbar:
            already_downloaded = sum(
                (temp_chunks_dir / f"chunk_{ci:05d}.part").stat().st_size
                for _, _, ci in chunks
                if (temp_chunks_dir / f"chunk_{ci:05d}.part").exists()
            )
            pbar.update(already_downloaded)
            
            for future in as_completed(futures):
                ci, sz = future.result()
                if not (temp_chunks_dir / f"chunk_{ci:05d}.part").exists() or sz != chunk_size:
                    pbar.update(sz)

    print(f"  [ASSEMBLING] Merging {len(chunks)} chunks into {out_file.name}...")
    with open(out_file, "wb") as f_out:
        for cs, ce, ci in chunks:
            c_file = temp_chunks_dir / f"chunk_{ci:05d}.part"
            with open(c_file, "rb") as f_in:
                while True:
                    buf = f_in.read(1024 * 1024)
                    if not buf:
                        break
                    f_out.write(buf)
            c_file.unlink(missing_ok=True)
            
    temp_chunks_dir.rmdir()
    print(f"  [OK] Block {b_id} assembly complete ({out_file.stat().st_size / (1024*1024):.2f} MB).")
    return out_file


def extract_block_entries(block_file: Path, block_info, entries_for_block, out_root: Path):
    """Extracts all target images from the downloaded block file."""
    b_id = block_info["id"]
    block_start = block_info["start"]
    print(f"\n>>> Extracting {len(entries_for_block)} files from Block {b_id} into {out_root}...")

    extracted_count = 0
    corrupt_count = 0

    with open(block_file, "rb") as f_block:
        for entry in tqdm(entries_for_block, desc=f"Extracting Block {b_id}"):
            local_offset = entry["offset"] - block_start
            f_block.seek(local_offset)
            hdr_bytes = f_block.read(30)
            if len(hdr_bytes) < 30:
                corrupt_count += 1
                continue
                
            sig = struct.unpack("<I", hdr_bytes[0:4])[0]
            if sig != 0x04034b50:
                corrupt_count += 1
                continue
                
            fn_len, extra_len = struct.unpack("<HH", hdr_bytes[26:30])
            comp_start = local_offset + 30 + fn_len + extra_len
            f_block.seek(comp_start)
            comp_bytes = f_block.read(entry["comp_size"])
            
            try:
                decomp_bytes = zlib.decompress(comp_bytes, -15)
            except Exception:
                corrupt_count += 1
                continue
                
            if not decomp_bytes.startswith(b"\xff\xd8\xff"):
                corrupt_count += 1
                continue
                
            class_name = entry["class"]
            class_dir = out_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            sleeves = "Half" if "half" in entry["name"].lower() else "Full"
            raw_fname = Path(entry["name"]).name
            clean_fname = f"{entry['age']}_{sleeves}Sleeves_{entry['raw_class']}_{raw_fname}"
            dest_path = class_dir / clean_fname
            
            with open(dest_path, "wb") as f_out:
                f_out.write(decomp_bytes)
            extracted_count += 1

    print(f"  [COMPLETE] Block {b_id}: {extracted_count} images extracted, {corrupt_count} corrupt.")
    return extracted_count, corrupt_count


def main():
    parser = argparse.ArgumentParser(description="Download and extract Mendeley ISL dataset (Adults & Teenagers only).")
    parser.add_argument("--blocks", type=str, default="5,6,7,8", help="Comma-separated block IDs to download (e.g. 5,6,7,8 for numerals)")
    parser.add_argument("--workers", type=int, default=8, help="Number of download threads")
    parser.add_argument("--downloads-dir", type=str, default="C:/Users/hp/ISL_downloads" if Path("C:/Users/hp/ISL_downloads").exists() else "data/downloads", help="Directory for temporary and index download files")
    parser.add_argument("--output-dir", type=str, default="data/mendeley_isl", help="Output directory for extracted images")
    parser.add_argument("--keep-blocks", action="store_true", help="Keep downloaded block .bin files after extraction")
    args = parser.parse_args()

    out_root = (PROJECT_ROOT / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    downloads_dir = Path(args.downloads_dir).resolve()
    downloads_dir.mkdir(parents=True, exist_ok=True)

    # Load alphabet entries
    alpha_file = downloads_dir / "mendeley_target_entries.json"
    alpha_entries = []
    if alpha_file.exists():
        with open(alpha_file, "r", encoding="utf-8") as f:
            alpha_entries = json.load(f)

    # Load numeral entries
    num_file = downloads_dir / "mendeley_numeral_target_entries.json"
    num_entries = []
    if num_file.exists():
        with open(num_file, "r", encoding="utf-8") as f:
            num_entries = json.load(f)

    all_target_entries = alpha_entries + num_entries

    block_id_list = [int(x.strip()) for x in args.blocks.split(",")]
    
    print("=" * 75)
    print("MENDELEY ISL SELECTIVE DOWNLOADER & EXTRACTOR")
    print(f"Target Directory: {out_root}")
    print(f"Target Entries: {len(all_target_entries)} (Alphabets: {len(alpha_entries)}, Numerals: {len(num_entries)})")
    print(f"Selected Blocks: {block_id_list}")
    print("=" * 75)

    grand_total_extracted = 0

    for block_info in BLOCKS:
        b_id = block_info["id"]
        if b_id not in block_id_list:
            continue

        entries_source = num_entries if block_info["category"] == "numerals" else alpha_entries
        b_entries = [
            e for e in entries_source
            if block_info["start"] <= e["offset"] <= block_info["end"]
        ]
        print(f"\nProcessing Block {b_id} ({block_info['name']}): {len(b_entries)} target images.")

        block_bin = downloads_dir / f"mendeley_block_{b_id}.bin"
        download_block_file(block_info, block_bin, max_workers=args.workers)
        extracted, corrupt = extract_block_entries(block_bin, block_info, b_entries, out_root)
        grand_total_extracted += extracted

        if not args.keep_blocks:
            print(f"  [CLEANUP] Removing block cache: {block_bin.name}")
            block_bin.unlink(missing_ok=True)

    print("\n" + "=" * 75)
    print(f"DOWNLOAD & EXTRACTION COMPLETE!")
    print(f"Total Images Extracted: {grand_total_extracted:,} images.")
    print("=" * 75)


if __name__ == "__main__":
    main()
