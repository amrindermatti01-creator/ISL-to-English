import os
from pathlib import Path

root = Path(r"C:\Users\hp\Real ISL Camera dataset")
print("Exploring dataset at:", root)
for sub in sorted(os.listdir(root)):
    subpath = root / sub
    if subpath.is_dir():
        items = list(subpath.iterdir())
        subdirs = [f for f in items if f.is_dir()]
        direct_files = [f for f in items if f.is_file()]
        print(f"\n[{sub}]: {len(subdirs)} subdirectories, {len(direct_files)} direct files")
        if subdirs:
            total_nested = sum(len(list(d.glob("*.*"))) for d in subdirs)
            print(f"  Total images across subdirectories: {total_nested}")
            classes = sorted([d.name for d in subdirs])
            print(f"  Classes ({len(classes)}): {classes[:10]} ... {classes[-5:]}")
            # print sample image counts
            sample_counts = {d.name: len(list(d.glob("*.*"))) for d in subdirs[:5]}
            print(f"  Sample counts per class: {sample_counts}")
        elif direct_files:
            print(f"  Direct files sample: {[f.name for f in direct_files[:5]]}")
