#!/usr/bin/env python
"""One-time script to convert DPABI .mat label files to JSON."""

import json
from pathlib import Path

import scipy.io as sio

TEMPLATES = Path(
    "/Users/jiaxiangli/neuroimaging/mritools/DPABI_V9.0_250415/Templates"
)

CONVERSIONS = [
    (TEMPLATES / "aal_Labels.mat", "aal_labels.json"),
    (
        TEMPLATES / "HarvardOxford-cort-maxprob-thr25-2mm_YCG_Labels.mat",
        "ho_cort_labels.json",
    ),
    (
        TEMPLATES / "HarvardOxford-sub-maxprob-thr25-2mm_YCG_Labels.mat",
        "ho_sub_labels.json",
    ),
]

OUT_DIR = Path(__file__).resolve().parent.parent / "src" / "grouvox" / "atlases"


def convert(mat_path: Path, out_name: str) -> None:
    mat = sio.loadmat(str(mat_path))
    ref = mat["Reference"]
    labels: dict[str, str] = {}
    for i in range(ref.shape[0]):
        name = str(ref[i, 0].flat[0])
        idx = int(ref[i, 1].flat[0])
        if idx == 0:
            continue
        labels[str(idx)] = name
    out_path = OUT_DIR / out_name
    out_path.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n")
    print(f"  {out_path.name}: {len(labels)} labels")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mat_path, out_name in CONVERSIONS:
        convert(mat_path, out_name)
    print("Done.")
