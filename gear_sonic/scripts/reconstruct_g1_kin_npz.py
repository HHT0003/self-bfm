#!/usr/bin/env python3
"""Reconstruct one G1 kinematic NPZ with a trained SONIC g1_kin checkpoint.

The checkpoint contains the model architecture and feature normalizer, so no
training-time configuration is needed. All source NPZ fields are preserved;
only ``joint_pos`` and ``joint_vel`` are replaced by reconstructed values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_g1_kin_offline import reconstruct_command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to best.pt or another SONIC checkpoint.")
    parser.add_argument("--input_npz", type=Path, required=True, help="Source G1 motion NPZ.")
    parser.add_argument("--output_npz", type=Path, required=True, help="Output reconstructed NPZ.")
    parser.add_argument("--output_metrics_json", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cpu", help="Inference device; default is cpu.")
    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=0)
    parser.add_argument("--window_stride", type=int, default=0)
    parser.add_argument("--no_cover_tail", action="store_true")
    args = parser.parse_args()

    if not args.ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.ckpt}")
    if not args.input_npz.is_file():
        raise FileNotFoundError(f"Input NPZ does not exist: {args.input_npz}")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    reconstruct_command(args)


if __name__ == "__main__":
    main()
