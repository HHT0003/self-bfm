#!/usr/bin/env python3
"""Play a reconstructed G1 motion NPZ in MuJoCo.

The NPZ is expected to contain ``joint_pos`` and either ``body_pos_w`` /
``body_quat_w`` or ``base_pos_w`` / ``base_quat_w``.  The viewer is passive,
so the camera can be freely rotated, panned, and zoomed with the mouse.
Keyboard controls: Space pause/resume, R reset to frame 0, . next frame, ,
previous frame.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ISAAC_TO_MUJOCO = np.asarray(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, required=True, help="Reconstructed motion NPZ.")
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path(__file__).resolve().parent / "g1" / "scene_29dof.xml",
        help="MuJoCo scene XML (robot plus floor/lighting).",
    )
    parser.add_argument("--fps", type=float, default=0.0, help="Override NPZ fps; 0 uses NPZ fps (default 50).")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--no_loop", action="store_true", help="Stop at the final frame instead of looping.")
    return parser.parse_args()


def load_motion(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=True) as data:
        if "joint_pos" not in data.files:
            raise KeyError(f"{path} is missing joint_pos")
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        if joint_pos.ndim != 2 or joint_pos.shape[1] != 29:
            raise ValueError(f"joint_pos must have shape (T,29), got {joint_pos.shape}")
        if "body_pos_w" in data.files:
            root_pos = np.asarray(data["body_pos_w"], dtype=np.float64)[:, 0]
        elif "base_pos_w" in data.files:
            root_pos = np.asarray(data["base_pos_w"], dtype=np.float64)
        else:
            raise KeyError("NPZ needs body_pos_w or base_pos_w")
        if "body_quat_w" in data.files:
            root_quat = np.asarray(data["body_quat_w"], dtype=np.float64)[:, 0]
        elif "base_quat_w" in data.files:
            root_quat = np.asarray(data["base_quat_w"], dtype=np.float64)
        else:
            raise KeyError("NPZ needs body_quat_w or base_quat_w")
        fps = float(np.asarray(data["fps"]).item()) if "fps" in data.files else 50.0

    if root_pos.shape != (joint_pos.shape[0], 3) or root_quat.shape != (joint_pos.shape[0], 4):
        raise ValueError(f"Root pose shapes are incompatible: {root_pos.shape}, {root_quat.shape}")
    norms = np.linalg.norm(root_quat, axis=1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError("Root quaternion contains a zero-norm frame")
    root_quat = root_quat / norms
    return joint_pos, root_pos, root_quat, fps


def check_mesh_assets(xml_path: Path) -> None:
    """Fail early with an actionable message when Git LFS meshes are absent."""
    mesh_dir = xml_path.parent / "meshes"
    if not mesh_dir.is_dir():
        return
    pointers = []
    for mesh_path in mesh_dir.iterdir():
        if not mesh_path.is_file() or mesh_path.stat().st_size > 1024:
            continue
        try:
            header = mesh_path.read_text(errors="ignore")[:64]
        except OSError:
            continue
        if header.startswith("version https://git-lfs.github.com/spec/v1"):
            pointers.append(mesh_path.name)
    if pointers:
        preview = ", ".join(pointers[:3])
        more = " ..." if len(pointers) > 3 else ""
        raise RuntimeError(
            f"MuJoCo mesh files are Git LFS pointers ({len(pointers)} found: {preview}{more}). "
            "Install Git LFS and run `git lfs pull`, or restore the actual STL files."
        )


def resolve_scene_xml(xml_path: Path) -> Path:
    """Use the bundled scene wrapper when the bare legacy robot XML is passed."""
    if xml_path.name == "g1_29dof_old.xml":
        scene_path = xml_path.with_name("scene_29dof.xml")
        if scene_path.is_file():
            print(f"[INFO] Using scene XML with floor and lighting: {scene_path}")
            return scene_path
    return xml_path


def keyboard_callback(key: int) -> None:
    # State is attached to the callback function by main().
    if key == ord(" "):
        keyboard_callback.paused = not keyboard_callback.paused
    elif key in (ord("r"), ord("R")):
        keyboard_callback.frame = 0
    elif key == ord("."):
        keyboard_callback.frame += 1
    elif key == ord(","):
        keyboard_callback.frame = max(0, keyboard_callback.frame - 1)


keyboard_callback.paused = False
keyboard_callback.frame = 0


def main() -> None:
    args = parse_args()
    if not args.npz.is_file():
        raise FileNotFoundError(args.npz)
    if not args.xml.is_file():
        raise FileNotFoundError(args.xml)

    args.xml = resolve_scene_xml(args.xml)
    check_mesh_assets(args.xml)
    joint_pos, root_pos, root_quat, npz_fps = load_motion(args.npz)
    fps = float(args.fps) if args.fps > 0 else npz_fps
    if fps <= 0:
        raise ValueError(f"Invalid fps: {fps}")
    keyboard_callback.frame = max(0, min(args.start_frame, joint_pos.shape[0] - 1))

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    if model.nq < 36:
        raise ValueError(f"Expected a free-base 29-DOF G1 model with nq>=36, got nq={model.nq}")

    dt = 1.0 / fps
    with mujoco.viewer.launch_passive(model, data, key_callback=keyboard_callback) as viewer:
        viewer.cam.lookat[:] = root_pos[0]
        viewer.cam.lookat[2] = max(0.7, float(root_pos[0, 2]))
        viewer.cam.distance = 3.5
        viewer.cam.azimuth = 120.0
        viewer.cam.elevation = -18.0
        while viewer.is_running():
            frame = keyboard_callback.frame
            if frame >= joint_pos.shape[0]:
                if args.no_loop:
                    frame = joint_pos.shape[0] - 1
                else:
                    frame %= joint_pos.shape[0]
                keyboard_callback.frame = frame

            data.qpos[:3] = root_pos[frame]
            data.qpos[3:7] = root_quat[frame]
            data.qpos[7:36] = joint_pos[frame, ISAAC_TO_MUJOCO]
            mujoco.mj_forward(model, data)
            viewer.sync()

            if not keyboard_callback.paused:
                keyboard_callback.frame += 1
            time.sleep(dt)


if __name__ == "__main__":
    main()
