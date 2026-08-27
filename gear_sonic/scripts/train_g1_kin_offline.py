#!/usr/bin/env python3
"""Offline G1 encoder -> FSQ -> g1_kin reconstruction trainer.

Trains only the kinematic tokenizer path used by GEAR-SONIC:

    joint_pos/joint_vel windows [B, 10, 58]
        -> G1 MLP encoder [B, 2, 32]
        -> FSQ
        -> g1_kin MLP decoder [B, 10, 58]

No Isaac Lab, no PPO, no g1_dyn, no SMPL/teleop encoders, and no robot
proprioception. NPZ files must contain ``joint_pos[T, 29]`` and
``joint_vel[T, 29]``. Extra keys (box, ball, contact, ...) are ignored.

Pass one or more ``--data_dir`` roots; ``*.npz`` is collected recursively so
nested layouts such as ``lafan1_g1/g1_mimic_npz`` work.

Default sampling matches SONIC's G1 future-reference: 10 frames at 50 Hz with
``frame_stride=5`` (0.1 s between frames).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from vector_quantize_pytorch import FSQ as VQFSQ
except ImportError:  # pragma: no cover
    VQFSQ = None

JOINT_DIM = 29
FEATURE_DIM = JOINT_DIM * 2  # pos + vel
REQUIRED_KEYS = ("joint_pos", "joint_vel")


def parse_int_list(text: str, *, name: str) -> list[int]:
    values = [s.strip() for s in str(text).split(",") if s.strip()]
    if not values:
        raise ValueError(f"{name} cannot be empty.")
    out: list[int] = []
    for raw in values:
        value = int(raw)
        if value <= 0:
            raise ValueError(f"{name} values must be > 0. Got {value}.")
        out.append(value)
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def window_span(seq_len: int, frame_stride: int) -> int:
    return 1 + (int(seq_len) - 1) * int(frame_stride)


def generate_window_starts(length: int, span: int, window_stride: int, cover_tail: bool) -> list[int]:
    if length < span:
        return []
    starts = list(range(0, length - span + 1, window_stride))
    last_needed = length - span
    if cover_tail and starts and starts[-1] != last_needed:
        starts.append(last_needed)
    elif not starts and last_needed >= 0:
        starts = [0]
    return starts


def split_file_indices(num_files: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    all_idx = list(range(num_files))
    if num_files <= 1:
        return all_idx, []
    rng = random.Random(seed)
    rng.shuffle(all_idx)
    n_val = max(1, int(round(num_files * val_ratio)))
    n_val = min(n_val, num_files - 1)
    return sorted(all_idx[n_val:]), sorted(all_idx[:n_val])


def load_pos_vel(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data.files]
        if missing:
            raise KeyError(f"{path} missing keys: {missing}")
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != JOINT_DIM:
        raise ValueError(f"{path}: joint_pos must be [T, {JOINT_DIM}], got {joint_pos.shape}")
    if joint_vel.shape != joint_pos.shape:
        raise ValueError(f"{path}: joint_vel shape {joint_vel.shape} != joint_pos {joint_pos.shape}")
    return joint_pos, joint_vel


def slice_window(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    start: int,
    seq_len: int,
    frame_stride: int,
) -> np.ndarray:
    indices = start + np.arange(seq_len) * frame_stride
    pos = joint_pos[indices]
    vel = joint_vel[indices]
    return np.concatenate([pos, vel], axis=1).astype(np.float32)


@dataclass
class FileIndex:
    path: Path
    length: int


class FeatureNormalizer(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.from_numpy(np.asarray(mean, dtype=np.float32)))
        self.register_buffer("std", torch.from_numpy(np.asarray(std, dtype=np.float32)))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def to_numpy(self) -> dict[str, np.ndarray]:
        return {
            "mean": self.mean.detach().cpu().numpy(),
            "std": self.std.detach().cpu().numpy(),
        }


class LazyWindowDataset(Dataset):
    """Load NPZ windows on demand so the full BONES-SEED corpus is not stored in RAM."""

    def __init__(
        self,
        files: list[FileIndex],
        window_refs: list[tuple[int, int]],
        seq_len: int,
        frame_stride: int,
    ):
        self.files = files
        self.window_refs = window_refs
        self.seq_len = int(seq_len)
        self.frame_stride = int(frame_stride)
        self._cache_key: str | None = None
        self._cache_pos: np.ndarray | None = None
        self._cache_vel: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.window_refs)

    def _load_file(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        key = str(path)
        if self._cache_key == key and self._cache_pos is not None and self._cache_vel is not None:
            return self._cache_pos, self._cache_vel
        joint_pos, joint_vel = load_pos_vel(path)
        self._cache_key = key
        self._cache_pos = joint_pos
        self._cache_vel = joint_vel
        return joint_pos, joint_vel

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, start = self.window_refs[idx]
        joint_pos, joint_vel = self._load_file(self.files[file_idx].path)
        window = slice_window(
            joint_pos,
            joint_vel,
            start=start,
            seq_len=self.seq_len,
            frame_stride=self.frame_stride,
        )
        return torch.from_numpy(window)


def collect_npz_paths(
    data_dirs: list[Path],
    *,
    recursive: bool,
    max_files: int | None,
) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for data_dir in data_dirs:
        if not data_dir.is_dir():
            raise FileNotFoundError(f"data dir does not exist: {data_dir}")
        iterator = data_dir.rglob("*.npz") if recursive else data_dir.glob("*.npz")
        dir_count = 0
        for path in sorted(iterator):
            if not path.is_file():
                continue
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
            dir_count += 1
        print(f"[INFO] {data_dir}: {dir_count} npz files", flush=True)
    if max_files is not None:
        found = found[: max(0, int(max_files))]
        print(f"[INFO] max_files cap applied: {len(found)} npz files", flush=True)
    if not found:
        raise FileNotFoundError(f"No .npz files found in: {[str(p) for p in data_dirs]}")
    return found


def index_npz_dirs(
    data_dirs: list[Path],
    seq_len: int,
    frame_stride: int,
    window_stride: int,
    cover_tail: bool,
    max_files: int | None,
    recursive: bool = True,
) -> tuple[list[FileIndex], np.ndarray, np.ndarray]:
    npz_paths = collect_npz_paths(data_dirs, recursive=recursive, max_files=max_files)
    span = window_span(seq_len, frame_stride)
    files: list[FileIndex] = []
    sum_vec = np.zeros(FEATURE_DIM, dtype=np.float64)
    sq_sum_vec = np.zeros(FEATURE_DIM, dtype=np.float64)
    count = 0
    skipped = 0

    for path in tqdm(npz_paths, desc="Indexing NPZ files"):
        try:
            joint_pos, joint_vel = load_pos_vel(path)
        except (KeyError, ValueError, OSError) as exc:
            skipped += 1
            print(f"[WARN] skip {path}: {exc}", flush=True)
            continue
        length = int(joint_pos.shape[0])
        if length < span:
            skipped += 1
            continue
        feat = np.concatenate([joint_pos, joint_vel], axis=1).astype(np.float64)
        sum_vec += feat.sum(axis=0)
        sq_sum_vec += np.square(feat).sum(axis=0)
        count += feat.shape[0]
        files.append(FileIndex(path=path, length=length))

    print(
        f"[INFO] indexed {len(files)} usable files "
        f"(skipped={skipped}, min_length={span}, frames={count})",
        flush=True,
    )
    if not files:
        raise RuntimeError(
            f"No usable motions with length >= {span} in {[str(p) for p in data_dirs]}"
        )
    if count <= 0:
        raise RuntimeError("No frames available to fit feature statistics.")

    mean = (sum_vec / count).astype(np.float32)
    var = np.maximum((sq_sum_vec / count) - np.square(mean), 1e-8)
    std = np.maximum(np.sqrt(var).astype(np.float32), 1e-6)
    return files, mean, std


def build_window_refs(
    files: list[FileIndex],
    file_indices: list[int],
    seq_len: int,
    frame_stride: int,
    window_stride: int,
    cover_tail: bool,
) -> list[tuple[int, int]]:
    span = window_span(seq_len, frame_stride)
    refs: list[tuple[int, int]] = []
    for file_idx in file_indices:
        starts = generate_window_starts(
            files[file_idx].length,
            span=span,
            window_stride=window_stride,
            cover_tail=cover_tail,
        )
        refs.extend((file_idx, start) for start in starts)
    return refs


class TemporalMLP(nn.Module):
    """MLP with SONIC BaseModule temporal flatten/reshape behavior.

    ``BaseModule`` cannot be imported here without pulling Isaac/torchvision.
    This copies the MLP path in ``gear_sonic.trl.modules.base_module.BaseModule``:
    flatten ``[B, T_in, F_in]`` -> MLP -> reshape ``[B, T_out, F_out]``.
    """

    def __init__(
        self,
        *,
        feature_in: int,
        feature_out: int,
        hidden_dims: list[int],
        num_input_temporal_dims: int,
        num_output_temporal_dims: int,
        activation: str = "SiLU",
    ):
        super().__init__()
        self.num_input_temporal_dims = int(num_input_temporal_dims)
        self.num_output_temporal_dims = int(num_output_temporal_dims)
        input_dim = int(feature_in) * self.num_input_temporal_dims
        output_dim = int(feature_out) * self.num_output_temporal_dims
        act_cls = getattr(nn, activation)
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), act_cls()]
        for idx, hidden in enumerate(hidden_dims):
            if idx == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden, output_dim))
            else:
                layers.append(nn.Linear(hidden, hidden_dims[idx + 1]))
                layers.append(act_cls())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"Expected at least 2D input, got {tuple(x.shape)}")
        flat = x.reshape(*x.shape[:-2], -1) if x.ndim >= 3 else x
        out = self.net(flat)
        return out.view(*out.shape[:-1], self.num_output_temporal_dims, -1)


class FallbackFSQ(nn.Module):
    """Finite Scalar Quantization with straight-through estimator.

    Used when ``vector_quantize_pytorch`` is not installed. Each last-dim
    channel is independently quantized to ``levels[i]`` values in [-1, 1].
    """

    def __init__(self, levels: list[int]):
        super().__init__()
        if not levels:
            raise ValueError("FSQ levels cannot be empty.")
        self.levels = [int(x) for x in levels]
        for i, level in enumerate(self.levels):
            if level < 2:
                raise ValueError(f"FSQ level must be >= 2. Got levels[{i}]={level}")
            centers = torch.linspace(-1.0, 1.0, level, dtype=torch.float32)
            self.register_buffer(f"centers_{i}", centers)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if z.shape[-1] != len(self.levels):
            raise ValueError(f"Expected last dim {len(self.levels)}, got {tuple(z.shape)}")
        quantized = []
        indices = []
        for dim, level in enumerate(self.levels):
            centers = getattr(self, f"centers_{dim}").to(device=z.device, dtype=z.dtype)
            zi = z[..., dim : dim + 1]
            dist = (zi - centers.view(*([1] * (zi.ndim - 1)), -1)).abs()
            idx = dist.argmin(dim=-1)
            hard = centers[idx]
            # tanh keeps the encoder in a bounded range similar to FSQ
            soft = torch.tanh(zi.squeeze(-1))
            quant = soft + (hard - soft).detach()
            quantized.append(quant)
            indices.append(idx)
        return torch.stack(quantized, dim=-1), torch.stack(indices, dim=-1)


def build_fsq(levels: list[int]) -> nn.Module:
    if VQFSQ is not None:
        return VQFSQ(levels=levels)
    print("[WARN] vector_quantize_pytorch not found; using FallbackFSQ.")
    return FallbackFSQ(levels)


class G1KinAutoEncoder(nn.Module):
    """SONIC-style G1 encoder + FSQ + g1_kin decoder, without action/policy heads."""

    def __init__(
        self,
        *,
        seq_len: int,
        feature_dim: int,
        max_num_tokens: int,
        token_dim: int,
        fsq_levels: list[int],
        enc_hidden_dims: list[int],
        dec_hidden_dims: list[int],
    ):
        super().__init__()
        if len(fsq_levels) != token_dim:
            raise ValueError(f"len(fsq_levels)={len(fsq_levels)} must equal token_dim={token_dim}")
        self.seq_len = int(seq_len)
        self.feature_dim = int(feature_dim)
        self.max_num_tokens = int(max_num_tokens)
        self.token_dim = int(token_dim)
        self.fsq_levels = [int(x) for x in fsq_levels]

        self.encoder = TemporalMLP(
            feature_in=self.feature_dim,
            feature_out=self.token_dim,
            hidden_dims=enc_hidden_dims,
            num_input_temporal_dims=self.seq_len,
            num_output_temporal_dims=self.max_num_tokens,
        )
        self.quantizer = build_fsq(self.fsq_levels)
        self.decoder = TemporalMLP(
            feature_in=self.token_dim,
            feature_out=self.feature_dim,
            hidden_dims=dec_hidden_dims,
            num_input_temporal_dims=self.max_num_tokens,
            num_output_temporal_dims=self.seq_len,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        result = self.quantizer(latent)
        if isinstance(result, tuple):
            quantized = result[0]
            indices = result[1] if len(result) > 1 else None
        else:
            quantized, indices = result, None
        return quantized, indices

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.decoder(tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantized, indices = self.encode(x)
        recon = self.decode(quantized)
        return recon, quantized, indices


class ScalarMeter:
    def __init__(self) -> None:
        self.total: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: dict[str, float], n: int) -> None:
        self.count += int(n)
        for key, value in metrics.items():
            self.total[key] = self.total.get(key, 0.0) + float(value) * n

    def average(self) -> dict[str, float]:
        if self.count <= 0:
            return {}
        return {key: value / self.count for key, value in self.total.items()}


def compute_losses(
    recon_norm: torch.Tensor,
    target_norm: torch.Tensor,
    target_raw: torch.Tensor,
    normalizer: FeatureNormalizer,
    w_pos: float,
    w_vel: float,
) -> dict[str, torch.Tensor]:
    recon_raw = normalizer.denormalize(recon_norm)
    pred_pos, pred_vel = recon_norm[..., :JOINT_DIM], recon_norm[..., JOINT_DIM:]
    tgt_pos, tgt_vel = target_norm[..., :JOINT_DIM], target_norm[..., JOINT_DIM:]
    loss_pos = F.mse_loss(pred_pos, tgt_pos)
    loss_vel = F.mse_loss(pred_vel, tgt_vel)
    total = w_pos * loss_pos + w_vel * loss_vel
    raw_mse_pos = F.mse_loss(recon_raw[..., :JOINT_DIM], target_raw[..., :JOINT_DIM])
    raw_mse_vel = F.mse_loss(recon_raw[..., JOINT_DIM:], target_raw[..., JOINT_DIM:])
    return {
        "total_loss": total,
        "loss_joint_pos": loss_pos,
        "loss_joint_vel": loss_vel,
        "raw_mse_joint_pos": raw_mse_pos,
        "raw_mse_joint_vel": raw_mse_vel,
    }


def sanitize_args(args: argparse.Namespace) -> dict[str, Any]:
    payload = {}
    for key, value in vars(args).items():
        if key == "func" or callable(value):
            continue
        payload[key] = value
    return payload


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def init_wandb_run(args: argparse.Namespace, out_dir: Path, config_payload: dict[str, Any]):
    if not args.use_wandb:
        return None
    try:
        import wandb
    except Exception as exc:
        raise ModuleNotFoundError(
            "wandb is not installed. Install it with: pip install wandb"
        ) from exc

    tags = [x.strip() for x in str(args.wandb_tags).split(",") if x.strip()]
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=tags if tags else None,
        mode=args.wandb_mode,
        dir=str(out_dir),
        config=to_serializable(config_payload),
    )


def run_epoch(
    *,
    model: G1KinAutoEncoder,
    loader: DataLoader,
    normalizer: FeatureNormalizer,
    device: torch.device,
    w_pos: float,
    w_vel: float,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    wandb_run: Any | None = None,
    step_start: int = 0,
    step_log_every: int = 0,
    step_phase: str = "train",
) -> tuple[dict[str, float], int]:
    is_train = optimizer is not None
    model.train(is_train)
    meter = ScalarMeter()
    step_count = int(step_start)
    context = torch.enable_grad() if is_train else torch.no_grad()
    iterator = tqdm(
        loader,
        desc=step_phase,
        leave=False,
        mininterval=30.0,
        disable=not is_train,
    )
    with context:
        for batch_raw in iterator:
            step_count += 1
            batch_raw = batch_raw.to(device=device, dtype=torch.float32)
            batch_norm = normalizer.normalize(batch_raw)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            recon_norm, _, _ = model(batch_norm)
            losses = compute_losses(
                recon_norm=recon_norm,
                target_norm=batch_norm,
                target_raw=batch_raw,
                normalizer=normalizer,
                w_pos=w_pos,
                w_vel=w_vel,
            )
            if is_train:
                losses["total_loss"].backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            items = {key: float(value.detach().cpu().item()) for key, value in losses.items()}
            meter.update(items, n=int(batch_raw.shape[0]))
            if is_train:
                iterator.set_postfix(loss=f"{items['total_loss']:.4f}", refresh=False)
            if is_train and step_count % 100 == 0:
                print(
                    f"[step {step_count}] {step_phase}_loss={items['total_loss']:.6f} "
                    f"pos={items['loss_joint_pos']:.6f} vel={items['loss_joint_vel']:.6f}",
                    flush=True,
                )
            if (
                is_train
                and wandb_run is not None
                and step_log_every > 0
                and (step_count % step_log_every == 0)
            ):
                wandb_run.log(
                    {
                        "train_step": int(step_count),
                        f"batch/{step_phase}_total_loss": items["total_loss"],
                        f"batch/{step_phase}_joint_pos": items["loss_joint_pos"],
                        f"batch/{step_phase}_joint_vel": items["loss_joint_vel"],
                    }
                )
    avg = meter.average()
    avg["raw_rmse_joint_pos"] = math.sqrt(max(avg.get("raw_mse_joint_pos", 0.0), 0.0))
    avg["raw_rmse_joint_vel"] = math.sqrt(max(avg.get("raw_mse_joint_vel", 0.0), 0.0))
    return avg, step_count


def build_model_from_args(args: argparse.Namespace) -> G1KinAutoEncoder:
    hidden_enc = parse_int_list(args.enc_hidden_dims, name="enc_hidden_dims")
    hidden_dec = parse_int_list(args.dec_hidden_dims, name="dec_hidden_dims")
    level_values = parse_int_list(str(args.fsq_levels), name="fsq_levels")
    if len(level_values) == 1:
        fsq_levels = level_values * int(args.token_dim)
    else:
        fsq_levels = level_values
    return G1KinAutoEncoder(
        seq_len=args.seq_len,
        feature_dim=FEATURE_DIM,
        max_num_tokens=args.max_num_tokens,
        token_dim=args.token_dim,
        fsq_levels=fsq_levels,
        enc_hidden_dims=hidden_enc,
        dec_hidden_dims=hidden_dec,
    )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def append_metrics_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    *,
    model: G1KinAutoEncoder,
    normalizer: FeatureNormalizer,
    args: argparse.Namespace,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "metrics": metrics,
            "args": sanitize_args(args),
            "model": model.state_dict(),
            "normalizer": normalizer.to_numpy(),
            "arch": {
                "seq_len": model.seq_len,
                "feature_dim": model.feature_dim,
                "max_num_tokens": model.max_num_tokens,
                "token_dim": model.token_dim,
                "fsq_levels": model.fsq_levels,
                "enc_hidden_dims": parse_int_list(args.enc_hidden_dims, name="enc_hidden_dims"),
                "dec_hidden_dims": parse_int_list(args.dec_hidden_dims, name="dec_hidden_dims"),
            },
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[G1KinAutoEncoder, FeatureNormalizer, dict[str, Any]]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    arch = payload["arch"]
    model = G1KinAutoEncoder(
        seq_len=arch["seq_len"],
        feature_dim=arch["feature_dim"],
        max_num_tokens=arch["max_num_tokens"],
        token_dim=arch["token_dim"],
        fsq_levels=arch["fsq_levels"],
        enc_hidden_dims=arch["enc_hidden_dims"],
        dec_hidden_dims=arch["dec_hidden_dims"],
    )
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    stats = payload["normalizer"]
    normalizer = FeatureNormalizer(mean=stats["mean"], std=stats["std"]).to(device)
    return model, normalizer, payload


def train_command(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    data_dirs = [Path(p) for p in args.data_dir]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files, mean, std = index_npz_dirs(
        data_dirs=data_dirs,
        seq_len=args.seq_len,
        frame_stride=args.frame_stride,
        window_stride=args.window_stride,
        cover_tail=not args.no_cover_tail,
        max_files=args.max_files,
        recursive=not args.no_recursive,
    )
    train_idx, val_idx = split_file_indices(len(files), val_ratio=args.val_ratio, seed=args.seed)
    train_refs = build_window_refs(
        files,
        train_idx,
        seq_len=args.seq_len,
        frame_stride=args.frame_stride,
        window_stride=args.window_stride,
        cover_tail=not args.no_cover_tail,
    )
    val_refs = build_window_refs(
        files,
        val_idx,
        seq_len=args.seq_len,
        frame_stride=args.frame_stride,
        window_stride=args.window_stride,
        cover_tail=not args.no_cover_tail,
    )
    if args.max_windows is not None:
        train_refs = train_refs[: args.max_windows]
        val_refs = val_refs[: max(1, args.max_windows // 10)]
    if not train_refs:
        raise RuntimeError("No training windows found. Check seq_len/frame_stride and data lengths.")

    normalizer = FeatureNormalizer(mean=mean, std=std).to(device)
    train_ds = LazyWindowDataset(files, train_refs, args.seq_len, args.frame_stride)
    val_ds = LazyWindowDataset(files, val_refs, args.seq_len, args.frame_stride)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_model_from_args(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    probe = next(iter(train_loader)).to(device)
    probe_norm = normalizer.normalize(probe)
    with torch.no_grad():
        recon, quantized, _ = model(probe_norm)
    print(
        "[INFO] shapes: "
        f"input={tuple(probe.shape)} latent={tuple(quantized.shape)} recon={tuple(recon.shape)}"
    )
    if tuple(probe.shape[1:]) != (args.seq_len, FEATURE_DIM):
        raise RuntimeError(f"Unexpected input shape {tuple(probe.shape)}")
    if tuple(quantized.shape[1:]) != (args.max_num_tokens, args.token_dim):
        raise RuntimeError(f"Unexpected token shape {tuple(quantized.shape)}")
    if tuple(recon.shape) != tuple(probe.shape):
        raise RuntimeError(f"Unexpected recon shape {tuple(recon.shape)}")

    config_payload = {
        "data_dirs": [str(p) for p in data_dirs],
        "num_files": len(files),
        "num_train_windows": len(train_refs),
        "num_val_windows": len(val_refs),
        "seq_len": args.seq_len,
        "frame_stride": args.frame_stride,
        "window_stride": args.window_stride,
        "feature_dim": FEATURE_DIM,
        "max_num_tokens": args.max_num_tokens,
        "token_dim": args.token_dim,
        "args": sanitize_args(args),
    }
    save_json(out_dir / "config.json", config_payload)
    print(
        f"[INFO] files={len(files)} train_windows={len(train_refs)} "
        f"val_windows={len(val_refs)} steps_per_epoch≈{len(train_loader)} device={device}",
        flush=True,
    )
    wandb_run = init_wandb_run(args, out_dir, config_payload)

    best_val = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        print(
            f"[INFO] start epoch {epoch}/{args.epochs} "
            f"steps≈{len(train_loader)} global_step={global_step}",
            flush=True,
        )
        train_metrics, global_step = run_epoch(
            model=model,
            loader=train_loader,
            normalizer=normalizer,
            device=device,
            w_pos=args.w_joint_pos,
            w_vel=args.w_joint_vel,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            wandb_run=wandb_run,
            step_start=global_step,
            step_log_every=args.wandb_log_every_steps if wandb_run is not None else 0,
            step_phase="train",
        )
        val_metrics = {}
        if len(val_refs) > 0:
            val_metrics, _ = run_epoch(
                model=model,
                loader=val_loader,
                normalizer=normalizer,
                device=device,
                w_pos=args.w_joint_pos,
                w_vel=args.w_joint_vel,
                optimizer=None,
                grad_clip=args.grad_clip,
            )
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        append_metrics_row(out_dir / "metrics.csv", row)
        val_loss = float(val_metrics.get("total_loss", train_metrics["total_loss"]))
        print(
            f"[epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['total_loss']:.6f} "
            f"val_loss={val_loss:.6f} "
            f"train_rmse_pos={train_metrics['raw_rmse_joint_pos']:.6f} "
            f"train_rmse_vel={train_metrics['raw_rmse_joint_vel']:.6f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_payload = {
                "epoch": epoch,
                "train/total_loss": train_metrics["total_loss"],
                "train/joint_pos": train_metrics["loss_joint_pos"],
                "train/joint_vel": train_metrics["loss_joint_vel"],
                "train/rmse_joint_pos": train_metrics["raw_rmse_joint_pos"],
                "train/rmse_joint_vel": train_metrics["raw_rmse_joint_vel"],
            }
            if val_metrics:
                wandb_payload.update(
                    {
                        "val/total_loss": val_metrics["total_loss"],
                        "val/joint_pos": val_metrics["loss_joint_pos"],
                        "val/joint_vel": val_metrics["loss_joint_vel"],
                        "val/rmse_joint_pos": val_metrics["raw_rmse_joint_pos"],
                        "val/rmse_joint_vel": val_metrics["raw_rmse_joint_vel"],
                    }
                )
            wandb_run.log(wandb_payload)
        save_checkpoint(
            out_dir / "last.pt",
            model=model,
            normalizer=normalizer,
            args=args,
            epoch=epoch,
            metrics=row,
        )
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                out_dir / "best.pt",
                model=model,
                normalizer=normalizer,
                args=args,
                epoch=epoch,
                metrics=row,
            )
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            save_checkpoint(
                out_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                normalizer=normalizer,
                args=args,
                epoch=epoch,
                metrics=row,
            )
    print(f"[INFO] training done. checkpoints in {out_dir}")
    if wandb_run is not None:
        wandb_run.finish()


def reconstruct_windows(
    model: G1KinAutoEncoder,
    normalizer: FeatureNormalizer,
    features: np.ndarray,
    seq_len: int,
    frame_stride: int,
    window_stride: int,
    cover_tail: bool,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    span = window_span(seq_len, frame_stride)
    starts = generate_window_starts(features.shape[0], span, window_stride, cover_tail)
    if not starts:
        raise ValueError(f"Motion length {features.shape[0]} is shorter than window span {span}")

    windows = []
    for start in starts:
        indices = start + np.arange(seq_len) * frame_stride
        windows.append(features[indices])
    windows_t = torch.from_numpy(np.stack(windows, axis=0)).to(device=device, dtype=torch.float32)

    recon_chunks = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, windows_t.shape[0], batch_size):
            batch = windows_t[begin : begin + batch_size]
            recon_norm, _, _ = model(normalizer.normalize(batch))
            recon_chunks.append(normalizer.denormalize(recon_norm).cpu().numpy())
    recon_windows = np.concatenate(recon_chunks, axis=0)

    acc = np.zeros_like(features, dtype=np.float64)
    cnt = np.zeros((features.shape[0], 1), dtype=np.float64)
    for i, start in enumerate(starts):
        indices = start + np.arange(seq_len) * frame_stride
        acc[indices] += recon_windows[i]
        cnt[indices] += 1.0
    cnt = np.maximum(cnt, 1.0)
    return (acc / cnt).astype(np.float32), np.asarray(starts, dtype=np.int32)


def reconstruct_command(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, normalizer, payload = load_checkpoint(Path(args.ckpt), device=device)
    arch = payload["arch"]
    seq_len = int(args.seq_len) if args.seq_len > 0 else int(arch["seq_len"])
    frame_stride = int(args.frame_stride) if args.frame_stride > 0 else int(payload["args"].get("frame_stride", 5))
    window_stride = int(args.window_stride) if args.window_stride > 0 else int(payload["args"].get("window_stride", 5))

    input_path = Path(args.input_npz)
    joint_pos, joint_vel = load_pos_vel(input_path)
    features = np.concatenate([joint_pos, joint_vel], axis=1)
    recon, starts = reconstruct_windows(
        model=model,
        normalizer=normalizer,
        features=features,
        seq_len=seq_len,
        frame_stride=frame_stride,
        window_stride=window_stride,
        cover_tail=not args.no_cover_tail,
        batch_size=args.batch_size,
        device=device,
    )
    recon_pos = recon[:, :JOINT_DIM]
    recon_vel = recon[:, JOINT_DIM:]
    pos_rmse = float(np.sqrt(np.mean(np.square(recon_pos - joint_pos))))
    vel_rmse = float(np.sqrt(np.mean(np.square(recon_vel - joint_vel))))

    with np.load(input_path, allow_pickle=True) as src:
        out_payload = {key: src[key] for key in src.files}
    out_payload["joint_pos"] = recon_pos
    out_payload["joint_vel"] = recon_vel
    out_payload["joint_pos_src"] = joint_pos
    out_payload["joint_vel_src"] = joint_vel

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **out_payload)

    metrics = {
        "input_npz": str(input_path),
        "output_npz": str(output_npz),
        "ckpt": str(args.ckpt),
        "num_windows": int(starts.shape[0]),
        "pos_rmse": pos_rmse,
        "vel_rmse": vel_rmse,
        "seq_len": seq_len,
        "frame_stride": frame_stride,
        "window_stride": window_stride,
    }
    if args.output_metrics_json:
        save_json(Path(args.output_metrics_json), metrics)
    print(json.dumps(metrics, indent=2))


def add_shared_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seq_len", type=int, default=10, help="Number of future reference frames.")
    parser.add_argument("--frame_stride", type=int, default=5, help="Frames between sampled future refs at 50 Hz.")
    parser.add_argument("--window_stride", type=int, default=5, help="Start-index stride when slicing training windows.")
    parser.add_argument("--no_cover_tail", action="store_true", help="Do not force a final tail-covering window.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline G1 encoder + FSQ + g1_kin trainer.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train encoder/FSQ/g1_kin on NPZ joint_pos/joint_vel windows.")
    p_train.add_argument(
        "--data_dir",
        type=str,
        nargs="+",
        required=True,
        help="One or more NPZ roots. Recursively collects *.npz unless --no_recursive.",
    )
    p_train.add_argument(
        "--no_recursive",
        action="store_true",
        help="Only load top-level *.npz in each --data_dir (skip nested folders).",
    )
    p_train.add_argument("--out_dir", type=str, required=True)
    add_shared_sampling_args(p_train)
    p_train.add_argument(
        "--enc_hidden_dims",
        type=str,
        default="2048,1024,512",
        help="Three-layer MLP. SONIC used four layers (2048,1024,512,512); three is enough for 580->64.",
    )
    p_train.add_argument("--dec_hidden_dims", type=str, default="2048,1024,512")
    p_train.add_argument("--max_num_tokens", type=int, default=2)
    p_train.add_argument("--token_dim", type=int, default=32)
    p_train.add_argument("--fsq_levels", type=str, default="32")
    p_train.add_argument("--batch_size", type=int, default=256)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--weight_decay", type=float, default=5e-5)
    p_train.add_argument("--grad_clip", type=float, default=1.0)
    p_train.add_argument("--val_ratio", type=float, default=0.05)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--device", type=str, default="cuda")
    p_train.add_argument("--num_workers", type=int, default=8)
    p_train.add_argument("--checkpoint_every", type=int, default=10)
    p_train.add_argument("--w_joint_pos", type=float, default=1.0)
    p_train.add_argument("--w_joint_vel", type=float, default=1.0)
    p_train.add_argument("--max_files", type=int, default=None, help="Optional cap on NPZ files (debug).")
    p_train.add_argument("--max_windows", type=int, default=None, help="Optional cap on train windows (debug).")
    p_train.add_argument("--use_wandb", action="store_true", default=False)
    p_train.add_argument("--wandb_project", type=str, default="general_self_bfm")
    p_train.add_argument("--wandb_entity", type=str, default=None)
    p_train.add_argument("--wandb_run_name", type=str, default=None)
    p_train.add_argument("--wandb_tags", type=str, default="g1_kin_offline,fsq")
    p_train.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=("online", "offline", "disabled"),
    )
    p_train.add_argument("--wandb_log_every_steps", type=int, default=20)
    p_train.set_defaults(func=train_command)

    p_recon = sub.add_parser("reconstruct", help="Reconstruct joint_pos/joint_vel for one NPZ.")
    p_recon.add_argument("--ckpt", type=str, required=True)
    p_recon.add_argument("--input_npz", type=str, required=True)
    p_recon.add_argument("--output_npz", type=str, required=True)
    p_recon.add_argument("--output_metrics_json", type=str, default=None)
    p_recon.add_argument("--seq_len", type=int, default=0, help="0 = use checkpoint seq_len.")
    p_recon.add_argument("--frame_stride", type=int, default=0, help="0 = use checkpoint frame_stride.")
    p_recon.add_argument("--window_stride", type=int, default=0, help="0 = use checkpoint window_stride.")
    p_recon.add_argument("--no_cover_tail", action="store_true")
    p_recon.add_argument("--batch_size", type=int, default=256)
    p_recon.add_argument("--device", type=str, default="cuda")
    p_recon.set_defaults(func=reconstruct_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
