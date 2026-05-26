from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler


# ============================================================
# Constants
# ============================================================
BOARD_ROWS = 18
BOARD_COLS = 14
ACTION_CLASSES = 56
STATE_VEC_DIM = 7 + 4 + 14

PIECE_ORDER = ["I", "O", "T", "S", "Z", "J", "L"]
PIECE_TO_INDEX = {kind: idx for idx, kind in enumerate(PIECE_ORDER)}

SHAPE_ID: Dict[str, int] = {
    "I": 1,
    "O": 2,
    "T": 3,
    "S": 4,
    "Z": 5,
    "J": 6,
    "L": 7,
}

# I piece is 3 cells, not 4.
# left_col14 semantics.
SHAPES_NN: Dict[str, List[List[Tuple[int, int]]]] = {
    "I": [
        [(0, 0), (0, 1), (0, 2)],
        [(0, 1), (1, 1), (2, 1)],
    ],
    "O": [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    "T": [
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (1, 2), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
    ],
    "S": [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 1), (1, 1), (1, 2), (2, 2)],
    ],
    "Z": [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 2), (1, 1), (1, 2), (2, 1)],
    ],
    "J": [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
    ],
    "L": [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (1, 2), (2, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
}


# ============================================================
# Model
# ============================================================
class BoardEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 9 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(x))


class StateEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_VEC_DIM, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionPolicyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.board_encoder = BoardEncoder()
        self.state_encoder = StateEncoder()
        self.head = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, ACTION_CLASSES),
        )

    def forward(self, board_tensor: torch.Tensor, state_tensor: torch.Tensor) -> torch.Tensor:
        board_feat = self.board_encoder(board_tensor)
        state_feat = self.state_encoder(state_tensor)
        fused = torch.cat([board_feat, state_feat], dim=1)
        return self.head(fused)


# ============================================================
# Helpers
# ============================================================
def one_hot(index: int, size: int) -> List[float]:
    out = [0.0] * size
    if 0 <= index < size:
        out[index] = 1.0
    return out


def build_state_vector(kind_id: int, cur_rot4: int, cur_col14: int) -> List[float]:
    return one_hot(kind_id, 7) + one_hot(cur_rot4, 4) + one_hot(cur_col14, BOARD_COLS)


def normalize_legal_masks(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[1] == 4 and arr.shape[2] == 14:
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[1] == 14 and arr.shape[2] == 4:
        return np.transpose(arr, (0, 2, 1)).astype(np.float32)
    if arr.ndim == 2 and arr.shape[1] == 56:
        return arr.reshape(arr.shape[0], 4, 14).astype(np.float32)
    raise ValueError(f"unsupported legal mask shape: {arr.shape}")


def normalize_targets(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr.astype(np.int64)
    if arr.ndim == 2 and arr.shape[1] == 56:
        return np.argmax(arr, axis=1).astype(np.int64)
    if arr.ndim == 3 and arr.shape[1] == 4 and arr.shape[2] == 14:
        return np.argmax(arr.reshape(arr.shape[0], -1), axis=1).astype(np.int64)
    if arr.ndim == 3 and arr.shape[1] == 14 and arr.shape[2] == 4:
        flat = np.transpose(arr, (0, 2, 1)).reshape(arr.shape[0], -1)
        return np.argmax(flat, axis=1).astype(np.int64)
    raise ValueError(f"unsupported targets shape: {arr.shape}")


def parse_pieces_to_state_vectors(pieces: np.ndarray) -> np.ndarray:
    if pieces.ndim != 2:
        raise ValueError(f"unsupported pieces shape: {pieces.shape}")

    n, d = pieces.shape
    if d == STATE_VEC_DIM:
        return pieces.astype(np.float32)
    if d == 3:
        out = np.zeros((n, STATE_VEC_DIM), dtype=np.float32)
        for i in range(n):
            kind_id = int(pieces[i, 0])
            cur_rot4 = int(pieces[i, 1])
            cur_col14 = int(pieces[i, 2])
            out[i] = np.asarray(build_state_vector(kind_id, cur_rot4, cur_col14), dtype=np.float32)
        return out
    if d > STATE_VEC_DIM:
        return pieces[:, :STATE_VEC_DIM].astype(np.float32)
    raise ValueError(f"unsupported pieces shape: {pieces.shape}")


def build_state_vecs_from_components(
    kind_ids: np.ndarray,
    cur_rot4: np.ndarray,
    cur_col14: np.ndarray,
    kind_dim: int = 7,
    rot_dim: int = 4,
    col_dim: int = 14,
) -> np.ndarray:
    n = len(kind_ids)
    out = np.zeros((n, kind_dim + rot_dim + col_dim), dtype=np.float32)

    kind_ids = kind_ids.astype(np.int64)
    cur_rot4 = cur_rot4.astype(np.int64)
    cur_col14 = cur_col14.astype(np.int64)

    valid_kind = (kind_ids >= 0) & (kind_ids < kind_dim)
    valid_rot = (cur_rot4 >= 0) & (cur_rot4 < rot_dim)
    valid_col = (cur_col14 >= 0) & (cur_col14 < col_dim)

    idx = np.arange(n, dtype=np.int64)

    out[idx[valid_kind], kind_ids[valid_kind]] = 1.0
    out[idx[valid_rot], kind_dim + cur_rot4[valid_rot]] = 1.0
    out[idx[valid_col], kind_dim + rot_dim + cur_col14[valid_col]] = 1.0

    return out


def load_first_existing_npy(root: Path, candidates: List[str]) -> np.ndarray:
    for name in candidates:
        path = root / name
        if path.exists():
            return np.load(path, allow_pickle=False)
    raise FileNotFoundError(
        f"None of the candidate files exist under {root}: {candidates}"
    )


def load_optional_idx(root: Path, name: str) -> Optional[np.ndarray]:
    path = root / name
    if path.exists():
        return np.load(path, allow_pickle=False).astype(np.int32)
    return None


def load_optional_flag_from_npz(pack: Optional[np.lib.npyio.NpzFile], name: str, n: int) -> np.ndarray:
    if pack is not None and name in pack:
        arr = np.asarray(pack[name]).astype(np.int64)
        if len(arr) != n:
            raise ValueError(f"{name} length mismatch in npz: {len(arr)} vs {n}")
        return arr
    return np.zeros(n, dtype=np.int64)


def mask_logits_with_legal(logits: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor:
    if legal_masks.ndim == 3:
        legal_flat = legal_masks.reshape(logits.size(0), -1)
    elif legal_masks.ndim == 2:
        legal_flat = legal_masks
    else:
        raise ValueError(f"unsupported legal_masks ndim: {legal_masks.ndim}")

    masked = logits.masked_fill(legal_flat <= 0.5, -1e9)
    bad_rows = legal_flat.sum(dim=1) < 0.5
    if bad_rows.any():
        masked[bad_rows] = logits[bad_rows]
    return masked


def build_random_split(
    n: int,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n < 3:
        raise ValueError(f"dataset too small after filtering: n={n}")

    idx = np.arange(n, dtype=np.int32)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def remap_split_indices(old_indices: np.ndarray, old_to_new: Dict[int, int]) -> np.ndarray:
    remapped = [old_to_new[int(i)] for i in old_indices.tolist() if int(i) in old_to_new]
    return np.asarray(remapped, dtype=np.int32)


def get_subset_weights(subset: Subset) -> np.ndarray:
    ds = subset.dataset
    idx = np.asarray(subset.indices, dtype=np.int64)
    if hasattr(ds, "sample_weights"):
        return np.asarray(ds.sample_weights, dtype=np.float32)[idx]
    return np.ones(len(idx), dtype=np.float32)


def build_adjusted_elite_weights(
    raw_weights: np.ndarray,
    is_repair: np.ndarray,
    is_rescue: np.ndarray,
    is_stable: np.ndarray,
    is_healthy_clear: np.ndarray,
    is_sustain: np.ndarray,
    is_tail: np.ndarray,
    alpha: float,
    clip_min: float,
    clip_max: float,
) -> np.ndarray:
    w = np.asarray(raw_weights, dtype=np.float32).copy()
    bad = (~np.isfinite(w)) | (w <= 0)
    w[bad] = 1.0

    mean_w = float(np.mean(w)) if len(w) > 0 else 1.0
    if mean_w <= 1e-8:
        mean_w = 1.0
    w = w / mean_w

    alpha = float(alpha)
    bonus = (
        0.55 * is_repair.astype(np.float32)
        + 0.60 * is_rescue.astype(np.float32)
        + 0.35 * is_stable.astype(np.float32)
        + 0.45 * is_healthy_clear.astype(np.float32)
        + 0.20 * is_sustain.astype(np.float32)
        + 0.12 * is_tail.astype(np.float32)
    )

    w = w + alpha * bonus
    w = np.clip(w, clip_min, clip_max).astype(np.float32)

    print(
        f"[ELITE-WEIGHT] alpha={alpha:.3f} clip=({clip_min:.2f}, {clip_max:.2f}) "
        f"min={float(w.min()):.4f} mean={float(w.mean()):.4f} max={float(w.max()):.4f}"
    )
    return w


def extract_state_dict_from_checkpoint(checkpoint: object) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("unsupported checkpoint format")

    for key in ["model_state_dict", "state_dict", "model", "net"]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint  # type: ignore[return-value]


# ============================================================
# Dataset base class
# ============================================================
class ActionDatasetBase(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.boards: np.ndarray
        self.actives: np.ndarray
        self.legals: np.ndarray
        self.pieces: np.ndarray
        self.targets: np.ndarray

        self.sample_weights: np.ndarray
        self.is_clear_gain: np.ndarray
        self.is_late_phase: np.ndarray
        self.source_id: np.ndarray

        self.is_repair_sample: np.ndarray
        self.is_rescue_sample: np.ndarray
        self.is_stable_sample: np.ndarray
        self.is_healthy_clear_sample: np.ndarray
        self.is_sustain_sample: np.ndarray
        self.is_tail_candidate: np.ndarray

        self.train_idx: np.ndarray
        self.val_idx: np.ndarray
        self.test_idx: np.ndarray

        self.integrity_report: Dict[str, object] = {}

    def __len__(self) -> int:
        return int(len(self.targets))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        board_tensor = np.stack([self.boards[idx], self.actives[idx]], axis=0).astype(np.float32)
        return {
            "board_tensor": torch.from_numpy(board_tensor),
            "state_tensor": torch.from_numpy(self.pieces[idx].astype(np.float32)),
            "legal_mask_4x14": torch.from_numpy(self.legals[idx].astype(np.float32)),
            "target_action": torch.tensor(int(self.targets[idx]), dtype=torch.long),
            "sample_weight": torch.tensor(float(self.sample_weights[idx]), dtype=torch.float32),
            "is_clear_gain": torch.tensor(int(self.is_clear_gain[idx]), dtype=torch.long),
            "is_late_phase": torch.tensor(int(self.is_late_phase[idx]), dtype=torch.long),
            "source_id": torch.tensor(int(self.source_id[idx]), dtype=torch.long),
            "is_repair_sample": torch.tensor(int(self.is_repair_sample[idx]), dtype=torch.long),
            "is_rescue_sample": torch.tensor(int(self.is_rescue_sample[idx]), dtype=torch.long),
            "is_stable_sample": torch.tensor(int(self.is_stable_sample[idx]), dtype=torch.long),
            "is_healthy_clear_sample": torch.tensor(int(self.is_healthy_clear_sample[idx]), dtype=torch.long),
            "is_sustain_sample": torch.tensor(int(self.is_sustain_sample[idx]), dtype=torch.long),
            "is_tail_candidate": torch.tensor(int(self.is_tail_candidate[idx]), dtype=torch.long),
        }

    def finalize_and_filter(self, seed: int, name: str) -> None:
        n0 = len(self.targets)
        idx = np.arange(n0, dtype=np.int32)

        target_ok = (self.targets >= 0) & (self.targets < ACTION_CLASSES)
        tgt_rot = self.targets // BOARD_COLS
        tgt_col = self.targets % BOARD_COLS

        legal_ok = np.zeros(n0, dtype=bool)
        valid_pos = np.where(target_ok)[0]
        legal_ok[valid_pos] = self.legals[valid_pos, tgt_rot[valid_pos], tgt_col[valid_pos]] > 0.5

        finite_ok = (
            np.isfinite(self.boards).all(axis=(1, 2))
            & np.isfinite(self.actives).all(axis=(1, 2))
            & np.isfinite(self.legals).all(axis=(1, 2))
            & np.isfinite(self.pieces).all(axis=1)
            & np.isfinite(self.sample_weights)
        )

        valid_mask = target_ok & legal_ok & finite_ok
        keep_idx = idx[valid_mask]

        removed_total = int(n0 - len(keep_idx))
        removed_target_range = int((~target_ok).sum())
        removed_target_illegal = int((target_ok & (~legal_ok)).sum())
        removed_nonfinite = int((~finite_ok).sum())

        if len(keep_idx) <= 0:
            raise ValueError(f"{name}: no valid samples after filtering")

        old_to_new = {int(old_i): new_i for new_i, old_i in enumerate(keep_idx.tolist())}

        old_train = self.train_idx.copy()
        old_val = self.val_idx.copy()
        old_test = self.test_idx.copy()

        self.boards = self.boards[keep_idx]
        self.actives = self.actives[keep_idx]
        self.legals = self.legals[keep_idx]
        self.pieces = self.pieces[keep_idx]
        self.targets = self.targets[keep_idx]
        self.sample_weights = self.sample_weights[keep_idx]
        self.is_clear_gain = self.is_clear_gain[keep_idx]
        self.is_late_phase = self.is_late_phase[keep_idx]
        self.source_id = self.source_id[keep_idx]
        self.is_repair_sample = self.is_repair_sample[keep_idx]
        self.is_rescue_sample = self.is_rescue_sample[keep_idx]
        self.is_stable_sample = self.is_stable_sample[keep_idx]
        self.is_healthy_clear_sample = self.is_healthy_clear_sample[keep_idx]
        self.is_sustain_sample = self.is_sustain_sample[keep_idx]
        self.is_tail_candidate = self.is_tail_candidate[keep_idx]

        self.train_idx = remap_split_indices(old_train, old_to_new)
        self.val_idx = remap_split_indices(old_val, old_to_new)
        self.test_idx = remap_split_indices(old_test, old_to_new)

        split_ok = len(self.train_idx) > 0 and len(self.val_idx) > 0 and len(self.test_idx) > 0
        if not split_ok:
            self.train_idx, self.val_idx, self.test_idx = build_random_split(
                n=len(self.targets),
                seed=seed,
            )

        self.integrity_report = {
            "name": name,
            "n_before": int(n0),
            "n_after": int(len(self.targets)),
            "removed_total": removed_total,
            "removed_target_range": removed_target_range,
            "removed_target_illegal": removed_target_illegal,
            "removed_nonfinite": removed_nonfinite,
            "split_fallback_random": int(not split_ok),
            "train": int(len(self.train_idx)),
            "val": int(len(self.val_idx)),
            "test": int(len(self.test_idx)),
        }

        print(
            f"[DATA] {name} | before={n0} after={len(self.targets)} "
            f"removed={removed_total} illegal_target={removed_target_illegal} "
            f"bad_range={removed_target_range} nonfinite={removed_nonfinite} "
            f"fallback_split={int(not split_ok)}"
        )


# ============================================================
# Base stabilizer dataset
# ============================================================
class BaseTeacherDataset(ActionDatasetBase):
    def __init__(self, data_dir: str, seed: int) -> None:
        super().__init__()
        root = Path(data_dir)
        if not root.exists():
            raise FileNotFoundError(f"base_data_dir not found: {root}")

        self.boards = load_first_existing_npy(root, ["boards.npy"]).astype(np.float32)
        self.actives = load_first_existing_npy(root, ["actives.npy", "active_masks.npy"]).astype(np.float32)
        self.legals = normalize_legal_masks(load_first_existing_npy(root, ["legals.npy", "legal_masks.npy"]))
        self.pieces = parse_pieces_to_state_vectors(load_first_existing_npy(root, ["pieces.npy", "state_vecs.npy"]))
        self.targets = normalize_targets(load_first_existing_npy(root, ["targets.npy", "target_actions.npy"])).astype(np.int64)

        n = len(self.targets)
        self.sample_weights = np.ones(n, dtype=np.float32)
        self.is_clear_gain = np.zeros(n, dtype=np.int64)
        self.is_late_phase = np.zeros(n, dtype=np.int64)
        self.source_id = np.zeros(n, dtype=np.int64)

        self.is_repair_sample = np.zeros(n, dtype=np.int64)
        self.is_rescue_sample = np.zeros(n, dtype=np.int64)
        self.is_stable_sample = np.zeros(n, dtype=np.int64)
        self.is_healthy_clear_sample = np.zeros(n, dtype=np.int64)
        self.is_sustain_sample = np.zeros(n, dtype=np.int64)
        self.is_tail_candidate = np.zeros(n, dtype=np.int64)

        train_idx = load_optional_idx(root, "train_idx.npy")
        val_idx = load_optional_idx(root, "val_idx.npy")
        test_idx = load_optional_idx(root, "test_idx.npy")
        if train_idx is None or val_idx is None or test_idx is None:
            self.train_idx, self.val_idx, self.test_idx = build_random_split(n=n, seed=seed)
        else:
            self.train_idx = train_idx
            self.val_idx = val_idx
            self.test_idx = test_idx

        self.finalize_and_filter(seed=seed, name="base_teacher")


# ============================================================
# Elite dataset
# ============================================================
class EliteV6Dataset(ActionDatasetBase):
    def __init__(
        self,
        root: str | Path,
        sample_weight_override: Optional[np.ndarray] = None,
        log_prefix: str = "elite_v6",
        seed: int = 42,
        elite_weight_alpha: float = 0.35,
        sample_weight_clip_min: float = 0.80,
        sample_weight_clip_max: float = 1.80,
        **kwargs,
    ):
        super().__init__()
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"elite_data_dir not found: {self.root}")

        self.boards = load_first_existing_npy(
            self.root,
            ["boards.npy", "boards_before.npy"],
        ).astype(np.float32)

        self.actives = load_first_existing_npy(
            self.root,
            ["actives.npy", "active_masks.npy"],
        ).astype(np.float32)

        self.legals = normalize_legal_masks(
            load_first_existing_npy(
                self.root,
                ["legal_masks.npy", "legals.npy", "legal_mask.npy"],
            )
        )

        state_vec_path = self.root / "state_vecs.npy"
        if state_vec_path.exists():
            self.pieces = np.load(state_vec_path, allow_pickle=False).astype(np.float32)
        else:
            pieces_path = self.root / "pieces.npy"
            if pieces_path.exists():
                self.pieces = parse_pieces_to_state_vectors(np.load(pieces_path, allow_pickle=False))
            else:
                kind_ids = load_first_existing_npy(
                    self.root,
                    ["kind_ids.npy", "kind_id.npy", "piece_ids.npy", "piece_id.npy"],
                )
                cur_rot4 = load_first_existing_npy(
                    self.root,
                    ["cur_rot4.npy", "rot_ids.npy", "rot_id.npy"],
                )
                cur_col14 = load_first_existing_npy(
                    self.root,
                    ["cur_col14.npy", "col_ids.npy", "col_id.npy"],
                )
                self.pieces = build_state_vecs_from_components(
                    kind_ids=kind_ids,
                    cur_rot4=cur_rot4,
                    cur_col14=cur_col14,
                )

        self.targets = normalize_targets(
            load_first_existing_npy(
                self.root,
                ["target_actions.npy", "targets.npy", "action_targets.npy", "target_action.npy"],
            )
        ).astype(np.int64)

        n = len(self.targets)
        if len(self.boards) != n:
            raise ValueError(f"boards length mismatch: {len(self.boards)} vs {n}")
        if len(self.actives) != n:
            raise ValueError(f"actives length mismatch: {len(self.actives)} vs {n}")
        if len(self.legals) != n:
            raise ValueError(f"legal_masks length mismatch: {len(self.legals)} vs {n}")
        if len(self.pieces) != n:
            raise ValueError(f"state/pieces length mismatch: {len(self.pieces)} vs {n}")

        npz_list = sorted(self.root.glob("selected_dataset_*.npz"))
        pack = np.load(npz_list[0], allow_pickle=False) if npz_list else None

        def load_flag(name: str) -> np.ndarray:
            npy_path = self.root / f"{name}.npy"
            if npy_path.exists():
                arr = np.load(npy_path, allow_pickle=False).astype(np.int64)
                if len(arr) != n:
                    raise ValueError(f"{name}.npy length mismatch: {len(arr)} vs {n}")
                return arr
            return load_optional_flag_from_npz(pack, name, n)

        self.is_repair_sample = load_flag("is_repair_sample")
        self.is_rescue_sample = load_flag("is_rescue_sample")
        self.is_stable_sample = load_flag("is_stable_sample")
        self.is_healthy_clear_sample = load_flag("is_healthy_clear_sample")
        self.is_sustain_sample = load_flag("is_sustain_sample")
        self.is_tail_candidate = load_flag("is_tail_candidate")

        if pack is not None and "lines_gain_immediate" in pack:
            self.is_clear_gain = (np.asarray(pack["lines_gain_immediate"]).astype(np.int16) > 0).astype(np.int64)
        else:
            self.is_clear_gain = self.is_healthy_clear_sample.copy()

        self.is_late_phase = (
            (self.is_rescue_sample > 0)
            | (self.is_tail_candidate > 0)
        ).astype(np.int64)

        self.source_id = np.ones(n, dtype=np.int64)

        if sample_weight_override is not None:
            raw_weights = np.asarray(sample_weight_override, dtype=np.float32)
        else:
            sw_path = self.root / "sample_weight_v6.npy"
            if sw_path.exists():
                raw_weights = np.load(sw_path, allow_pickle=False).astype(np.float32)
            else:
                raw_weights = np.ones(n, dtype=np.float32)

        if len(raw_weights) != n:
            raise ValueError(f"sample_weight length mismatch: {len(raw_weights)} vs {n}")

        self.sample_weights = build_adjusted_elite_weights(
            raw_weights=raw_weights,
            is_repair=self.is_repair_sample,
            is_rescue=self.is_rescue_sample,
            is_stable=self.is_stable_sample,
            is_healthy_clear=self.is_healthy_clear_sample,
            is_sustain=self.is_sustain_sample,
            is_tail=self.is_tail_candidate,
            alpha=elite_weight_alpha,
            clip_min=sample_weight_clip_min,
            clip_max=sample_weight_clip_max,
        )

        train_idx = load_optional_idx(self.root, "train_idx.npy")
        val_idx = load_optional_idx(self.root, "val_idx.npy")
        test_idx = load_optional_idx(self.root, "test_idx.npy")

        if train_idx is None and pack is not None and "train_idx" in pack:
            train_idx = np.asarray(pack["train_idx"]).astype(np.int32)
        if val_idx is None and pack is not None and "val_idx" in pack:
            val_idx = np.asarray(pack["val_idx"]).astype(np.int32)
        if test_idx is None and pack is not None and "test_idx" in pack:
            test_idx = np.asarray(pack["test_idx"]).astype(np.int32)

        if train_idx is None or val_idx is None or test_idx is None:
            self.train_idx, self.val_idx, self.test_idx = build_random_split(n=n, seed=seed)
        else:
            self.train_idx = train_idx
            self.val_idx = val_idx
            self.test_idx = test_idx

        self.finalize_and_filter(seed=seed, name=log_prefix)


# ============================================================
# Mini strict-eval environment
# ============================================================
@dataclass
class PieceNN:
    kind: str
    rotation: int
    row: int
    col: int

    def cells(self) -> List[Tuple[int, int]]:
        rotation_count = len(SHAPES_NN[self.kind])
        rotation_index = self.rotation % rotation_count
        offsets = SHAPES_NN[self.kind][rotation_index]
        return [(self.row + r, self.col + c) for r, c in offsets]


def occ_board(board: List[List[int]]) -> List[List[int]]:
    return [[1 if int(cell) != 0 else 0 for cell in row] for row in board]


def board_to_np(board_occ: List[List[int]]) -> np.ndarray:
    return np.asarray(board_occ, dtype=np.int8)


def column_heights_eval(board_occ: np.ndarray) -> np.ndarray:
    heights = np.zeros((BOARD_COLS,), dtype=np.int32)
    for col in range(BOARD_COLS):
        filled = np.where(board_occ[:, col] > 0)[0]
        heights[col] = 0 if filled.size == 0 else BOARD_ROWS - int(filled[0])
    return heights


def count_holes_eval(board_occ: np.ndarray) -> int:
    holes = 0
    for col in range(BOARD_COLS):
        seen_filled = False
        for row in range(BOARD_ROWS):
            if int(board_occ[row, col]) != 0:
                seen_filled = True
            elif seen_filled:
                holes += 1
    return holes


def count_transitions_eval(board_occ: np.ndarray, axis: int) -> int:
    total = 0
    if axis == 0:
        for row in range(BOARD_ROWS):
            prev = 1
            for col in range(BOARD_COLS):
                cur = 1 if int(board_occ[row, col]) != 0 else 0
                if cur != prev:
                    total += 1
                prev = cur
            if prev == 0:
                total += 1
    else:
        for col in range(BOARD_COLS):
            prev = 1
            for row in range(BOARD_ROWS):
                cur = 1 if int(board_occ[row, col]) != 0 else 0
                if cur != prev:
                    total += 1
                prev = cur
            if prev == 0:
                total += 1
    return total


def count_near_clear_eval(board_occ: np.ndarray) -> int:
    return int(np.sum(board_occ.sum(axis=1) >= BOARD_COLS - 1))


def eval_board_features(board_occ: np.ndarray) -> Dict[str, float]:
    heights = column_heights_eval(board_occ)
    return {
        "agg_h": float(heights.sum()),
        "max_h": float(heights.max()) if heights.size > 0 else 0.0,
        "holes": float(count_holes_eval(board_occ)),
        "bump": float(np.abs(heights[1:] - heights[:-1]).sum()) if heights.size > 1 else 0.0,
        "near": float(count_near_clear_eval(board_occ)),
        "row_t": float(count_transitions_eval(board_occ, axis=0)),
        "col_t": float(count_transitions_eval(board_occ, axis=1)),
    }


def simulate_piece_on_board(
    board_occ: List[List[int]],
    piece: PieceNN,
) -> Optional[Tuple[np.ndarray, int]]:
    out = board_to_np(board_occ).copy()
    for row, col in piece.cells():
        if row < 0 or row >= BOARD_ROWS or col < 0 or col >= BOARD_COLS:
            return None
        if out[row, col] != 0:
            return None
        out[row, col] = 1

    full_rows = np.where(out.sum(axis=1) == BOARD_COLS)[0]
    lines = int(full_rows.size)
    if lines > 0:
        keep = np.delete(out, full_rows, axis=0)
        out = np.vstack([np.zeros((lines, BOARD_COLS), dtype=np.int8), keep]).astype(np.int8)
    return out, lines


def placement_safety_score(
    board_before: List[List[int]],
    board_after: np.ndarray,
    lines: int,
) -> float:
    before = eval_board_features(board_to_np(board_before))
    after = eval_board_features(board_after)

    new_holes = max(0.0, after["holes"] - before["holes"])
    height_gain = max(0.0, after["max_h"] - before["max_h"])
    danger_height = max(0.0, after["max_h"] - 10.0)
    critical_height = max(0.0, after["max_h"] - 13.0)

    score = 0.0
    score += 9.0 * float(lines)
    score += 1.2 * after["near"]
    score -= 5.5 * new_holes
    score -= 2.0 * after["holes"]
    score -= 0.85 * after["bump"]
    score -= 0.08 * after["agg_h"]
    score -= 0.22 * after["row_t"]
    score -= 0.18 * after["col_t"]
    score -= 1.0 * height_gain
    score -= 1.3 * danger_height
    score -= 2.5 * critical_height * critical_height
    return float(score)


def flatten_4x14(matrix_4x14: List[List[float]]) -> List[float]:
    out: List[float] = []
    for rot in range(4):
        for col in range(BOARD_COLS):
            out.append(float(matrix_4x14[rot][col]))
    return out


def decode_action(action_idx: int) -> Tuple[int, int]:
    rot4 = int(action_idx) // BOARD_COLS
    col14 = int(action_idx) % BOARD_COLS
    return rot4, col14


def leftmost_col_nn(piece: PieceNN) -> int:
    return min(col for _, col in piece.cells())


def left_col14_to_anchor_col_nn(kind: str, rot4: int, left_col14: int) -> int:
    rotation_count = len(SHAPES_NN[kind])
    rot_idx = rot4 % rotation_count
    offsets = SHAPES_NN[kind][rot_idx]
    min_offset_col = min(col for _, col in offsets)
    return left_col14 - min_offset_col


def build_active_mask_nn(piece: PieceNN) -> List[List[int]]:
    mask = [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
    for row, col in piece.cells():
        if 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS:
            mask[row][col] = 1
    return mask


class MiniStrictEvalGame:
    def __init__(self, rng: random.Random, max_pieces: int) -> None:
        self.rng = rng
        self.max_pieces = max_pieces
        self.board: List[List[int]] = [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        self.score = 0
        self.lines_cleared_total = 0
        self.pieces_placed = 0
        self.game_over = False
        self.stop_reason = "running"
        self.current_piece: Optional[PieceNN] = None
        self.next_piece_kind: str = ""
        self.bag: List[str] = []
        self.next_piece_kind = self._draw_from_bag()
        self.spawn_piece()

    def _refill_bag(self) -> None:
        self.bag = list(SHAPES_NN.keys())
        self.rng.shuffle(self.bag)

    def _draw_from_bag(self) -> str:
        if not self.bag:
            self._refill_bag()
        return self.bag.pop()

    def spawn_piece(self) -> None:
        if self.pieces_placed >= self.max_pieces:
            self.current_piece = None
            self.game_over = True
            self.stop_reason = "max_pieces_cap"
            return

        kind = self.next_piece_kind
        self.next_piece_kind = self._draw_from_bag()

        spawn_col = BOARD_COLS // 2 - 2
        for offset in [0, -1, 1, -2, 2, -3, 3]:
            candidate = PieceNN(kind=kind, rotation=0, row=0, col=spawn_col + offset)
            if self.is_valid_position(candidate):
                self.current_piece = candidate
                return

        self.current_piece = PieceNN(kind=kind, rotation=0, row=0, col=spawn_col)
        self.game_over = True
        self.stop_reason = "top_out"

    def is_valid_position(self, piece: PieceNN) -> bool:
        for row, col in piece.cells():
            if col < 0 or col >= BOARD_COLS:
                return False
            if row < 0 or row >= BOARD_ROWS:
                return False
            if self.board[row][col] != 0:
                return False
        return True

    @staticmethod
    def is_valid_position_on_board(board_occ: List[List[int]], piece: PieceNN) -> bool:
        for row, col in piece.cells():
            if col < 0 or col >= BOARD_COLS:
                return False
            if row < 0 or row >= BOARD_ROWS:
                return False
            if int(board_occ[row][col]) != 0:
                return False
        return True

    def hard_drop_row(self, kind: str, rotation: int, col: int) -> Optional[int]:
        row = 0
        piece = PieceNN(kind=kind, rotation=rotation, row=row, col=col)
        if not self.is_valid_position(piece):
            return None

        while True:
            next_piece = PieceNN(kind=kind, rotation=rotation, row=row + 1, col=col)
            if self.is_valid_position(next_piece):
                row += 1
            else:
                return row

    @staticmethod
    def hard_drop_row_on_board(board_occ: List[List[int]], kind: str, rotation: int, col: int) -> Optional[int]:
        row = 0
        piece = PieceNN(kind=kind, rotation=rotation, row=row, col=col)
        if not MiniStrictEvalGame.is_valid_position_on_board(board_occ, piece):
            return None

        while True:
            next_piece = PieceNN(kind=kind, rotation=rotation, row=row + 1, col=col)
            if MiniStrictEvalGame.is_valid_position_on_board(board_occ, next_piece):
                row += 1
            else:
                return row

    def clear_lines(self) -> int:
        new_board = [row for row in self.board if any(cell == 0 for cell in row)]
        cleared = BOARD_ROWS - len(new_board)
        while len(new_board) < BOARD_ROWS:
            new_board.insert(0, [0 for _ in range(BOARD_COLS)])
        self.board = new_board
        return cleared

    def lock_piece(self, piece: PieceNN) -> int:
        for row, col in piece.cells():
            if 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS:
                self.board[row][col] = SHAPE_ID[piece.kind]

        cleared = self.clear_lines()
        self.lines_cleared_total += cleared
        self.pieces_placed += 1

        score_table = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        self.score += score_table.get(cleared, 0)

        self.spawn_piece()
        return cleared

    def get_current_piece_kind_index(self) -> Optional[int]:
        if self.current_piece is None:
            return None
        return PIECE_TO_INDEX[self.current_piece.kind]

    def get_current_piece_rot4(self) -> Optional[int]:
        if self.current_piece is None:
            return None
        return self.current_piece.rotation % 4

    def get_current_piece_col14(self) -> Optional[int]:
        if self.current_piece is None:
            return None
        return leftmost_col_nn(self.current_piece)

    def build_legal_mask_4x14(self, kind: str) -> List[List[int]]:
        board_occ = occ_board(self.board)
        legal = [[0 for _ in range(BOARD_COLS)] for _ in range(4)]
        rotations = len(SHAPES_NN[kind])
        checked = set()

        for rot in range(rotations):
            offsets = SHAPES_NN[kind][rot]
            min_col = min(col for _, col in offsets)
            max_col = max(col for _, col in offsets)

            for anchor_col in range(-min_col, BOARD_COLS - max_col):
                key = (rot, anchor_col, tuple(sorted(offsets)))
                if key in checked:
                    continue
                checked.add(key)

                final_row = self.hard_drop_row_on_board(board_occ=board_occ, kind=kind, rotation=rot, col=anchor_col)
                if final_row is None:
                    continue

                final_piece = PieceNN(kind=kind, rotation=rot, row=final_row, col=anchor_col)
                left_col = leftmost_col_nn(final_piece)
                if 0 <= left_col < BOARD_COLS:
                    legal[rot % 4][left_col] = 1

        return legal


@torch.no_grad()
def choose_action_greedy(
    model: ActionPolicyNet,
    device: torch.device,
    game: MiniStrictEvalGame,
    policy_topk: int = 8,
    safety_weight: float = 0.70,
) -> Optional[Dict[str, int]]:
    if game.current_piece is None:
        return None

    kind = game.current_piece.kind
    kind_id = game.get_current_piece_kind_index()
    cur_rot4 = game.get_current_piece_rot4()
    cur_col14 = game.get_current_piece_col14()

    if kind_id is None or cur_rot4 is None or cur_col14 is None:
        return None

    board_occ = occ_board(game.board)
    active_mask = build_active_mask_nn(game.current_piece)
    legal_mask_4x14 = game.build_legal_mask_4x14(kind)

    board_tensor = torch.tensor([[board_occ, active_mask]], dtype=torch.float32, device=device)
    state_tensor = torch.tensor(
        [build_state_vector(kind_id, cur_rot4, cur_col14)],
        dtype=torch.float32,
        device=device,
    )
    legal_flat = torch.tensor([flatten_4x14(legal_mask_4x14)], dtype=torch.float32, device=device)

    logits = model(board_tensor, state_tensor)
    masked_logits = mask_logits_with_legal(logits, legal_flat)

    legal_np = np.asarray(flatten_4x14(legal_mask_4x14), dtype=np.float32)
    legal_idx = np.where(legal_np > 0.5)[0]
    if legal_idx.size <= 0:
        return None

    candidate_count = min(max(1, int(policy_topk)), int(legal_idx.size))
    top_actions = torch.topk(masked_logits, k=candidate_count, dim=1).indices[0].detach().cpu().numpy()
    policy_values = masked_logits[0, top_actions].detach().cpu().numpy().astype(np.float32)

    candidate_rows: List[Dict[str, object]] = []
    board_before = occ_board(game.board)
    for action_idx, policy_value in zip(top_actions.tolist(), policy_values.tolist()):
        pred_rot4, pred_col14 = decode_action(int(action_idx))
        pred_anchor_col = left_col14_to_anchor_col_nn(kind, pred_rot4, pred_col14)
        target_final_row = game.hard_drop_row(kind=kind, rotation=pred_rot4, col=pred_anchor_col)
        if target_final_row is None:
            continue

        final_piece = PieceNN(kind=kind, rotation=pred_rot4, row=target_final_row, col=pred_anchor_col)
        if not game.is_valid_position(final_piece):
            continue
        if leftmost_col_nn(final_piece) != pred_col14:
            continue

        simulated = simulate_piece_on_board(board_before, final_piece)
        if simulated is None:
            continue
        board_after, lines = simulated
        candidate_rows.append(
            {
                "action": int(action_idx),
                "policy": float(policy_value),
                "safety": placement_safety_score(board_before, board_after, lines),
            }
        )

    if not candidate_rows:
        chosen_action = int(masked_logits.argmax(dim=1).item())
    else:
        policy_arr = np.asarray([row["policy"] for row in candidate_rows], dtype=np.float32)
        safety_arr = np.asarray([row["safety"] for row in candidate_rows], dtype=np.float32)

        def norm(arr: np.ndarray) -> np.ndarray:
            std = float(arr.std())
            if std < 1e-6:
                return np.zeros_like(arr)
            return (arr - float(arr.mean())) / std

        before_feat = eval_board_features(board_to_np(board_before))
        danger = before_feat["max_h"] >= 10.0 or before_feat["holes"] >= 3.0 or before_feat["bump"] >= 18.0
        local_safety_weight = float(safety_weight) * (1.35 if danger else 1.0)
        combined = norm(policy_arr) + local_safety_weight * norm(safety_arr)
        best_i = int(np.argmax(combined))
        chosen_action = int(candidate_rows[best_i]["action"])

    pred_rot4, pred_col14 = decode_action(chosen_action)
    pred_anchor_col = left_col14_to_anchor_col_nn(kind, pred_rot4, pred_col14)

    target_final_row = game.hard_drop_row(kind=kind, rotation=pred_rot4, col=pred_anchor_col)
    if target_final_row is None:
        return None

    final_piece = PieceNN(kind=kind, rotation=pred_rot4, row=target_final_row, col=pred_anchor_col)
    if not game.is_valid_position(final_piece):
        return None
    if leftmost_col_nn(final_piece) != pred_col14:
        return None

    return {
        "chosen_action": chosen_action,
        "chosen_rot4": pred_rot4,
        "chosen_col14": pred_col14,
        "chosen_anchor_col": pred_anchor_col,
        "chosen_final_row": target_final_row,
    }


@torch.no_grad()
def rollout_one_episode_strict(
    model: nn.Module,
    device: torch.device,
    seed: int,
    max_pieces: int = 500,
    policy_topk: int = 8,
    safety_weight: float = 0.70,
) -> Dict[str, object]:
    rng = random.Random(seed)
    game = MiniStrictEvalGame(rng=rng, max_pieces=max_pieces)

    policy_fail_count = 0

    while not game.game_over and game.current_piece is not None:
        action = choose_action_greedy(
            model=model,
            device=device,
            game=game,
            policy_topk=policy_topk,
            safety_weight=safety_weight,
        )
        if action is None:
            policy_fail_count += 1
            game.game_over = True
            game.stop_reason = "policy_fail"
            break

        locked_piece = PieceNN(
            kind=game.current_piece.kind,
            rotation=int(action["chosen_rot4"]),
            row=int(action["chosen_final_row"]),
            col=int(action["chosen_anchor_col"]),
        )
        if not game.is_valid_position(locked_piece):
            policy_fail_count += 1
            game.game_over = True
            game.stop_reason = "policy_fail"
            break

        game.lock_piece(locked_piece)

    return {
        "seed": seed,
        "lines": int(game.lines_cleared_total),
        "pieces": int(game.pieces_placed),
        "score": int(game.score),
        "stop_reason": game.stop_reason,
        "policy_fail_count": int(policy_fail_count),
    }


# ============================================================
# Eval summary
# ============================================================
@dataclass
class EvalSummary:
    mean_lines: float
    median_lines: float
    std_lines: float
    min_lines: float
    max_lines: float
    mean_pieces: float
    s10: float
    s20: float
    s30: float
    s40: float
    s50: float
    early_death_lt5: float
    metric: float


def run_eval(
    model: nn.Module,
    device: torch.device,
    episodes: int,
    seed_start: int,
    max_pieces: int,
    piece_coef: float,
    s10_coef: float,
    s20_coef: float,
    s30_coef: float,
    s40_coef: float,
    s50_coef: float,
    early_death_coef: float,
    max_lines_coef: float,
    std_lines_coef: float,
    policy_topk: int = 8,
    safety_weight: float = 0.70,
    verbose: bool = False,
    tag: str = "EVAL",
) -> EvalSummary:
    model.eval()

    line_list: List[float] = []
    piece_list: List[float] = []

    for i in range(episodes):
        seed = seed_start + i
        result = rollout_one_episode_strict(
            model=model,
            device=device,
            seed=seed,
            max_pieces=max_pieces,
            policy_topk=policy_topk,
            safety_weight=safety_weight,
        )
        line_list.append(float(result["lines"]))
        piece_list.append(float(result["pieces"]))

        if verbose:
            print(
                f"[{tag}] ep={i + 1:03d}/{episodes:03d} seed={seed} "
                f"pieces={int(result['pieces']):04d} lines={int(result['lines']):04d} "
                f"score={int(result['score']):06d} stop={result['stop_reason']}"
            )

    lines_arr = np.asarray(line_list, dtype=np.float32)
    pieces_arr = np.asarray(piece_list, dtype=np.float32)

    mean_lines = float(lines_arr.mean()) if lines_arr.size > 0 else 0.0
    median_lines = float(np.median(lines_arr)) if lines_arr.size > 0 else 0.0
    std_lines = float(lines_arr.std()) if lines_arr.size > 0 else 0.0
    min_lines = float(lines_arr.min()) if lines_arr.size > 0 else 0.0
    max_lines = float(lines_arr.max()) if lines_arr.size > 0 else 0.0
    mean_pieces = float(pieces_arr.mean()) if pieces_arr.size > 0 else 0.0

    s10 = float(np.mean(lines_arr >= 10)) if lines_arr.size > 0 else 0.0
    s20 = float(np.mean(lines_arr >= 20)) if lines_arr.size > 0 else 0.0
    s30 = float(np.mean(lines_arr >= 30)) if lines_arr.size > 0 else 0.0
    s40 = float(np.mean(lines_arr >= 40)) if lines_arr.size > 0 else 0.0
    s50 = float(np.mean(lines_arr >= 50)) if lines_arr.size > 0 else 0.0
    early_death_lt5 = float(np.mean(lines_arr < 5)) if lines_arr.size > 0 else 0.0

    metric = (
        mean_lines
        + piece_coef * mean_pieces
        + s10_coef * s10
        + s20_coef * s20
        + s30_coef * s30
        + s40_coef * s40
        + s50_coef * s50
        + max_lines_coef * max_lines
        - early_death_coef * early_death_lt5
        - std_lines_coef * std_lines
    )

    return EvalSummary(
        mean_lines=mean_lines,
        median_lines=median_lines,
        std_lines=std_lines,
        min_lines=min_lines,
        max_lines=max_lines,
        mean_pieces=mean_pieces,
        s10=s10,
        s20=s20,
        s30=s30,
        s40=s40,
        s50=s50,
        early_death_lt5=early_death_lt5,
        metric=metric,
    )


# ============================================================
# Epoch metrics
# ============================================================
@dataclass
class EpochMetrics:
    loss: float
    acc: float
    top3: float
    elite_acc: float
    elite_clear_acc: float
    elite_late_acc: float
    base_acc: float
    repair_acc: float
    rescue_acc: float
    stable_acc: float
    healthy_clear_acc: float
    invalid_target_seen: int
    valid_target_used: int


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    is_train: bool,
    label_smoothing: float = 0.0,
    grad_clip: float = 1.0,
    sample_weight_clip_min: float = 0.80,
    sample_weight_clip_max: float = 2.20,
) -> EpochMetrics:
    model.train(is_train)

    total_loss = 0.0
    total_count = 0

    total_correct = 0
    total_top3 = 0

    total_elite = 0
    correct_elite = 0

    total_clear = 0
    correct_clear = 0

    total_late = 0
    correct_late = 0

    total_base = 0
    correct_base = 0

    total_repair = 0
    correct_repair = 0

    total_rescue = 0
    correct_rescue = 0

    total_stable = 0
    correct_stable = 0

    total_healthy_clear = 0
    correct_healthy_clear = 0

    invalid_count = 0
    used_count = 0

    for batch in loader:
        board = batch["board_tensor"].to(device)
        state_vec = batch["state_tensor"].to(device)
        target = batch["target_action"].to(device)
        legal_mask = batch["legal_mask_4x14"].to(device)
        sample_weight = batch["sample_weight"].to(device)

        source_id = batch["source_id"].to(device)
        is_elite_sample = source_id > 0
        is_repair_sample = batch["is_repair_sample"].to(device).bool()
        is_rescue_sample = batch["is_rescue_sample"].to(device).bool()
        is_stable_sample = batch["is_stable_sample"].to(device).bool()
        is_healthy_clear_sample = batch["is_healthy_clear_sample"].to(device).bool()
        is_clear_gain = batch["is_clear_gain"].to(device).bool()
        is_late_phase = batch["is_late_phase"].to(device).bool()
        is_sustain_sample = batch["is_sustain_sample"].to(device).bool()
        is_tail_candidate = batch["is_tail_candidate"].to(device).bool()

        logits = model(board, state_vec)

        legal_flat = legal_mask.reshape(legal_mask.shape[0], -1)
        valid_mask = (target >= 0) & (target < logits.shape[1])
        legal_target_ok = legal_flat.gather(
            1,
            target.clamp(0, logits.shape[1] - 1).unsqueeze(1),
        ).squeeze(1) > 0.5
        valid_mask = valid_mask & legal_target_ok

        invalid_count += int((~valid_mask).sum().item())
        if valid_mask.sum().item() <= 0:
            continue

        logits_v = logits[valid_mask]
        target_v = target[valid_mask]
        legal_flat_v = legal_flat[valid_mask]
        weight_v = sample_weight[valid_mask]

        elite_v = is_elite_sample[valid_mask]
        repair_v = is_repair_sample[valid_mask]
        rescue_v = is_rescue_sample[valid_mask]
        stable_v = is_stable_sample[valid_mask]
        healthy_clear_v = is_healthy_clear_sample[valid_mask]
        clear_v = is_clear_gain[valid_mask]
        late_v = is_late_phase[valid_mask]
        sustain_v = is_sustain_sample[valid_mask]
        tail_v = is_tail_candidate[valid_mask]

        with torch.set_grad_enabled(is_train):
            logits_train_v = mask_logits_with_legal(logits_v, legal_flat_v)
            ce_per = torch_f.cross_entropy(
                logits_train_v,
                target_v,
                reduction="none",
                label_smoothing=label_smoothing,
            )

            dyn_weight = weight_v.clone()

            # 下一轮重点：更重修复和救场，更少强调尾部拔高
            dyn_weight = dyn_weight + 0.06 * repair_v.float()
            dyn_weight = dyn_weight + 0.08 * rescue_v.float()
            dyn_weight = dyn_weight + 0.03 * stable_v.float()
            dyn_weight = dyn_weight + 0.03 * healthy_clear_v.float()
            dyn_weight = dyn_weight + 0.03 * sustain_v.float()
            dyn_weight = dyn_weight + 0.01 * tail_v.float()

            dyn_weight = torch.clamp(
                dyn_weight,
                min=float(sample_weight_clip_min),
                max=float(sample_weight_clip_max + 0.20),
            )

            loss = (ce_per * dyn_weight).sum() / dyn_weight.sum().clamp_min(1e-6)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        logits_eval_v = mask_logits_with_legal(logits_v, legal_flat_v)
        pred = torch.argmax(logits_eval_v, dim=1)
        top3 = torch.topk(logits_eval_v, k=min(3, logits_eval_v.shape[1]), dim=1).indices
        correct = pred == target_v
        top3_correct = (top3 == target_v.unsqueeze(1)).any(dim=1)

        bs = int(target_v.shape[0])
        used_count += bs

        total_loss += float(loss.item()) * bs
        total_count += bs
        total_correct += int(correct.sum().item())
        total_top3 += int(top3_correct.sum().item())

        elite_mask = elite_v
        base_mask = ~elite_v
        clear_mask = clear_v
        late_mask = late_v
        repair_mask = repair_v
        rescue_mask = rescue_v
        stable_mask = stable_v
        healthy_clear_mask = healthy_clear_v

        total_elite += int(elite_mask.sum().item())
        correct_elite += int(correct[elite_mask].sum().item())

        total_base += int(base_mask.sum().item())
        correct_base += int(correct[base_mask].sum().item())

        total_clear += int(clear_mask.sum().item())
        correct_clear += int(correct[clear_mask].sum().item())

        total_late += int(late_mask.sum().item())
        correct_late += int(correct[late_mask].sum().item())

        total_repair += int(repair_mask.sum().item())
        correct_repair += int(correct[repair_mask].sum().item())

        total_rescue += int(rescue_mask.sum().item())
        correct_rescue += int(correct[rescue_mask].sum().item())

        total_stable += int(stable_mask.sum().item())
        correct_stable += int(correct[stable_mask].sum().item())

        total_healthy_clear += int(healthy_clear_mask.sum().item())
        correct_healthy_clear += int(correct[healthy_clear_mask].sum().item())

    def safe_acc(correct_n: int, total_n: int) -> float:
        return float(correct_n / total_n) if total_n > 0 else 0.0

    return EpochMetrics(
        loss=float(total_loss / max(total_count, 1)),
        acc=safe_acc(total_correct, total_count),
        top3=safe_acc(total_top3, total_count),
        elite_acc=safe_acc(correct_elite, total_elite),
        elite_clear_acc=safe_acc(correct_clear, total_clear),
        elite_late_acc=safe_acc(correct_late, total_late),
        base_acc=safe_acc(correct_base, total_base),
        repair_acc=safe_acc(correct_repair, total_repair),
        rescue_acc=safe_acc(correct_rescue, total_rescue),
        stable_acc=safe_acc(correct_stable, total_stable),
        healthy_clear_acc=safe_acc(correct_healthy_clear, total_healthy_clear),
        invalid_target_seen=int(invalid_count),
        valid_target_used=int(used_count),
    )


# ============================================================
# Loader build
# ============================================================
def build_mixed_train_loader(
    elite_train: Subset,
    base_train: Optional[Subset],
    batch_size: int,
    elite_mix_ratio: float,
    samples_per_epoch: int,
    seed: int,
) -> Tuple[DataLoader, int]:
    elite_mix_ratio = max(0.0, min(1.0, float(elite_mix_ratio)))

    if base_train is None or len(base_train) == 0 or elite_mix_ratio >= 0.999:
        ds = elite_train
        weights = get_subset_weights(elite_train).astype(np.float64)
        weights = weights / max(weights.sum(), 1.0)
        num_samples = samples_per_epoch if samples_per_epoch > 0 else len(elite_train)

        g = torch.Generator()
        g.manual_seed(seed)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=int(num_samples),
            replacement=True,
            generator=g,
        )
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False)
        return loader, int(num_samples)

    base_mix_ratio = 1.0 - elite_mix_ratio
    mixed_dataset = ConcatDataset([elite_train, base_train])

    elite_weights = get_subset_weights(elite_train).astype(np.float64)
    elite_weights = elite_weights / max(elite_weights.sum(), 1.0)

    base_weights = get_subset_weights(base_train).astype(np.float64)
    base_weights = base_weights / max(base_weights.sum(), 1.0)

    weights = np.concatenate(
        [elite_mix_ratio * elite_weights, base_mix_ratio * base_weights],
        axis=0,
    )

    num_samples = samples_per_epoch if samples_per_epoch > 0 else (len(elite_train) + len(base_train))

    g = torch.Generator()
    g.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(num_samples),
        replacement=True,
        generator=g,
    )
    loader = DataLoader(mixed_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=False)
    return loader, int(num_samples)


def save_metrics_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_acc",
                "train_top3",
                "train_elite_acc",
                "train_elite_clear_acc",
                "train_elite_late_acc",
                "train_base_acc",
                "train_repair_acc",
                "train_rescue_acc",
                "train_stable_acc",
                "train_healthy_clear_acc",
                "train_invalid_target_seen",
                "train_valid_target_used",
                "val_loss",
                "val_acc",
                "val_top3",
                "val_elite_acc",
                "val_elite_clear_acc",
                "val_elite_late_acc",
                "val_base_acc",
                "val_repair_acc",
                "val_rescue_acc",
                "val_stable_acc",
                "val_healthy_clear_acc",
                "val_invalid_target_seen",
                "val_valid_target_used",
                "heldout_mean_lines",
                "heldout_median_lines",
                "heldout_std_lines",
                "heldout_min_lines",
                "heldout_max_lines",
                "heldout_mean_pieces",
                "heldout_s10",
                "heldout_s20",
                "heldout_s30",
                "heldout_s40",
                "heldout_s50",
                "heldout_early_death_lt5",
                "heldout_metric",
                "shadow_mean_lines",
                "shadow_median_lines",
                "shadow_std_lines",
                "shadow_min_lines",
                "shadow_max_lines",
                "shadow_mean_pieces",
                "shadow_s10",
                "shadow_s20",
                "shadow_s30",
                "shadow_s40",
                "shadow_s50",
                "shadow_early_death_lt5",
                "shadow_metric",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--elite_data_dir",
        type=str,
        default=r"C:\Users\25361\PycharmProjects\pythonProject3\out_task8_elite_v6\v6_struct_mainline",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default=r"C:\Users\25361\PycharmProjects\pythonProject3\out_task6_data",
    )
    parser.add_argument(
        "--init_model",
        type=str,
        default=r"C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_mainline_next\model_best_action.pt",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out_task8_selfimit_v6_mainline_v2",
    )
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--elite_mix_ratio", type=float, default=0.60)
    parser.add_argument("--samples_per_epoch", type=int, default=0)
    parser.add_argument("--disable_base_stabilizer", action="store_true")

    parser.add_argument("--elite_weight_alpha", type=float, default=0.35)
    parser.add_argument("--sample_weight_clip_min", type=float, default=0.80)
    parser.add_argument("--sample_weight_clip_max", type=float, default=1.80)
    parser.add_argument("--eval_policy_topk", type=int, default=8)
    parser.add_argument("--eval_safety_weight", type=float, default=0.70)

    # heldout：继续保留，但只是辅助监控
    parser.add_argument("--heldout_eval_episodes", type=int, default=30)
    parser.add_argument("--heldout_eval_seed_start", type=int, default=7001)
    parser.add_argument("--heldout_eval_max_pieces", type=int, default=500)
    parser.add_argument("--heldout_piece_coef", type=float, default=0.05)
    parser.add_argument("--heldout_s10_coef", type=float, default=0.0)
    parser.add_argument("--heldout_s20_coef", type=float, default=5.0)
    parser.add_argument("--heldout_s30_coef", type=float, default=8.0)
    parser.add_argument("--heldout_s40_coef", type=float, default=5.0)
    parser.add_argument("--heldout_s50_coef", type=float, default=7.0)
    parser.add_argument("--heldout_early_death_coef", type=float, default=3.0)
    parser.add_argument("--heldout_max_lines_coef", type=float, default=0.03)
    parser.add_argument("--heldout_std_lines_coef", type=float, default=0.0)
    parser.add_argument("--heldout_eval_verbose", action="store_true")

    # shadow：更重视低波动、低分尾部、中段稳定
    parser.add_argument("--shadow_eval_episodes", type=int, default=20)
    parser.add_argument("--shadow_eval_seed_start", type=int, default=1001)
    parser.add_argument("--shadow_eval_max_pieces", type=int, default=500)
    parser.add_argument("--shadow_piece_coef", type=float, default=0.02)
    parser.add_argument("--shadow_s10_coef", type=float, default=2.0)
    parser.add_argument("--shadow_s20_coef", type=float, default=4.0)
    parser.add_argument("--shadow_s30_coef", type=float, default=6.0)
    parser.add_argument("--shadow_s40_coef", type=float, default=0.0)
    parser.add_argument("--shadow_s50_coef", type=float, default=0.0)
    parser.add_argument("--shadow_early_death_coef", type=float, default=3.0)
    parser.add_argument("--shadow_max_lines_coef", type=float, default=0.01)
    parser.add_argument("--shadow_std_lines_coef", type=float, default=0.12)
    parser.add_argument("--shadow_eval_verbose", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] device={device}")

    elite_ds = EliteV6Dataset(
        args.elite_data_dir,
        seed=args.seed,
        elite_weight_alpha=args.elite_weight_alpha,
        sample_weight_clip_min=args.sample_weight_clip_min,
        sample_weight_clip_max=args.sample_weight_clip_max,
    )
    elite_train = Subset(elite_ds, elite_ds.train_idx.tolist())
    elite_val = Subset(elite_ds, elite_ds.val_idx.tolist())
    elite_test = Subset(elite_ds, elite_ds.test_idx.tolist())

    base_ds: Optional[BaseTeacherDataset] = None
    base_train = None
    base_val = None
    base_test = None

    if not args.disable_base_stabilizer:
        base_ds = BaseTeacherDataset(args.base_data_dir, seed=args.seed + 1)
        base_train = Subset(base_ds, base_ds.train_idx.tolist())
        base_val = Subset(base_ds, base_ds.val_idx.tolist())
        base_test = Subset(base_ds, base_ds.test_idx.tolist())

    print(
        f"[INFO] elite dataset | total={len(elite_ds)} "
        f"train={len(elite_train)} val={len(elite_val)} test={len(elite_test)}"
    )
    if base_ds is not None:
        print(
            f"[INFO] base dataset | total={len(base_ds)} "
            f"train={len(base_train)} val={len(base_val)} test={len(base_test)}"
        )

    train_loader, samples_per_epoch = build_mixed_train_loader(
        elite_train=elite_train,
        base_train=base_train,
        batch_size=args.batch_size,
        elite_mix_ratio=args.elite_mix_ratio,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
    )

    if base_val is not None and base_test is not None:
        val_dataset = ConcatDataset([elite_val, base_val])
        test_dataset = ConcatDataset([elite_test, base_test])
    else:
        val_dataset = elite_val
        test_dataset = elite_test

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    print(f"[INFO] train samples_per_epoch={samples_per_epoch}")
    print(f"[INFO] elite_mix_ratio={args.elite_mix_ratio:.3f}")
    print(
        f"[INFO] elite_weight_alpha={args.elite_weight_alpha:.3f} "
        f"sample_weight_clip=({args.sample_weight_clip_min:.2f}, {args.sample_weight_clip_max:.2f})"
    )
    print(
        f"[INFO] heldout_eval episodes={args.heldout_eval_episodes} "
        f"seed_start={args.heldout_eval_seed_start} max_pieces={args.heldout_eval_max_pieces}"
    )
    print(
        f"[INFO] shadow_eval episodes={args.shadow_eval_episodes} "
        f"seed_start={args.shadow_eval_seed_start} max_pieces={args.shadow_eval_max_pieces}"
    )
    print(
        f"[INFO] eval_policy_topk={args.eval_policy_topk} "
        f"eval_safety_weight={args.eval_safety_weight:.3f}"
    )

    model = ActionPolicyNet().to(device)

    checkpoint = torch.load(args.init_model, map_location=device)
    state_dict = extract_state_dict_from_checkpoint(checkpoint)

    model_state = model.state_dict()
    loadable_state: Dict[str, torch.Tensor] = {}
    skipped_keys: List[str] = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            loadable_state[k] = v
        else:
            skipped_keys.append(k)

    model_state.update(loadable_state)
    model.load_state_dict(model_state, strict=False)

    print(f"[INIT] init_model={args.init_model}")
    print(f"[INIT] loaded_tensors={len(loadable_state)}")
    print(f"[INIT] skipped_tensors={len(skipped_keys)}")
    print(f"[INIT] first_skipped_keys={skipped_keys[:12]}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_metric = -1e18
    best_epoch = -1
    history_rows: List[Dict[str, object]] = []

    best_path = out_dir / "model_best_action.pt"
    last_path = out_dir / "model_last_action.pt"
    metrics_csv = out_dir / "train_metrics.csv"
    info_json = out_dir / "run_info.json"

    info_payload = {
        "elite_data_dir": args.elite_data_dir,
        "base_data_dir": args.base_data_dir,
        "init_model": args.init_model,
        "elite_integrity": elite_ds.integrity_report,
        "base_integrity": base_ds.integrity_report if base_ds is not None else {},
        "elite_total": len(elite_ds),
        "elite_train": len(elite_train),
        "elite_val": len(elite_val),
        "elite_test": len(elite_test),
        "base_total": len(base_ds) if base_ds is not None else 0,
        "base_train": len(base_train) if base_train is not None else 0,
        "base_val": len(base_val) if base_val is not None else 0,
        "base_test": len(base_test) if base_test is not None else 0,
        "elite_mix_ratio": args.elite_mix_ratio,
        "elite_weight_alpha": args.elite_weight_alpha,
        "sample_weight_clip_min": args.sample_weight_clip_min,
        "sample_weight_clip_max": args.sample_weight_clip_max,
        "eval_policy_topk": args.eval_policy_topk,
        "eval_safety_weight": args.eval_safety_weight,
        "samples_per_epoch": samples_per_epoch,
        "args": vars(args),
    }
    info_json.write_text(json.dumps(info_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            is_train=True,
            label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip,
            sample_weight_clip_min=args.sample_weight_clip_min,
            sample_weight_clip_max=args.sample_weight_clip_max,
        )
        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            is_train=False,
            label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip,
            sample_weight_clip_min=args.sample_weight_clip_min,
            sample_weight_clip_max=args.sample_weight_clip_max,
        )

        heldout_eval = run_eval(
            model=model,
            device=device,
            episodes=args.heldout_eval_episodes,
            seed_start=args.heldout_eval_seed_start,
            max_pieces=args.heldout_eval_max_pieces,
            piece_coef=args.heldout_piece_coef,
            s10_coef=args.heldout_s10_coef,
            s20_coef=args.heldout_s20_coef,
            s30_coef=args.heldout_s30_coef,
            s40_coef=args.heldout_s40_coef,
            s50_coef=args.heldout_s50_coef,
            early_death_coef=args.heldout_early_death_coef,
            max_lines_coef=args.heldout_max_lines_coef,
            std_lines_coef=args.heldout_std_lines_coef,
            policy_topk=args.eval_policy_topk,
            safety_weight=args.eval_safety_weight,
            verbose=args.heldout_eval_verbose,
            tag="HELDOUT",
        )

        shadow_eval = run_eval(
            model=model,
            device=device,
            episodes=args.shadow_eval_episodes,
            seed_start=args.shadow_eval_seed_start,
            max_pieces=args.shadow_eval_max_pieces,
            piece_coef=args.shadow_piece_coef,
            s10_coef=args.shadow_s10_coef,
            s20_coef=args.shadow_s20_coef,
            s30_coef=args.shadow_s30_coef,
            s40_coef=args.shadow_s40_coef,
            s50_coef=args.shadow_s50_coef,
            early_death_coef=args.shadow_early_death_coef,
            max_lines_coef=args.shadow_max_lines_coef,
            std_lines_coef=args.shadow_std_lines_coef,
            policy_topk=args.eval_policy_topk,
            safety_weight=args.eval_safety_weight,
            verbose=args.shadow_eval_verbose,
            tag="SHADOW",
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.acc,
            "train_top3": train_metrics.top3,
            "train_elite_acc": train_metrics.elite_acc,
            "train_elite_clear_acc": train_metrics.elite_clear_acc,
            "train_elite_late_acc": train_metrics.elite_late_acc,
            "train_base_acc": train_metrics.base_acc,
            "train_repair_acc": train_metrics.repair_acc,
            "train_rescue_acc": train_metrics.rescue_acc,
            "train_stable_acc": train_metrics.stable_acc,
            "train_healthy_clear_acc": train_metrics.healthy_clear_acc,
            "train_invalid_target_seen": train_metrics.invalid_target_seen,
            "train_valid_target_used": train_metrics.valid_target_used,
            "val_loss": val_metrics.loss,
            "val_acc": val_metrics.acc,
            "val_top3": val_metrics.top3,
            "val_elite_acc": val_metrics.elite_acc,
            "val_elite_clear_acc": val_metrics.elite_clear_acc,
            "val_elite_late_acc": val_metrics.elite_late_acc,
            "val_base_acc": val_metrics.base_acc,
            "val_repair_acc": val_metrics.repair_acc,
            "val_rescue_acc": val_metrics.rescue_acc,
            "val_stable_acc": val_metrics.stable_acc,
            "val_healthy_clear_acc": val_metrics.healthy_clear_acc,
            "val_invalid_target_seen": val_metrics.invalid_target_seen,
            "val_valid_target_used": val_metrics.valid_target_used,
            "heldout_mean_lines": heldout_eval.mean_lines,
            "heldout_median_lines": heldout_eval.median_lines,
            "heldout_std_lines": heldout_eval.std_lines,
            "heldout_min_lines": heldout_eval.min_lines,
            "heldout_max_lines": heldout_eval.max_lines,
            "heldout_mean_pieces": heldout_eval.mean_pieces,
            "heldout_s10": heldout_eval.s10,
            "heldout_s20": heldout_eval.s20,
            "heldout_s30": heldout_eval.s30,
            "heldout_s40": heldout_eval.s40,
            "heldout_s50": heldout_eval.s50,
            "heldout_early_death_lt5": heldout_eval.early_death_lt5,
            "heldout_metric": heldout_eval.metric,
            "shadow_mean_lines": shadow_eval.mean_lines,
            "shadow_median_lines": shadow_eval.median_lines,
            "shadow_std_lines": shadow_eval.std_lines,
            "shadow_min_lines": shadow_eval.min_lines,
            "shadow_max_lines": shadow_eval.max_lines,
            "shadow_mean_pieces": shadow_eval.mean_pieces,
            "shadow_s10": shadow_eval.s10,
            "shadow_s20": shadow_eval.s20,
            "shadow_s30": shadow_eval.s30,
            "shadow_s40": shadow_eval.s40,
            "shadow_s50": shadow_eval.s50,
            "shadow_early_death_lt5": shadow_eval.early_death_lt5,
            "shadow_metric": shadow_eval.metric,
        }
        history_rows.append(row)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics.loss:.4f} acc={train_metrics.acc:.4f} top3={train_metrics.top3:.4f} "
            f"elite={train_metrics.elite_acc:.4f} elite_clear={train_metrics.elite_clear_acc:.4f} "
            f"elite_late={train_metrics.elite_late_acc:.4f} base={train_metrics.base_acc:.4f} "
            f"repair={train_metrics.repair_acc:.4f} rescue={train_metrics.rescue_acc:.4f} "
            f"stable={train_metrics.stable_acc:.4f} healthy_clear={train_metrics.healthy_clear_acc:.4f} "
            f"invalid={train_metrics.invalid_target_seen} used={train_metrics.valid_target_used} | "
            f"val loss={val_metrics.loss:.4f} acc={val_metrics.acc:.4f} top3={val_metrics.top3:.4f} "
            f"elite={val_metrics.elite_acc:.4f} elite_clear={val_metrics.elite_clear_acc:.4f} "
            f"elite_late={val_metrics.elite_late_acc:.4f} base={val_metrics.base_acc:.4f} "
            f"repair={val_metrics.repair_acc:.4f} rescue={val_metrics.rescue_acc:.4f} "
            f"stable={val_metrics.stable_acc:.4f} healthy_clear={val_metrics.healthy_clear_acc:.4f} | "
            f"heldout lines={heldout_eval.mean_lines:.4f} median={heldout_eval.median_lines:.4f} "
            f"std={heldout_eval.std_lines:.4f} s10={heldout_eval.s10:.4f} s20={heldout_eval.s20:.4f} "
            f"s30={heldout_eval.s30:.4f} early={heldout_eval.early_death_lt5:.4f} "
            f"min={heldout_eval.min_lines:.1f} max={heldout_eval.max_lines:.1f} metric={heldout_eval.metric:.4f} | "
            f"shadow lines={shadow_eval.mean_lines:.4f} median={shadow_eval.median_lines:.4f} "
            f"std={shadow_eval.std_lines:.4f} s10={shadow_eval.s10:.4f} s20={shadow_eval.s20:.4f} "
            f"s30={shadow_eval.s30:.4f} early={shadow_eval.early_death_lt5:.4f} "
            f"min={shadow_eval.min_lines:.1f} max={shadow_eval.max_lines:.1f} metric={shadow_eval.metric:.4f}"
        )

        # best 继续由 shadow formal 主导，但这次它已经更重视低波动和低分尾部
        if shadow_eval.metric > best_metric:
            best_metric = shadow_eval.metric
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "args": vars(args),
                    "heldout_eval": {
                        "mean_lines": heldout_eval.mean_lines,
                        "median_lines": heldout_eval.median_lines,
                        "std_lines": heldout_eval.std_lines,
                        "min_lines": heldout_eval.min_lines,
                        "max_lines": heldout_eval.max_lines,
                        "mean_pieces": heldout_eval.mean_pieces,
                        "s10": heldout_eval.s10,
                        "s20": heldout_eval.s20,
                        "s30": heldout_eval.s30,
                        "s40": heldout_eval.s40,
                        "s50": heldout_eval.s50,
                        "early_death_lt5": heldout_eval.early_death_lt5,
                        "metric": heldout_eval.metric,
                    },
                    "shadow_eval": {
                        "mean_lines": shadow_eval.mean_lines,
                        "median_lines": shadow_eval.median_lines,
                        "std_lines": shadow_eval.std_lines,
                        "min_lines": shadow_eval.min_lines,
                        "max_lines": shadow_eval.max_lines,
                        "mean_pieces": shadow_eval.mean_pieces,
                        "s10": shadow_eval.s10,
                        "s20": shadow_eval.s20,
                        "s30": shadow_eval.s30,
                        "s40": shadow_eval.s40,
                        "s50": shadow_eval.s50,
                        "early_death_lt5": shadow_eval.early_death_lt5,
                        "metric": shadow_eval.metric,
                    },
                },
                best_path,
            )
            print(f"[SAVE] best model updated by shadow formal eval -> {best_path}")

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "args": vars(args),
        },
        last_path,
    )
    print(f"[SAVE] last model -> {last_path}")

    save_metrics_csv(metrics_csv, history_rows)
    print(f"[SAVE] metrics csv -> {metrics_csv}")

    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"], strict=True)

    test_metrics = run_one_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        device=device,
        is_train=False,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip,
        sample_weight_clip_min=args.sample_weight_clip_min,
        sample_weight_clip_max=args.sample_weight_clip_max,
    )

    final_heldout_eval = run_eval(
        model=model,
        device=device,
        episodes=args.heldout_eval_episodes,
        seed_start=args.heldout_eval_seed_start,
        max_pieces=args.heldout_eval_max_pieces,
        piece_coef=args.heldout_piece_coef,
        s10_coef=args.heldout_s10_coef,
        s20_coef=args.heldout_s20_coef,
        s30_coef=args.heldout_s30_coef,
        s40_coef=args.heldout_s40_coef,
        s50_coef=args.heldout_s50_coef,
        early_death_coef=args.heldout_early_death_coef,
        max_lines_coef=args.heldout_max_lines_coef,
        std_lines_coef=args.heldout_std_lines_coef,
        policy_topk=args.eval_policy_topk,
        safety_weight=args.eval_safety_weight,
        verbose=False,
        tag="HELDOUT",
    )

    final_shadow_eval = run_eval(
        model=model,
        device=device,
        episodes=args.shadow_eval_episodes,
        seed_start=args.shadow_eval_seed_start,
        max_pieces=args.shadow_eval_max_pieces,
        piece_coef=args.shadow_piece_coef,
        s10_coef=args.shadow_s10_coef,
        s20_coef=args.shadow_s20_coef,
        s30_coef=args.shadow_s30_coef,
        s40_coef=args.shadow_s40_coef,
        s50_coef=args.shadow_s50_coef,
        early_death_coef=args.shadow_early_death_coef,
        max_lines_coef=args.shadow_max_lines_coef,
        std_lines_coef=args.shadow_std_lines_coef,
        policy_topk=args.eval_policy_topk,
        safety_weight=args.eval_safety_weight,
        verbose=False,
        tag="SHADOW",
    )

    print(
        f"[TEST] best_epoch={best_epoch} "
        f"loss={test_metrics.loss:.4f} acc={test_metrics.acc:.4f} top3={test_metrics.top3:.4f} "
        f"elite={test_metrics.elite_acc:.4f} elite_clear={test_metrics.elite_clear_acc:.4f} "
        f"elite_late={test_metrics.elite_late_acc:.4f} base={test_metrics.base_acc:.4f} "
        f"repair={test_metrics.repair_acc:.4f} rescue={test_metrics.rescue_acc:.4f} "
        f"stable={test_metrics.stable_acc:.4f} healthy_clear={test_metrics.healthy_clear_acc:.4f} "
        f"invalid={test_metrics.invalid_target_seen} used={test_metrics.valid_target_used} | "
        f"heldout metric={final_heldout_eval.metric:.4f} mean={final_heldout_eval.mean_lines:.4f} "
        f"median={final_heldout_eval.median_lines:.4f} std={final_heldout_eval.std_lines:.4f} "
        f"s10={final_heldout_eval.s10:.4f} s20={final_heldout_eval.s20:.4f} s30={final_heldout_eval.s30:.4f} "
        f"early_lt5={final_heldout_eval.early_death_lt5:.4f} min={final_heldout_eval.min_lines:.1f} "
        f"max={final_heldout_eval.max_lines:.1f} | "
        f"shadow metric={final_shadow_eval.metric:.4f} mean={final_shadow_eval.mean_lines:.4f} "
        f"median={final_shadow_eval.median_lines:.4f} std={final_shadow_eval.std_lines:.4f} "
        f"s10={final_shadow_eval.s10:.4f} s20={final_shadow_eval.s20:.4f} s30={final_shadow_eval.s30:.4f} "
        f"early_lt5={final_shadow_eval.early_death_lt5:.4f} min={final_shadow_eval.min_lines:.1f} "
        f"max={final_shadow_eval.max_lines:.1f}"
    )


if __name__ == "__main__":
    main()
