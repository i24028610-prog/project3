from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from torch.utils.data import DataLoader, Dataset, Subset


# ============================================================
# Constants
# ============================================================
BOARD_ROWS = 18
BOARD_COLS = 14
ACTION_CLASSES = 56  # 4 rotations * 14 visible-left columns
STATE_VEC_DIM = 7 + 4 + 14  # kind one-hot + rot one-hot + col one-hot

SECTION_CURRENT_STATE = "CURRENT_STATE_7x4x14"
SECTION_TARGET_14x4 = "TARGET_14x4"
SECTION_LEGAL_TARGET_14x4 = "LEGAL_TARGET_14x4"
SECTION_SCORE_TARGET_14x4 = "SCORE_TARGET_14x4"
SECTION_PROB_TARGET_14x4 = "PROB_TARGET_14x4"

SECTION_BOARD_BEFORE = "BOARD_BEFORE"
SECTION_ACTIVE_MASK = "ACTIVE_MASK"
SECTION_BOARD_AFTER_LOCK = "BOARD_AFTER_LOCK"
SECTION_BOARD_AFTER_CLEAR = "BOARD_AFTER_CLEAR"
SECTION_LEGAL_MASK = "LEGAL_MASK"
SECTION_ACTION_SCORE = "ACTION_SCORE"
SECTION_ACTION_PROB = "ACTION_PROB"
SECTION_META = "META"

PRIORITY_NORMAL = "normal"
PRIORITY_NEAR_CLEAR = "near_clear"
PRIORITY_CLEAR = "clear"

KIND_LIST = ["I", "O", "T", "S", "Z", "J", "L"]
KIND_TO_ID = {kind: idx for idx, kind in enumerate(KIND_LIST)}
ID_TO_KIND = {idx: kind for kind, idx in KIND_TO_ID.items()}

# 注意：I 型方块固定为 3 格，不是标准 4 格
SHAPES: Dict[str, List[List[Tuple[int, int]]]] = {
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

MATRIX_SPECS = {
    SECTION_CURRENT_STATE: (28, 14, "int"),
    SECTION_TARGET_14x4: (14, 4, "int"),
    SECTION_LEGAL_TARGET_14x4: (14, 4, "int"),
    SECTION_SCORE_TARGET_14x4: (14, 4, "float"),
    SECTION_PROB_TARGET_14x4: (14, 4, "float"),
    SECTION_BOARD_BEFORE: (18, 14, "int"),
    SECTION_ACTIVE_MASK: (18, 14, "int"),
    SECTION_BOARD_AFTER_LOCK: (18, 14, "int"),
    SECTION_BOARD_AFTER_CLEAR: (18, 14, "int"),
    SECTION_LEGAL_MASK: (4, 14, "int"),
    SECTION_ACTION_SCORE: (4, 14, "float"),
    SECTION_ACTION_PROB: (4, 14, "float"),
}


# ============================================================
# Dataclasses
# ============================================================
@dataclass
class Task5ParsedSample:
    board_before: List[List[int]]
    active_mask: List[List[int]]
    board_after_lock: List[List[int]]
    board_after_clear: List[List[int]]

    legal_mask_4x14: List[List[int]]
    action_score_4x14: List[List[float]]
    action_prob_4x14: List[List[float]]

    kind: str
    kind_id: int
    cur_rot4: int
    cur_col14: int
    tgt_rot4: int
    tgt_col14: int
    target_action: int

    lines_cleared: int
    teacher_score: float
    priority_class: str


@dataclass
class TrainBatch:
    board_tensor: torch.Tensor         # [B, 2, 18, 14]
    state_tensor: torch.Tensor         # [B, 25]
    target_action: torch.Tensor        # [B]
    legal_mask_flat: torch.Tensor      # [B, 56]
    soft_target_flat: torch.Tensor     # [B, 56]
    sample_weight: torch.Tensor        # [B]
    lines_cleared: torch.Tensor        # [B]
    teacher_score: torch.Tensor        # [B]


# ============================================================
# Basic utilities
# ============================================================
def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def row_is_blank(row: Sequence[str]) -> bool:
    return all(str(cell).strip() == "" for cell in row)


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def flatten_4x14(matrix_4x14: Sequence[Sequence[float]]) -> List[float]:
    out: List[float] = []
    for rot in range(4):
        for col in range(BOARD_COLS):
            out.append(float(matrix_4x14[rot][col]))
    return out


def one_hot(index: int, size: int) -> List[float]:
    out = [0.0] * size
    if 0 <= index < size:
        out[index] = 1.0
    return out


def build_state_vector(
    kind_id: int,
    cur_rot4: int,
    cur_col14: int,
) -> List[float]:
    return (
        one_hot(kind_id, 7)
        + one_hot(cur_rot4, 4)
        + one_hot(cur_col14, BOARD_COLS)
    )


def occ_board(board: Sequence[Sequence[int]]) -> List[List[int]]:
    """
    把 board_before 从形状 ID 棋盘转成 occupancy 棋盘。
    这样更利于后续迁移到 exe，因为 exe 里通常只能稳拿到占用信息。
    """
    return [
        [1 if safe_int(cell) != 0 else 0 for cell in row]
        for row in board
    ]


def normalize_soft_target(
    probs: Sequence[float],
    legal_mask: Sequence[float],
) -> List[float]:
    """
    保证 soft target 只落在合法动作上，并归一化。
    如果输入异常，则退化成合法动作均匀分布。
    """
    filtered = [
        max(0.0, float(p)) if float(m) > 0.5 else 0.0
        for p, m in zip(probs, legal_mask)
    ]
    total = sum(filtered)
    if total > 0:
        return [p / total for p in filtered]

    legal_indices = [i for i, m in enumerate(legal_mask) if float(m) > 0.5]
    out = [0.0] * len(legal_mask)
    if not legal_indices:
        return out

    value = 1.0 / len(legal_indices)
    for idx in legal_indices:
        out[idx] = value
    return out


def priority_weight(
    priority_class: str,
    lines_cleared: int,
    normal_weight: float,
    near_clear_weight: float,
    clear_weight: float,
    clear_bonus_per_line: float,
) -> float:
    weight = normal_weight
    if priority_class == PRIORITY_NEAR_CLEAR:
        weight = near_clear_weight
    elif priority_class == PRIORITY_CLEAR:
        weight = clear_weight

    if lines_cleared > 0:
        weight += clear_bonus_per_line * float(lines_cleared)

    return float(weight)


# ============================================================
# CSV parser for task5 block format
# ============================================================
def parse_matrix_block(
    rows: List[List[str]],
    start_idx: int,
    row_count: int,
    col_count: int,
    value_type: str,
) -> Tuple[List[List[float]], int]:
    data: List[List[float]] = []
    idx = start_idx + 1

    for _ in range(row_count):
        if idx >= len(rows):
            raise ValueError("CSV ended unexpectedly while reading matrix block.")
        row = rows[idx]
        parsed_row: List[float] = []
        for col in range(col_count):
            cell = row[col] if col < len(row) else ""
            if value_type == "int":
                parsed_row.append(float(safe_int(cell)))
            else:
                parsed_row.append(float(safe_float(cell)))
        data.append(parsed_row)
        idx += 1

    while idx < len(rows) and row_is_blank(rows[idx]):
        idx += 1

    return data, idx


def parse_meta_block(
    rows: List[List[str]],
    start_idx: int,
) -> Tuple[Dict[str, str], int]:
    """
    解析 META block。

    当前 task5 写出的格式是：
    第 1 行：META,kind,kind_id,...
    第 2 行：META,I,0,...
    第 3 行：空行

    所以：
    - rows[start_idx]     是表头
    - rows[start_idx + 1] 是数值
    """
    if start_idx + 1 >= len(rows):
        raise ValueError("CSV ended unexpectedly while reading META block.")

    header = rows[start_idx]
    value = rows[start_idx + 1]

    meta: Dict[str, str] = {}
    for k, v in zip(header, value):
        key = str(k).strip()
        if key == "":
            continue
        meta[key] = str(v).strip()

    idx = start_idx + 2
    while idx < len(rows) and row_is_blank(rows[idx]):
        idx += 1

    return meta, idx


def finalize_sample(raw: Dict[str, object]) -> Task5ParsedSample:
    required = [
        SECTION_BOARD_BEFORE,
        SECTION_ACTIVE_MASK,
        SECTION_BOARD_AFTER_LOCK,
        SECTION_BOARD_AFTER_CLEAR,
        SECTION_LEGAL_MASK,
        SECTION_ACTION_SCORE,
        SECTION_ACTION_PROB,
        SECTION_META,
    ]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing sections in sample: {missing}")

    meta = raw[SECTION_META]
    assert isinstance(meta, dict)

    board_before = [
        [safe_int(x) for x in row]
        for row in raw[SECTION_BOARD_BEFORE]
    ]
    active_mask = [
        [safe_int(x) for x in row]
        for row in raw[SECTION_ACTIVE_MASK]
    ]
    board_after_lock = [
        [safe_int(x) for x in row]
        for row in raw[SECTION_BOARD_AFTER_LOCK]
    ]
    board_after_clear = [
        [safe_int(x) for x in row]
        for row in raw[SECTION_BOARD_AFTER_CLEAR]
    ]

    legal_mask_4x14 = [
        [safe_int(x) for x in row]
        for row in raw[SECTION_LEGAL_MASK]
    ]
    action_score_4x14 = [
        [safe_float(x) for x in row]
        for row in raw[SECTION_ACTION_SCORE]
    ]
    action_prob_4x14 = [
        [safe_float(x) for x in row]
        for row in raw[SECTION_ACTION_PROB]
    ]

    sample = Task5ParsedSample(
        board_before=board_before,
        active_mask=active_mask,
        board_after_lock=board_after_lock,
        board_after_clear=board_after_clear,
        legal_mask_4x14=legal_mask_4x14,
        action_score_4x14=action_score_4x14,
        action_prob_4x14=action_prob_4x14,
        kind=meta.get("kind", ""),
        kind_id=safe_int(meta.get("kind_id", -1), -1),
        cur_rot4=safe_int(meta.get("cur_rot4", -1), -1),
        cur_col14=safe_int(meta.get("cur_col14", -1), -1),
        tgt_rot4=safe_int(meta.get("tgt_rot4", -1), -1),
        tgt_col14=safe_int(meta.get("tgt_col14", -1), -1),
        target_action=safe_int(meta.get("target_action", -1), -1),
        lines_cleared=safe_int(meta.get("lines_cleared", 0), 0),
        teacher_score=safe_float(meta.get("teacher_score", 0.0), 0.0),
        priority_class=meta.get("priority_class", PRIORITY_NORMAL),
    )

    legal_flat = flatten_4x14(sample.legal_mask_4x14)
    if not (0 <= sample.target_action < ACTION_CLASSES):
        raise ValueError(f"Invalid target_action: {sample.target_action}")
    if legal_flat[sample.target_action] < 0.5:
        raise ValueError(
            "Target action is not legal. "
            f"target_action={sample.target_action}, kind={sample.kind}"
        )

    return sample


def parse_task5_block_csv(
    csv_path: str | Path,
    max_samples: Optional[int] = None,
) -> List[Task5ParsedSample]:
    rows: List[List[str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        rows = [row for row in reader]

    samples: List[Task5ParsedSample] = []
    current: Dict[str, object] = {}
    idx = 0

    while idx < len(rows):
        row = rows[idx]
        if row_is_blank(row):
            idx += 1
            continue

        title = str(row[0]).strip()

        if title in MATRIX_SPECS:
            row_count, col_count, value_type = MATRIX_SPECS[title]
            matrix, idx = parse_matrix_block(
                rows,
                idx,
                row_count=row_count,
                col_count=col_count,
                value_type=value_type,
            )
            current[title] = matrix
            continue

        if title == SECTION_META:
            meta, idx = parse_meta_block(rows, idx)
            current[SECTION_META] = meta
            samples.append(finalize_sample(current))
            current = {}
            if max_samples is not None and len(samples) >= max_samples:
                break
            continue

        idx += 1

    return samples


# ============================================================
# Dataset
# ============================================================
class Task5ActionDataset(Dataset):
    def __init__(
        self,
        samples: List[Task5ParsedSample],
        normal_weight: float = 1.0,
        near_clear_weight: float = 1.5,
        clear_weight: float = 3.0,
        clear_bonus_per_line: float = 0.35,
    ) -> None:
        self.samples = samples
        self.normal_weight = normal_weight
        self.near_clear_weight = near_clear_weight
        self.clear_weight = clear_weight
        self.clear_bonus_per_line = clear_bonus_per_line

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        s = self.samples[index]

        board_occ = occ_board(s.board_before)
        active_mask = [
            [1 if safe_int(cell) != 0 else 0 for cell in row]
            for row in s.active_mask
        ]

        board_tensor = torch.tensor(
            [board_occ, active_mask],
            dtype=torch.float32,
        )

        state_vec = build_state_vector(
            kind_id=s.kind_id,
            cur_rot4=s.cur_rot4,
            cur_col14=s.cur_col14,
        )
        state_tensor = torch.tensor(state_vec, dtype=torch.float32)

        legal_flat = flatten_4x14(s.legal_mask_4x14)
        soft_flat = flatten_4x14(s.action_prob_4x14)
        soft_flat = normalize_soft_target(soft_flat, legal_flat)

        legal_mask_tensor = torch.tensor(legal_flat, dtype=torch.float32)
        soft_target_tensor = torch.tensor(soft_flat, dtype=torch.float32)

        weight = priority_weight(
            priority_class=s.priority_class,
            lines_cleared=s.lines_cleared,
            normal_weight=self.normal_weight,
            near_clear_weight=self.near_clear_weight,
            clear_weight=self.clear_weight,
            clear_bonus_per_line=self.clear_bonus_per_line,
        )

        out = {
            "board_tensor": board_tensor,
            "state_tensor": state_tensor,
            "target_action": torch.tensor(s.target_action, dtype=torch.long),
            "legal_mask_flat": legal_mask_tensor,
            "soft_target_flat": soft_target_tensor,
            "sample_weight": torch.tensor(weight, dtype=torch.float32),
            "lines_cleared": torch.tensor(s.lines_cleared, dtype=torch.float32),
            "teacher_score": torch.tensor(s.teacher_score, dtype=torch.float32),
        }
        return out


def collate_batch(items: List[Dict[str, torch.Tensor]]) -> TrainBatch:
    return TrainBatch(
        board_tensor=torch.stack([x["board_tensor"] for x in items], dim=0),
        state_tensor=torch.stack([x["state_tensor"] for x in items], dim=0),
        target_action=torch.stack([x["target_action"] for x in items], dim=0),
        legal_mask_flat=torch.stack([x["legal_mask_flat"] for x in items], dim=0),
        soft_target_flat=torch.stack([x["soft_target_flat"] for x in items], dim=0),
        sample_weight=torch.stack([x["sample_weight"] for x in items], dim=0),
        lines_cleared=torch.stack([x["lines_cleared"] for x in items], dim=0),
        teacher_score=torch.stack([x["teacher_score"] for x in items], dim=0),
    )


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
            nn.MaxPool2d(kernel_size=2),        # 18x14 -> 9x7
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
        x = self.net(x)
        x = self.proj(x)
        return x


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

    def forward(
        self,
        board_tensor: torch.Tensor,
        state_tensor: torch.Tensor,
    ) -> torch.Tensor:
        board_feat = self.board_encoder(board_tensor)
        state_feat = self.state_encoder(state_tensor)
        fused = torch.cat([board_feat, state_feat], dim=1)
        logits = self.head(fused)
        return logits


# ============================================================
# Legal-action helpers for training / inference
# ============================================================
def apply_legal_mask(
    logits: torch.Tensor,
    legal_mask_flat: torch.Tensor,
    large_negative: float = -1e9,
) -> torch.Tensor:
    """
    对非法动作强行压低分数。
    legal_mask_flat: [B, 56] or [56]
    """
    if legal_mask_flat.dim() == 1:
        legal_mask_flat = legal_mask_flat.unsqueeze(0)
    return logits.masked_fill(legal_mask_flat <= 0.5, large_negative)


@dataclass
class Piece:
    kind: str
    rotation: int
    row: int
    col: int

    def cells(self) -> List[Tuple[int, int]]:
        rot_count = len(SHAPES[self.kind])
        rot_idx = self.rotation % rot_count
        offsets = SHAPES[self.kind][rot_idx]
        return [(self.row + r, self.col + c) for r, c in offsets]


def is_valid_position_on_board(
    board_occ: Sequence[Sequence[int]],
    piece: Piece,
) -> bool:
    for row_idx, col_idx in piece.cells():
        if col_idx < 0 or col_idx >= BOARD_COLS:
            return False
        if row_idx < 0 or row_idx >= BOARD_ROWS:
            return False
        if int(board_occ[row_idx][col_idx]) != 0:
            return False
    return True


def hard_drop_row_on_board(
    board_occ: Sequence[Sequence[int]],
    kind: str,
    rotation: int,
    anchor_col: int,
) -> Optional[int]:
    row = 0
    piece = Piece(kind=kind, rotation=rotation, row=row, col=anchor_col)
    if not is_valid_position_on_board(board_occ, piece):
        return None

    while True:
        next_piece = Piece(
            kind=kind,
            rotation=rotation,
            row=row + 1,
            col=anchor_col,
        )
        if is_valid_position_on_board(board_occ, next_piece):
            row += 1
        else:
            return row


def leftmost_col_of_piece(piece: Piece) -> int:
    return min(col for _, col in piece.cells())


def left_col14_to_anchor_col(kind: str, rot4: int, left_col14: int) -> int:
    """
    网络输出是 left_col14。
    控制层如果需要 anchor_col，用这个函数转换。
    """
    rotation_count = len(SHAPES[kind])
    rot_idx = rot4 % rotation_count
    offsets = SHAPES[kind][rot_idx]
    min_offset_col = min(col for _, col in offsets)
    return left_col14 - min_offset_col


def build_legal_mask_4x14_from_board(
    board_occ: Sequence[Sequence[int]],
    kind: str,
) -> List[List[int]]:
    legal = [[0 for _ in range(BOARD_COLS)] for _ in range(4)]
    rotations = len(SHAPES[kind])

    checked = set()
    for rot in range(rotations):
        offsets = SHAPES[kind][rot]
        min_col = min(c for _, c in offsets)
        max_col = max(c for _, c in offsets)

        for anchor_col in range(-min_col, BOARD_COLS - max_col):
            key = (rot, anchor_col, tuple(sorted(offsets)))
            if key in checked:
                continue
            checked.add(key)

            final_row = hard_drop_row_on_board(
                board_occ,
                kind=kind,
                rotation=rot,
                anchor_col=anchor_col,
            )
            if final_row is None:
                continue

            final_piece = Piece(
                kind=kind,
                rotation=rot,
                row=final_row,
                col=anchor_col,
            )
            left_col14 = leftmost_col_of_piece(final_piece)
            if 0 <= left_col14 < BOARD_COLS:
                legal[rot % 4][left_col14] = 1

    return legal


def decode_action(action_idx: int) -> Tuple[int, int]:
    rot4 = int(action_idx) // BOARD_COLS
    col14 = int(action_idx) % BOARD_COLS
    return rot4, col14


# ============================================================
# Loss / metrics
# ============================================================
def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    denom = weights.sum().clamp_min(1e-8)
    return (values * weights).sum() / denom


def compute_losses(
    logits: torch.Tensor,
    target_action: torch.Tensor,
    legal_mask_flat: torch.Tensor,
    soft_target_flat: torch.Tensor,
    sample_weight: torch.Tensor,
    ce_weight: float,
    soft_weight: float,
) -> Dict[str, torch.Tensor]:
    masked_logits = apply_legal_mask(logits, legal_mask_flat)

    ce_per = torch_f.cross_entropy(
        masked_logits,
        target_action,
        reduction="none",
    )

    log_probs = torch_f.log_softmax(masked_logits, dim=1)
    soft_per = -(soft_target_flat * log_probs).sum(dim=1)

    ce_loss = weighted_mean(ce_per, sample_weight)
    soft_loss = weighted_mean(soft_per, sample_weight)
    total = ce_weight * ce_loss + soft_weight * soft_loss

    return {
        "loss": total,
        "ce_loss": ce_loss,
        "soft_loss": soft_loss,
        "masked_logits": masked_logits,
    }


@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,
    masked_logits: torch.Tensor,
    target_action: torch.Tensor,
    legal_mask_flat: torch.Tensor,
) -> Dict[str, float]:
    pred_raw = logits.argmax(dim=1)
    pred_masked = masked_logits.argmax(dim=1)
    top3 = masked_logits.topk(k=min(3, masked_logits.size(1)), dim=1).indices

    target_acc = (pred_masked == target_action).float().mean().item()
    teacher_match = target_acc

    raw_legal = []
    for i in range(logits.size(0)):
        raw_legal.append(float(legal_mask_flat[i, pred_raw[i]].item() > 0.5))
    raw_legal_rate = float(sum(raw_legal) / max(1, len(raw_legal)))

    top3_hit = []
    for i in range(logits.size(0)):
        top3_hit.append(float(int(target_action[i].item()) in top3[i].tolist()))
    top3_hit_rate = float(sum(top3_hit) / max(1, len(top3_hit)))

    return {
        "target_acc": target_acc,
        "teacher_match": teacher_match,
        "raw_legal_rate": raw_legal_rate,
        "top3_hit_rate": top3_hit_rate,
    }


# ============================================================
# Train / eval loops
# ============================================================
def move_batch_to_device(batch: TrainBatch, device: torch.device) -> TrainBatch:
    return TrainBatch(
        board_tensor=batch.board_tensor.to(device),
        state_tensor=batch.state_tensor.to(device),
        target_action=batch.target_action.to(device),
        legal_mask_flat=batch.legal_mask_flat.to(device),
        soft_target_flat=batch.soft_target_flat.to(device),
        sample_weight=batch.sample_weight.to(device),
        lines_cleared=batch.lines_cleared.to(device),
        teacher_score=batch.teacher_score.to(device),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    ce_weight: float,
    soft_weight: float,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    ce_sum = 0.0
    soft_sum = 0.0
    acc_sum = 0.0
    match_sum = 0.0
    legal_sum = 0.0
    top3_sum = 0.0
    count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        logits = model(batch.board_tensor, batch.state_tensor)
        losses = compute_losses(
            logits=logits,
            target_action=batch.target_action,
            legal_mask_flat=batch.legal_mask_flat,
            soft_target_flat=batch.soft_target_flat,
            sample_weight=batch.sample_weight,
            ce_weight=ce_weight,
            soft_weight=soft_weight,
        )

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

        metrics = compute_metrics(
            logits=logits.detach(),
            masked_logits=losses["masked_logits"].detach(),
            target_action=batch.target_action,
            legal_mask_flat=batch.legal_mask_flat,
        )

        bs = batch.target_action.size(0)
        loss_sum += float(losses["loss"].item()) * bs
        ce_sum += float(losses["ce_loss"].item()) * bs
        soft_sum += float(losses["soft_loss"].item()) * bs
        acc_sum += metrics["target_acc"] * bs
        match_sum += metrics["teacher_match"] * bs
        legal_sum += metrics["raw_legal_rate"] * bs
        top3_sum += metrics["top3_hit_rate"] * bs
        count += bs

    if count == 0:
        return {
            "loss": 0.0,
            "ce_loss": 0.0,
            "soft_loss": 0.0,
            "target_acc": 0.0,
            "teacher_match": 0.0,
            "raw_legal_rate": 0.0,
            "top3_hit_rate": 0.0,
        }

    return {
        "loss": loss_sum / count,
        "ce_loss": ce_sum / count,
        "soft_loss": soft_sum / count,
        "target_acc": acc_sum / count,
        "teacher_match": match_sum / count,
        "raw_legal_rate": legal_sum / count,
        "top3_hit_rate": top3_sum / count,
    }


# ============================================================
# Inference API for task3 / task1
# ============================================================
@torch.no_grad()
def predict_action_from_state(
    model: nn.Module,
    device: torch.device,
    board_before_18x14: Sequence[Sequence[int]],
    active_mask_18x14: Sequence[Sequence[int]],
    kind_id: int,
    cur_rot4: int,
    cur_col14: int,
    legal_mask_4x14: Optional[Sequence[Sequence[int]]] = None,
    kind: Optional[str] = None,
) -> Dict[str, int]:
    """
    给 task3 / task1 用的推理入口。

    返回：
    - action_idx
    - pred_rot4
    - pred_col14   (这里是 left_col14)
    - pred_anchor_col  (如果 kind 可用，会一并返回)
    """
    board_occ = occ_board(board_before_18x14)
    active_mask = [
        [1 if safe_int(cell) != 0 else 0 for cell in row]
        for row in active_mask_18x14
    ]

    if legal_mask_4x14 is None:
        if kind is None:
            raise ValueError("If legal_mask_4x14 is None, kind must be provided.")
        legal_mask_4x14 = build_legal_mask_4x14_from_board(board_occ, kind)

    board_tensor = torch.tensor(
        [[board_occ, active_mask]],
        dtype=torch.float32,
        device=device,
    )
    state_tensor = torch.tensor(
        [build_state_vector(kind_id, cur_rot4, cur_col14)],
        dtype=torch.float32,
        device=device,
    )
    legal_flat = torch.tensor(
        [flatten_4x14(legal_mask_4x14)],
        dtype=torch.float32,
        device=device,
    )

    logits = model(board_tensor, state_tensor)
    masked_logits = apply_legal_mask(logits, legal_flat)
    action_idx = int(masked_logits.argmax(dim=1).item())
    pred_rot4, pred_col14 = decode_action(action_idx)

    out = {
        "action_idx": action_idx,
        "pred_rot4": pred_rot4,
        "pred_col14": pred_col14,
    }
    if kind is not None:
        out["pred_anchor_col"] = left_col14_to_anchor_col(
            kind=kind,
            rot4=pred_rot4,
            left_col14=pred_col14,
        )
    return out


# ============================================================
# Checkpoint IO
# ============================================================
def save_checkpoint(
    model: nn.Module,
    out_path: str | Path,
    epoch: int,
    args: argparse.Namespace,
    metrics: Dict[str, float],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": vars(args),
        "board_rows": BOARD_ROWS,
        "board_cols": BOARD_COLS,
        "action_classes": ACTION_CLASSES,
        "kind_list": KIND_LIST,
        "i_piece_is_3_cells": True,
    }
    torch.save(payload, out_path)


def load_model_checkpoint(
    model_path: str | Path,
    device: torch.device,
) -> Tuple[ActionPolicyNet, Dict[str, object]]:
    ckpt = torch.load(model_path, map_location=device)
    model = ActionPolicyNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


# ============================================================
# CLI commands
# ============================================================
def command_inspect(args: argparse.Namespace) -> None:
    samples = parse_task5_block_csv(args.csv, max_samples=args.limit)
    print(f"[INFO] parsed samples={len(samples)}")
    for idx, sample in enumerate(samples[: args.limit]):
        legal_flat = flatten_4x14(sample.legal_mask_4x14)
        soft_flat = normalize_soft_target(
            flatten_4x14(sample.action_prob_4x14),
            legal_flat,
        )
        print("-" * 72)
        print(f"sample={idx}")
        print(
            f"kind={sample.kind} kind_id={sample.kind_id} "
            f"cur_rot4={sample.cur_rot4} cur_col14={sample.cur_col14}"
        )
        print(
            f"tgt_rot4={sample.tgt_rot4} tgt_col14={sample.tgt_col14} "
            f"target_action={sample.target_action}"
        )
        print(
            f"lines_cleared={sample.lines_cleared} "
            f"teacher_score={sample.teacher_score:.6f} "
            f"priority_class={sample.priority_class}"
        )
        print(
            f"target_is_legal={int(legal_flat[sample.target_action] > 0.5)} "
            f"soft_sum={sum(soft_flat):.6f}"
        )
    print("-" * 72)


def split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_train = min(max(n_train, 1), n - 2) if n >= 3 else max(1, n - 1)
    n_val = min(max(n_val, 1), n - n_train - 1) if n >= 3 else 0

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    if len(test_idx) == 0 and len(val_idx) > 1:
        test_idx.append(val_idx.pop())
    if len(val_idx) == 0 and len(test_idx) > 1:
        val_idx.append(test_idx.pop())

    return train_idx, val_idx, test_idx


def command_train(args: argparse.Namespace) -> None:
    ensure_dir(args.out_dir)
    seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"[INFO] device={device}")

    samples = parse_task5_block_csv(args.csv, max_samples=args.max_samples)
    if len(samples) < 10:
        raise ValueError("Sample count is too small. Need at least 10 samples.")

    print(f"[INFO] samples total={len(samples)}")

    dataset = Task5ActionDataset(
        samples=samples,
        normal_weight=args.normal_weight,
        near_clear_weight=args.near_clear_weight,
        clear_weight=args.clear_weight,
        clear_bonus_per_line=args.clear_bonus_per_line,
    )

    train_idx, val_idx, test_idx = split_indices(
        n=len(dataset),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(
        f"[INFO] split train={len(train_idx)} "
        f"val={len(val_idx)} test={len(test_idx)}"
    )

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_batch,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = ActionPolicyNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )

    history_path = os.path.join(args.out_dir, "history.csv")
    best_path = os.path.join(args.out_dir, "model_best_action.pt")
    last_path = os.path.join(args.out_dir, "model_last_action.pt")
    config_path = os.path.join(args.out_dir, "train_config.json")

    with open(config_path, "w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    best_score = -1.0
    best_val_loss = float("inf")

    with open(history_path, "w", encoding="utf-8", newline="") as fp_hist:
        writer = csv.writer(fp_hist)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_ce",
                "train_soft",
                "train_acc",
                "train_legal",
                "train_top3",
                "val_loss",
                "val_ce",
                "val_soft",
                "val_acc",
                "val_legal",
                "val_top3",
                "lr",
            ]
        )

        print(f"[INFO] saving best model to: {best_path}")

        for epoch in range(1, args.epochs + 1):
            train_metrics = run_epoch(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                ce_weight=args.ce_weight,
                soft_weight=args.soft_weight,
            )
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                ce_weight=args.ce_weight,
                soft_weight=args.soft_weight,
            )

            lr_now = optimizer.param_groups[0]["lr"]
            writer.writerow(
                [
                    epoch,
                    f"{train_metrics['loss']:.6f}",
                    f"{train_metrics['ce_loss']:.6f}",
                    f"{train_metrics['soft_loss']:.6f}",
                    f"{train_metrics['target_acc']:.6f}",
                    f"{train_metrics['raw_legal_rate']:.6f}",
                    f"{train_metrics['top3_hit_rate']:.6f}",
                    f"{val_metrics['loss']:.6f}",
                    f"{val_metrics['ce_loss']:.6f}",
                    f"{val_metrics['soft_loss']:.6f}",
                    f"{val_metrics['target_acc']:.6f}",
                    f"{val_metrics['raw_legal_rate']:.6f}",
                    f"{val_metrics['top3_hit_rate']:.6f}",
                    f"{lr_now:.8f}",
                ]
            )
            fp_hist.flush()

            print(
                f"Epoch {epoch:03d} | "
                f"train loss={train_metrics['loss']:.4f} "
                f"ce={train_metrics['ce_loss']:.4f} "
                f"soft={train_metrics['soft_loss']:.4f} "
                f"target_acc={train_metrics['target_acc']:.4f} "
                f"legal={train_metrics['raw_legal_rate']:.4f} "
                f"top3={train_metrics['top3_hit_rate']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} "
                f"ce={val_metrics['ce_loss']:.4f} "
                f"soft={val_metrics['soft_loss']:.4f} "
                f"target_acc={val_metrics['target_acc']:.4f} "
                f"legal={val_metrics['raw_legal_rate']:.4f} "
                f"top3={val_metrics['top3_hit_rate']:.4f}"
            )

            improved = (
                val_metrics["target_acc"] > best_score
                or (
                    math.isclose(val_metrics["target_acc"], best_score, rel_tol=1e-9)
                    and val_metrics["loss"] < best_val_loss
                )
            )
            if improved:
                best_score = val_metrics["target_acc"]
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    model=model,
                    out_path=best_path,
                    epoch=epoch,
                    args=args,
                    metrics=val_metrics,
                )

            save_checkpoint(
                model=model,
                out_path=last_path,
                epoch=epoch,
                args=args,
                metrics=val_metrics,
            )
            scheduler.step()

    print("[INFO] evaluating best model on test split...")
    best_model, _ = load_model_checkpoint(best_path, device)
    test_metrics = run_epoch(
        model=best_model,
        loader=test_loader,
        device=device,
        optimizer=None,
        ce_weight=args.ce_weight,
        soft_weight=args.soft_weight,
    )
    print(
        f"[TEST] loss={test_metrics['loss']:.4f} "
        f"ce={test_metrics['ce_loss']:.4f} "
        f"soft={test_metrics['soft_loss']:.4f} "
        f"target_acc={test_metrics['target_acc']:.4f} "
        f"legal={test_metrics['raw_legal_rate']:.4f} "
        f"top3={test_metrics['top3_hit_rate']:.4f}"
    )


def command_eval(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"[INFO] device={device}")

    samples = parse_task5_block_csv(args.csv, max_samples=args.max_samples)
    dataset = Task5ActionDataset(samples=samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    model, ckpt = load_model_checkpoint(args.model, device)
    print(f"[INFO] loaded model epoch={ckpt.get('epoch', -1)}")
    metrics = run_epoch(
        model=model,
        loader=loader,
        device=device,
        optimizer=None,
        ce_weight=args.ce_weight,
        soft_weight=args.soft_weight,
    )
    print(
        f"[EVAL] loss={metrics['loss']:.4f} "
        f"ce={metrics['ce_loss']:.4f} "
        f"soft={metrics['soft_loss']:.4f} "
        f"target_acc={metrics['target_acc']:.4f} "
        f"legal={metrics['raw_legal_rate']:.4f} "
        f"top3={metrics['top3_hit_rate']:.4f}"
    )


def command_predict_csv_sample(args: argparse.Namespace) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    samples = parse_task5_block_csv(args.csv, max_samples=None)

    if not (0 <= args.index < len(samples)):
        raise IndexError(f"index out of range: {args.index}, total={len(samples)}")

    sample = samples[args.index]
    model, _ = load_model_checkpoint(args.model, device)

    pred = predict_action_from_state(
        model=model,
        device=device,
        board_before_18x14=sample.board_before,
        active_mask_18x14=sample.active_mask,
        kind_id=sample.kind_id,
        cur_rot4=sample.cur_rot4,
        cur_col14=sample.cur_col14,
        legal_mask_4x14=sample.legal_mask_4x14,
        kind=sample.kind,
    )

    print(f"[SAMPLE] index={args.index}")
    print(
        f"[GT] kind={sample.kind} cur_rot4={sample.cur_rot4} "
        f"cur_col14={sample.cur_col14} "
        f"tgt_rot4={sample.tgt_rot4} tgt_col14={sample.tgt_col14} "
        f"target_action={sample.target_action}"
    )
    print(
        f"[PR] action_idx={pred['action_idx']} "
        f"pred_rot4={pred['pred_rot4']} "
        f"pred_col14={pred['pred_col14']} "
        f"pred_anchor_col={pred.get('pred_anchor_col', -999)}"
    )


# ============================================================
# Argparse
# ============================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Task6: train a 56-class action network from task5 block CSV.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_inspect = subparsers.add_parser("inspect")
    p_inspect.add_argument("--csv", type=str, required=True)
    p_inspect.add_argument("--limit", type=int, default=3)

    p_train = subparsers.add_parser("train")
    p_train.add_argument("--csv", type=str, required=True)
    p_train.add_argument("--out_dir", type=str, default="out_task6")
    p_train.add_argument("--max_samples", type=int, default=0)
    p_train.add_argument("--epochs", type=int, default=80)
    p_train.add_argument("--batch_size", type=int, default=256)
    p_train.add_argument("--lr", type=float, default=5e-4)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--ce_weight", type=float, default=1.0)
    p_train.add_argument("--soft_weight", type=float, default=0.35)
    p_train.add_argument("--train_ratio", type=float, default=0.8)
    p_train.add_argument("--val_ratio", type=float, default=0.1)
    p_train.add_argument("--normal_weight", type=float, default=1.0)
    p_train.add_argument("--near_clear_weight", type=float, default=2.5)
    p_train.add_argument("--clear_weight", type=float, default=5.0)
    p_train.add_argument("--clear_bonus_per_line", type=float, default=0.8)
    p_train.add_argument("--num_workers", type=int, default=0)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--cpu", action="store_true")

    p_eval = subparsers.add_parser("eval")
    p_eval.add_argument("--csv", type=str, required=True)
    p_eval.add_argument("--model", type=str, required=True)
    p_eval.add_argument("--max_samples", type=int, default=0)
    p_eval.add_argument("--batch_size", type=int, default=256)
    p_eval.add_argument("--num_workers", type=int, default=0)
    p_eval.add_argument("--ce_weight", type=float, default=1.0)
    p_eval.add_argument("--soft_weight", type=float, default=0.7)
    p_eval.add_argument("--cpu", action="store_true")

    p_pred = subparsers.add_parser("predict_csv_sample")
    p_pred.add_argument("--csv", type=str, required=True)
    p_pred.add_argument("--model", type=str, required=True)
    p_pred.add_argument("--index", type=int, default=0)
    p_pred.add_argument("--cpu", action="store_true")

    return parser


# ============================================================
# Main
# ============================================================
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if hasattr(args, "max_samples") and safe_int(getattr(args, "max_samples", 0)) <= 0:
        setattr(args, "max_samples", None)

    if args.command == "inspect":
        command_inspect(args)
        return

    if args.command == "train":
        command_train(args)
        return

    if args.command == "eval":
        command_eval(args)
        return

    if args.command == "predict_csv_sample":
        command_predict_csv_sample(args)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()