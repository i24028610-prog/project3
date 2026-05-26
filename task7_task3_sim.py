from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import pygame
import torch
import torch.nn as nn


# =========================
# Config
# =========================
CELL_SIZE = 28
BOARD_ROWS = 18
BOARD_COLS = 14

FPS = 60
AUTO_STEP_MS = 320

WINDOW_BG = (245, 247, 250)
GRID_LINE_COLOR = (210, 214, 220)
TEXT_COLOR = (35, 42, 52)
PANEL_BG = (232, 236, 242)
BORDER_COLOR = (60, 70, 85)
PATH_COLOR = (255, 140, 80)
TARGET_COLOR = (220, 70, 70)
LEGAL_COLOR = (70, 180, 90)

TOP_MARGIN = 60
BOTTOM_MARGIN = 110
LEFT_MARGIN = 40
RIGHT_PANEL_WIDTH = 360

WINDOW_WIDTH = (
    LEFT_MARGIN + BOARD_COLS * CELL_SIZE + 40 + RIGHT_PANEL_WIDTH + 80
)
WINDOW_HEIGHT = TOP_MARGIN + BOARD_ROWS * CELL_SIZE + BOTTOM_MARGIN

PIECE_ORDER = ["I", "O", "T", "S", "Z", "J", "L"]
PIECE_TO_INDEX = {kind: idx for idx, kind in enumerate(PIECE_ORDER)}

STATE_VEC_DIM = 7 + 4 + 14
ACTION_CLASSES = 56


# =========================
# Tetromino definitions
# 注意：必须和 task6 完全一致
# I 型块是 3 格，不是标准 4 格
# 并且竖直形状用 col offset = 1
# 这样 left_col14 语义才和 task6 保持一致
# =========================
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

SHAPE_COLORS: Dict[str, Tuple[int, int, int]] = {
    "I": (66, 200, 245),
    "O": (247, 208, 70),
    "T": (180, 120, 245),
    "S": (94, 204, 120),
    "Z": (235, 99, 99),
    "J": (92, 125, 245),
    "L": (245, 165, 84),
}

SHAPE_ID: Dict[str, int] = {
    "I": 1,
    "O": 2,
    "T": 3,
    "S": 4,
    "Z": 5,
    "J": 6,
    "L": 7,
}

ID_TO_KIND: Dict[int, str] = {v: k for k, v in SHAPE_ID.items()}


# =========================
# Task6 model
# =========================
class BoardEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 18x14 -> 9x7
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
        return self.head(fused)


# =========================
# Piece
# =========================
@dataclass
class Piece:
    kind: str
    rotation: int
    row: int
    col: int

    def cells(self) -> List[Tuple[int, int]]:
        rotation_count = len(SHAPES[self.kind])
        rotation_index = self.rotation % rotation_count
        offsets = SHAPES[self.kind][rotation_index]
        return [(self.row + row, self.col + col) for row, col in offsets]


# =========================
# Shared helpers
# =========================
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
    return [
        [1 if int(cell) != 0 else 0 for cell in row]
        for row in board
    ]


def build_active_mask(piece: Piece) -> List[List[int]]:
    mask = [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
    for row, col in piece.cells():
        if 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS:
            mask[row][col] = 1
    return mask


def flatten_4x14(matrix_4x14: Sequence[Sequence[float]]) -> List[float]:
    out: List[float] = []
    for rot in range(4):
        for col in range(BOARD_COLS):
            out.append(float(matrix_4x14[rot][col]))
    return out


def decode_action(action_idx: int) -> Tuple[int, int]:
    rot4 = int(action_idx) // BOARD_COLS
    col14 = int(action_idx) % BOARD_COLS
    return rot4, col14


def apply_legal_mask(
    logits: torch.Tensor,
    legal_mask_flat: torch.Tensor,
    large_negative: float = -1e9,
) -> torch.Tensor:
    if legal_mask_flat.dim() == 1:
        legal_mask_flat = legal_mask_flat.unsqueeze(0)
    return logits.masked_fill(legal_mask_flat <= 0.5, large_negative)


def leftmost_col(piece: Piece) -> int:
    return min(col for _, col in piece.cells())


def left_col14_to_anchor_col(kind: str, rot4: int, left_col14: int) -> int:
    rotation_count = len(SHAPES[kind])
    rot_idx = rot4 % rotation_count
    offsets = SHAPES[kind][rot_idx]
    min_offset_col = min(col for _, col in offsets)
    return left_col14 - min_offset_col


def is_valid_piece_on_board(board_occ: Sequence[Sequence[int]], piece: Piece) -> bool:
    for row, col in piece.cells():
        if col < 0 or col >= BOARD_COLS:
            return False
        if row < 0 or row >= BOARD_ROWS:
            return False
        if int(board_occ[row][col]) != 0:
            return False
    return True


def hard_drop_row_on_board(
    board_occ: Sequence[Sequence[int]],
    kind: str,
    rotation: int,
    col: int,
) -> Optional[int]:
    row = 0
    piece = Piece(kind=kind, rotation=rotation, row=row, col=col)
    if not is_valid_piece_on_board(board_occ, piece):
        return None

    while True:
        next_piece = Piece(kind=kind, rotation=rotation, row=row + 1, col=col)
        if is_valid_piece_on_board(board_occ, next_piece):
            row += 1
        else:
            return row


def simulate_piece_on_board(
    board_occ: Sequence[Sequence[int]],
    piece: Piece,
) -> Optional[Tuple[List[List[int]], int]]:
    out = [[1 if int(cell) != 0 else 0 for cell in row] for row in board_occ]
    for row, col in piece.cells():
        if row < 0 or row >= BOARD_ROWS or col < 0 or col >= BOARD_COLS:
            return None
        if out[row][col] != 0:
            return None
        out[row][col] = 1

    kept_rows = [row[:] for row in out if sum(row) < BOARD_COLS]
    cleared = BOARD_ROWS - len(kept_rows)
    while len(kept_rows) < BOARD_ROWS:
        kept_rows.insert(0, [0 for _ in range(BOARD_COLS)])
    return kept_rows, cleared


def column_heights(board_occ: Sequence[Sequence[int]]) -> List[int]:
    heights: List[int] = []
    for col in range(BOARD_COLS):
        height = 0
        for row in range(BOARD_ROWS):
            if int(board_occ[row][col]) != 0:
                height = BOARD_ROWS - row
                break
        heights.append(height)
    return heights


def count_holes(board_occ: Sequence[Sequence[int]]) -> int:
    holes = 0
    for col in range(BOARD_COLS):
        seen_filled = False
        for row in range(BOARD_ROWS):
            if int(board_occ[row][col]) != 0:
                seen_filled = True
            elif seen_filled:
                holes += 1
    return holes


def count_row_transitions(board_occ: Sequence[Sequence[int]]) -> int:
    total = 0
    for row in range(BOARD_ROWS):
        prev = 1
        for col in range(BOARD_COLS):
            cur = 1 if int(board_occ[row][col]) != 0 else 0
            if cur != prev:
                total += 1
            prev = cur
        if prev == 0:
            total += 1
    return total


def count_col_transitions(board_occ: Sequence[Sequence[int]]) -> int:
    total = 0
    for col in range(BOARD_COLS):
        prev = 1
        for row in range(BOARD_ROWS):
            cur = 1 if int(board_occ[row][col]) != 0 else 0
            if cur != prev:
                total += 1
            prev = cur
        if prev == 0:
            total += 1
    return total


def count_near_clear_rows(board_occ: Sequence[Sequence[int]]) -> int:
    return sum(1 for row in board_occ if sum(1 for cell in row if int(cell) != 0) >= BOARD_COLS - 1)


def board_features(board_occ: Sequence[Sequence[int]]) -> Dict[str, float]:
    heights = column_heights(board_occ)
    bump = sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))
    return {
        "agg_h": float(sum(heights)),
        "max_h": float(max(heights) if heights else 0),
        "holes": float(count_holes(board_occ)),
        "bump": float(bump),
        "near": float(count_near_clear_rows(board_occ)),
        "row_t": float(count_row_transitions(board_occ)),
        "col_t": float(count_col_transitions(board_occ)),
    }


def placement_safety_score(
    board_before: Sequence[Sequence[int]],
    board_after: Sequence[Sequence[int]],
    lines: int,
) -> float:
    before = board_features(board_before)
    after = board_features(board_after)

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


def normalize_scores(values: List[float]) -> List[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    std = var ** 0.5
    if std < 1e-6:
        return [0.0 for _ in values]
    return [(x - mean) / std for x in values]


# =========================
# NN policy wrapper
# =========================
class NNPolicy:
    def __init__(
        self,
        model_path: str,
        cpu: bool = False,
        policy_topk: int = 8,
        safety_weight: float = 0.70,
    ) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not cpu else "cpu"
        )
        self.policy_topk = max(1, int(policy_topk))
        self.safety_weight = float(safety_weight)
        self.model = ActionPolicyNet().to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                state_dict = checkpoint["model"]
            elif "net" in checkpoint and isinstance(checkpoint["net"], dict):
                state_dict = checkpoint["net"]
            else:
                state_dict = checkpoint
        else:
            raise ValueError("checkpoint format not supported")

        model_state = self.model.state_dict()
        loadable_state: Dict[str, torch.Tensor] = {}
        skipped_keys: List[str] = []

        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                loadable_state[k] = v
            else:
                skipped_keys.append(k)

        model_state.update(loadable_state)
        self.model.load_state_dict(model_state, strict=False)
        self.model.eval()

        self.model_path = model_path

        print(f"[LOAD] model_path={model_path}")
        print(f"[LOAD] loaded_tensors={len(loadable_state)}")
        print(f"[LOAD] skipped_tensors={len(skipped_keys)}")
        print(f"[LOAD] first_skipped_keys={skipped_keys[:12]}")
        print(f"[LOAD] policy_topk={self.policy_topk}")
        print(f"[LOAD] safety_weight={self.safety_weight:.3f}")

    @torch.no_grad()
    def predict(
        self,
        board_before: Sequence[Sequence[int]],
        active_mask: Sequence[Sequence[int]],
        kind: str,
        kind_id: int,
        cur_rot4: int,
        cur_col14: int,
        legal_mask_4x14: Sequence[Sequence[int]],
    ) -> Dict[str, object]:
        board_occ = occ_board(board_before)

        board_tensor = torch.tensor(
            [[board_occ, active_mask]],
            dtype=torch.float32,
            device=self.device,
        )
        state_tensor = torch.tensor(
            [build_state_vector(kind_id, cur_rot4, cur_col14)],
            dtype=torch.float32,
            device=self.device,
        )
        legal_flat = torch.tensor(
            [flatten_4x14(legal_mask_4x14)],
            dtype=torch.float32,
            device=self.device,
        )

        logits = self.model(board_tensor, state_tensor)
        masked_logits = apply_legal_mask(logits, legal_flat)

        raw_action = int(logits.argmax(dim=1).item())
        policy_argmax_action = int(masked_logits.argmax(dim=1).item())
        masked_action = self.choose_safe_action(
            board_occ=board_occ,
            kind=kind,
            legal_mask_4x14=legal_mask_4x14,
            masked_logits=masked_logits,
            fallback_action=policy_argmax_action,
        )

        pred_rot4, pred_col14 = decode_action(masked_action)
        pred_anchor_col = left_col14_to_anchor_col(
            kind=kind,
            rot4=pred_rot4,
            left_col14=pred_col14,
        )

        return {
            "raw_action": raw_action,
            "masked_action": masked_action,
            "policy_argmax_action": policy_argmax_action,
            "pred_rot4": pred_rot4,
            "pred_col14": pred_col14,
            "pred_anchor_col": pred_anchor_col,
            "logits": logits.squeeze(0).detach().cpu(),
            "masked_logits": masked_logits.squeeze(0).detach().cpu(),
        }

    def choose_safe_action(
        self,
        board_occ: Sequence[Sequence[int]],
        kind: str,
        legal_mask_4x14: Sequence[Sequence[int]],
        masked_logits: torch.Tensor,
        fallback_action: int,
    ) -> int:
        legal_flat = flatten_4x14(legal_mask_4x14)
        legal_actions = [idx for idx, value in enumerate(legal_flat) if value > 0.5]
        if not legal_actions:
            return int(fallback_action)

        topk = min(self.policy_topk, len(legal_actions))
        top = torch.topk(masked_logits, k=topk, dim=1)
        top_actions = [int(x) for x in top.indices[0].detach().cpu().tolist()]
        top_values = [float(x) for x in top.values[0].detach().cpu().tolist()]

        candidates: List[Dict[str, float]] = []
        for action_idx, policy_value in zip(top_actions, top_values):
            rot4, left_col14 = decode_action(action_idx)
            anchor_col = left_col14_to_anchor_col(kind, rot4, left_col14)
            final_row = hard_drop_row_on_board(board_occ, kind, rot4, anchor_col)
            if final_row is None:
                continue

            final_piece = Piece(kind=kind, rotation=rot4, row=final_row, col=anchor_col)
            if leftmost_col(final_piece) != left_col14:
                continue

            simulated = simulate_piece_on_board(board_occ, final_piece)
            if simulated is None:
                continue

            board_after, lines = simulated
            candidates.append(
                {
                    "action": float(action_idx),
                    "policy": float(policy_value),
                    "safety": placement_safety_score(board_occ, board_after, lines),
                }
            )

        if not candidates:
            return int(fallback_action)

        policy_scores = normalize_scores([float(row["policy"]) for row in candidates])
        safety_scores = normalize_scores([float(row["safety"]) for row in candidates])
        before = board_features(board_occ)
        danger = before["max_h"] >= 10.0 or before["holes"] >= 3.0 or before["bump"] >= 18.0
        safety_weight = self.safety_weight * (1.35 if danger else 1.0)

        best_i = 0
        best_score = -1e18
        for i, row in enumerate(candidates):
            total = policy_scores[i] + safety_weight * safety_scores[i]
            if total > best_score:
                best_score = total
                best_i = i

        return int(candidates[best_i]["action"])


# =========================
# Game
# =========================
class TetrisGame:
    def __init__(
        self,
        policy: NNPolicy,
        rng: random.Random,
        rng_mode: str,
        seed_value: Optional[int],
    ) -> None:
        self.policy = policy
        self.rng = rng
        self.rng_mode = rng_mode
        self.seed_value = seed_value

        self.board: List[List[int]] = [
            [0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)
        ]
        self.score = 0
        self.lines_cleared_total = 0
        self.pieces_placed = 0
        self.game_over = False

        self.target_rotation: Optional[int] = None
        self.target_left_col14: Optional[int] = None
        self.target_anchor_col: Optional[int] = None
        self.target_final_row: Optional[int] = None
        self.new_piece_spawned = True

        self.action_queue: Deque[str] = deque()
        self.last_action = "NONE"

        self.last_pred_raw_action: Optional[int] = None
        self.last_pred_masked_action: Optional[int] = None
        self.last_pred_rot4: Optional[int] = None
        self.last_pred_col14: Optional[int] = None
        self.last_pred_anchor_col: Optional[int] = None
        self.last_legal_mask_4x14: Optional[List[List[int]]] = None

        self.bag: List[str] = []
        self.current_piece: Optional[Piece] = None
        self.next_piece_kind = self._draw_from_bag()

        self.spawn_piece()

    # -------------------------
    # Piece generation
    # -------------------------
    def _refill_bag(self) -> None:
        self.bag = list(SHAPES.keys())
        self.rng.shuffle(self.bag)

    def _draw_from_bag(self) -> str:
        if not self.bag:
            self._refill_bag()
        return self.bag.pop()

    def spawn_piece(self) -> None:
        kind = self.next_piece_kind
        self.next_piece_kind = self._draw_from_bag()

        spawn_col = BOARD_COLS // 2 - 2

        for offset in [0, -1, 1, -2, 2, -3, 3]:
            candidate = Piece(
                kind=kind,
                rotation=0,
                row=0,
                col=spawn_col + offset,
            )
            if self.is_valid_position(candidate):
                self.current_piece = candidate
                self.target_rotation = None
                self.target_left_col14 = None
                self.target_anchor_col = None
                self.target_final_row = None
                self.new_piece_spawned = True
                self.action_queue.clear()
                self.last_action = "SPAWN"
                return

        self.current_piece = Piece(kind=kind, rotation=0, row=0, col=spawn_col)
        self.game_over = True

    # -------------------------
    # Collision / placement
    # -------------------------
    def is_valid_position(self, piece: Piece) -> bool:
        for row, col in piece.cells():
            if col < 0 or col >= BOARD_COLS:
                return False
            if row < 0 or row >= BOARD_ROWS:
                return False
            if self.board[row][col] != 0:
                return False
        return True

    @staticmethod
    def is_valid_position_on_board(
        board_occ: Sequence[Sequence[int]],
        piece: Piece,
    ) -> bool:
        for row, col in piece.cells():
            if col < 0 or col >= BOARD_COLS:
                return False
            if row < 0 or row >= BOARD_ROWS:
                return False
            if int(board_occ[row][col]) != 0:
                return False
        return True

    def hard_drop_row(
        self,
        kind: str,
        rotation: int,
        col: int,
    ) -> Optional[int]:
        row = 0
        piece = Piece(kind=kind, rotation=rotation, row=row, col=col)

        if not self.is_valid_position(piece):
            return None

        while True:
            next_piece = Piece(
                kind=kind,
                rotation=rotation,
                row=row + 1,
                col=col,
            )
            if self.is_valid_position(next_piece):
                row += 1
            else:
                return row

    @staticmethod
    def hard_drop_row_on_board(
        board_occ: Sequence[Sequence[int]],
        kind: str,
        rotation: int,
        col: int,
    ) -> Optional[int]:
        row = 0
        piece = Piece(kind=kind, rotation=rotation, row=row, col=col)
        if not TetrisGame.is_valid_position_on_board(board_occ, piece):
            return None

        while True:
            next_piece = Piece(
                kind=kind,
                rotation=rotation,
                row=row + 1,
                col=col,
            )
            if TetrisGame.is_valid_position_on_board(board_occ, next_piece):
                row += 1
            else:
                return row

    def lock_piece(self, piece: Piece) -> None:
        for row, col in piece.cells():
            if 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS:
                self.board[row][col] = SHAPE_ID[piece.kind]

        cleared = self.clear_lines()
        self.lines_cleared_total += cleared
        self.pieces_placed += 1

        score_table = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        self.score += score_table.get(cleared, 0)

        self.spawn_piece()

    def clear_lines(self) -> int:
        new_board = [
            row for row in self.board if any(cell == 0 for cell in row)
        ]
        cleared = BOARD_ROWS - len(new_board)

        while len(new_board) < BOARD_ROWS:
            new_board.insert(0, [0 for _ in range(BOARD_COLS)])

        self.board = new_board
        return cleared

    # -------------------------
    # NN input / legal mask
    # -------------------------
    def get_current_piece_kind_index(self) -> Optional[int]:
        if self.current_piece is None:
            return None
        return PIECE_TO_INDEX[self.current_piece.kind]

    def get_current_piece_rot4(self) -> Optional[int]:
        if self.current_piece is None:
            return None
        return self.current_piece.rotation % 4

    def get_current_piece_col14(self) -> Optional[int]:
        """
        注意：这里必须返回 left_col14，而不是 anchor_col。
        """
        if self.current_piece is None:
            return None
        return leftmost_col(self.current_piece)

    def build_legal_mask_4x14(self, kind: str) -> List[List[int]]:
        board_occ = occ_board(self.board)
        legal = [[0 for _ in range(BOARD_COLS)] for _ in range(4)]

        rotations = len(SHAPES[kind])
        checked = set()

        for rot in range(rotations):
            offsets = SHAPES[kind][rot]
            min_col = min(col for _, col in offsets)
            max_col = max(col for _, col in offsets)

            for anchor_col in range(-min_col, BOARD_COLS - max_col):
                key = (rot, anchor_col, tuple(sorted(offsets)))
                if key in checked:
                    continue
                checked.add(key)

                final_row = self.hard_drop_row_on_board(
                    board_occ=board_occ,
                    kind=kind,
                    rotation=rot,
                    col=anchor_col,
                )
                if final_row is None:
                    continue

                final_piece = Piece(
                    kind=kind,
                    rotation=rot,
                    row=final_row,
                    col=anchor_col,
                )
                left_col = leftmost_col(final_piece)
                if 0 <= left_col < BOARD_COLS:
                    legal[rot % 4][left_col] = 1

        return legal

    # -------------------------
    # NN decision
    # -------------------------
    def choose_best_move_by_nn(
        self,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        返回：
        (target_rot4, target_left_col14, target_anchor_col, target_final_row)
        """
        if self.current_piece is None:
            return None

        kind = self.current_piece.kind
        kind_id = self.get_current_piece_kind_index()
        cur_rot4 = self.get_current_piece_rot4()
        cur_col14 = self.get_current_piece_col14()

        if kind_id is None or cur_rot4 is None or cur_col14 is None:
            return None

        active_mask = build_active_mask(self.current_piece)
        legal_mask_4x14 = self.build_legal_mask_4x14(kind)
        self.last_legal_mask_4x14 = [row[:] for row in legal_mask_4x14]

        pred = self.policy.predict(
            board_before=self.board,
            active_mask=active_mask,
            kind=kind,
            kind_id=kind_id,
            cur_rot4=cur_rot4,
            cur_col14=cur_col14,
            legal_mask_4x14=legal_mask_4x14,
        )

        target_rot4 = int(pred["pred_rot4"])
        target_left_col14 = int(pred["pred_col14"])
        target_anchor_col = int(pred["pred_anchor_col"])

        target_final_row = self.hard_drop_row(
            kind=kind,
            rotation=target_rot4,
            col=target_anchor_col,
        )
        if target_final_row is None:
            return None

        final_piece = Piece(
            kind=kind,
            rotation=target_rot4,
            row=target_final_row,
            col=target_anchor_col,
        )
        if not self.is_valid_position(final_piece):
            return None

        if leftmost_col(final_piece) != target_left_col14:
            return None

        self.last_pred_raw_action = int(pred["raw_action"])
        self.last_pred_masked_action = int(pred["masked_action"])
        self.last_pred_rot4 = target_rot4
        self.last_pred_col14 = target_left_col14
        self.last_pred_anchor_col = target_anchor_col

        return (
            target_rot4,
            target_left_col14,
            target_anchor_col,
            target_final_row,
        )

    # -------------------------
    # 直接到目标落点，不再走过程
    # -------------------------
    def plan_actions_for_current_piece(self) -> bool:
        if self.current_piece is None:
            return False

        move = self.choose_best_move_by_nn()
        if move is None:
            return False

        (
            self.target_rotation,
            self.target_left_col14,
            self.target_anchor_col,
            self.target_final_row,
        ) = move

        final_piece = Piece(
            kind=self.current_piece.kind,
            rotation=self.target_rotation,
            row=self.target_final_row,
            col=self.target_anchor_col,
        )

        if not self.is_valid_position(final_piece):
            return False

        if leftmost_col(final_piece) != self.target_left_col14:
            return False

        self.current_piece = final_piece
        self.action_queue.clear()
        self.last_action = "NN_DIRECT"
        return True

    def execute_one_action(self) -> None:
        if self.current_piece is None:
            return
        self.last_action = "LOCK"
        self.lock_piece(self.current_piece)

    def auto_step(self) -> None:
        if self.game_over or self.current_piece is None:
            return

        ok = self.plan_actions_for_current_piece()
        if not ok:
            self.game_over = True
            return

        self.new_piece_spawned = False
        self.execute_one_action()


# =========================
# Renderer
# =========================
class TetrisRenderer:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        pygame.font.init()
        self.font_small = pygame.font.SysFont("microsoftyahei", 18)
        self.font_medium = pygame.font.SysFont(
            "microsoftyahei",
            23,
            bold=True,
        )
        self.font_title = pygame.font.SysFont(
            "microsoftyahei",
            24,
            bold=True,
        )

        self.board_x = LEFT_MARGIN
        self.board_y = TOP_MARGIN
        self.panel_x = self.board_x + BOARD_COLS * CELL_SIZE + 30
        self.panel_y = TOP_MARGIN
        self.panel_w = RIGHT_PANEL_WIDTH
        self.panel_h = BOARD_ROWS * CELL_SIZE

    def draw(self, game: TetrisGame, paused: bool) -> None:
        self.screen.fill(WINDOW_BG)
        self.draw_title(paused)
        self.draw_board(game)
        self.draw_side_panel(game)
        self.draw_footer_hint()
        pygame.display.flip()

    def draw_title(self, paused: bool) -> None:
        title = "Task 3 - Tetris NN Auto Player"
        subtitle = "task6 模型接入版：left_col14 对齐 + legal mask + anchor_col 解码"

        self.screen.blit(
            self.font_title.render(title, True, TEXT_COLOR),
            (LEFT_MARGIN, 10),
        )
        self.screen.blit(
            self.font_small.render(subtitle, True, (80, 88, 98)),
            (LEFT_MARGIN, 36),
        )

        if paused:
            pause_text = self.font_medium.render("PAUSED", True, TARGET_COLOR)
            self.screen.blit(pause_text, (WINDOW_WIDTH - 150, 16))

    def draw_board(self, game: TetrisGame) -> None:
        board_rect = pygame.Rect(
            self.board_x,
            self.board_y,
            BOARD_COLS * CELL_SIZE,
            BOARD_ROWS * CELL_SIZE,
        )
        pygame.draw.rect(self.screen, (255, 255, 255), board_rect, border_radius=6)
        pygame.draw.rect(
            self.screen,
            BORDER_COLOR,
            board_rect,
            width=2,
            border_radius=6,
        )

        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                x = self.board_x + col * CELL_SIZE
                y = self.board_y + row * CELL_SIZE
                cell_rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

                value = game.board[row][col]
                if value != 0:
                    kind = ID_TO_KIND.get(value, "O")
                    color = SHAPE_COLORS.get(kind, (130, 130, 130))
                    inner_rect = cell_rect.inflate(-4, -4)
                    pygame.draw.rect(
                        self.screen,
                        color,
                        inner_rect,
                        border_radius=4,
                    )

                pygame.draw.rect(self.screen, GRID_LINE_COLOR, cell_rect, width=1)

        self.draw_current_piece(game)

    def draw_current_piece(self, game: TetrisGame) -> None:
        if game.game_over or game.current_piece is None:
            return

        for row, col in game.current_piece.cells():
            if 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS:
                x = self.board_x + col * CELL_SIZE
                y = self.board_y + row * CELL_SIZE
                cell_rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                inner_rect = cell_rect.inflate(-4, -4)

                color = SHAPE_COLORS[game.current_piece.kind]
                pygame.draw.rect(
                    self.screen,
                    color,
                    inner_rect,
                    border_radius=4,
                )
                pygame.draw.rect(
                    self.screen,
                    (255, 255, 255),
                    inner_rect,
                    width=2,
                    border_radius=4,
                )

    def draw_side_panel(self, game: TetrisGame) -> None:
        panel_rect = pygame.Rect(
            self.panel_x,
            self.panel_y,
            self.panel_w,
            self.panel_h,
        )
        pygame.draw.rect(self.screen, PANEL_BG, panel_rect, border_radius=8)
        pygame.draw.rect(
            self.screen,
            BORDER_COLOR,
            panel_rect,
            width=2,
            border_radius=8,
        )

        y = self.panel_y + 12
        self.screen.blit(
            self.font_medium.render("NN 控制状态", True, TEXT_COLOR),
            (self.panel_x + 14, y),
        )
        y += 38

        cur_piece = game.current_piece
        cur_kind = cur_piece.kind if cur_piece is not None else "-"
        cur_rot4 = game.get_current_piece_rot4()
        cur_col14 = game.get_current_piece_col14()

        stats = [
            f"Score: {game.score}",
            f"Lines: {game.lines_cleared_total}",
            f"Pieces: {game.pieces_placed}",
            f"Board: {BOARD_ROWS}x{BOARD_COLS}",
            f"Mode: NN direct lock",
            f"Piece RNG: {game.rng_mode}",
            f"Seed: {game.seed_value}",
            "",
            f"Current kind: {cur_kind}",
            f"Current rot4: {cur_rot4}",
            f"Current left_col14: {cur_col14}",
            "",
            f"Pred raw action: {game.last_pred_raw_action}",
            f"Pred safe action: {game.last_pred_masked_action}",
            f"Pred rot4: {game.last_pred_rot4}",
            f"Pred left_col14: {game.last_pred_col14}",
            f"Pred anchor_col: {game.last_pred_anchor_col}",
            f"Target final row: {game.target_final_row}",
            "",
            f"Last action: {game.last_action}",
            f"Queue left: {len(game.action_queue)}",
        ]

        for line in stats:
            self.screen.blit(
                self.font_small.render(line, True, TEXT_COLOR),
                (self.panel_x + 14, y),
            )
            y += 21

        y += 6
        self.screen.blit(
            self.font_medium.render("Next", True, TEXT_COLOR),
            (self.panel_x + 14, y),
        )
        y += 38

        preview_rect = pygame.Rect(self.panel_x + 14, y, 150, 110)
        pygame.draw.rect(self.screen, (255, 255, 255), preview_rect, border_radius=6)
        pygame.draw.rect(
            self.screen,
            BORDER_COLOR,
            preview_rect,
            width=2,
            border_radius=6,
        )
        self.draw_next_piece(game.next_piece_kind, preview_rect)

        y += 126
        if game.game_over:
            self.screen.blit(
                self.font_medium.render("GAME OVER", True, TARGET_COLOR),
                (self.panel_x + 14, y),
            )
            self.screen.blit(
                self.font_small.render("R 重开 / ESC 退出", True, (90, 90, 90)),
                (self.panel_x + 14, y + 30),
            )

    def draw_next_piece(self, kind: str, preview_rect: pygame.Rect) -> None:
        offsets = SHAPES[kind][0]
        color = SHAPE_COLORS[kind]

        min_row = min(row for row, _ in offsets)
        max_row = max(row for row, _ in offsets)
        min_col = min(col for _, col in offsets)
        max_col = max(col for _, col in offsets)

        shape_h = (max_row - min_row + 1) * CELL_SIZE
        shape_w = (max_col - min_col + 1) * CELL_SIZE

        start_x = preview_rect.x + (preview_rect.width - shape_w) // 2
        start_y = preview_rect.y + (preview_rect.height - shape_h) // 2

        for row, col in offsets:
            x = start_x + (col - min_col) * CELL_SIZE
            y = start_y + (row - min_row) * CELL_SIZE
            cell_rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            inner_rect = cell_rect.inflate(-4, -4)

            pygame.draw.rect(self.screen, color, inner_rect, border_radius=4)
            pygame.draw.rect(self.screen, GRID_LINE_COLOR, cell_rect, width=1)

    def draw_footer_hint(self) -> None:
        y = TOP_MARGIN + BOARD_ROWS * CELL_SIZE + 16
        hints = [
            "R: 重开",
            "P: 暂停/继续",
            "ESC: 退出",
            "Task3 NN = 直接推理最终落点并锁定",
        ]
        text = "    |    ".join(hints)
        self.screen.blit(
            self.font_small.render(text, True, (75, 82, 90)),
            (LEFT_MARGIN, y),
        )


# =========================
# RNG helpers
# =========================
def make_game_rng(
    fixed_seed: bool,
    seed: Optional[int],
) -> Tuple[random.Random, str, int]:
    if fixed_seed:
        seed_value = 42 if seed is None else seed
        return random.Random(seed_value), "FIXED", seed_value

    seed_value = random.SystemRandom().randrange(1, 2**31 - 1)
    return random.Random(seed_value), "RANDOM", seed_value


# =========================
# Main
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=r"C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfimit_v6_mainline_v2\model_best_action.pt",
    )
    parser.add_argument(
        "--auto_step_ms",
        type=int,
        default=AUTO_STEP_MS,
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
    )
    parser.add_argument(
        "--policy_topk",
        type=int,
        default=8,
        help="从神经网络合法动作前 K 名中选择，K=1 等价于原始 argmax。",
    )
    parser.add_argument(
        "--safety_weight",
        type=float,
        default=0.70,
        help="结构安全重排权重；0 表示只按神经网络分数。",
    )
    parser.add_argument(
        "--fixed_seed",
        action="store_true",
        help="启用固定 seed 调试模式；默认关闭，默认随机出块。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="固定 seed 数值，仅在 --fixed_seed 模式下生效。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    game_rng, rng_mode, seed_value = make_game_rng(
        fixed_seed=args.fixed_seed,
        seed=args.seed,
    )
    fixed_mode = bool(args.fixed_seed)

    policy = NNPolicy(
        model_path=args.model,
        cpu=args.cpu,
        policy_topk=args.policy_topk,
        safety_weight=args.safety_weight,
    )

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Task 3 - Tetris NN Auto Player")
    clock = pygame.time.Clock()

    game = TetrisGame(
        policy=policy,
        rng=game_rng,
        rng_mode=rng_mode,
        seed_value=seed_value,
    )
    renderer = TetrisRenderer(screen)

    paused = False
    auto_timer = 0

    print(f"[INFO] model loaded from: {args.model}")
    print(f"[INFO] device: {policy.device}")
    print(f"[INFO] piece rng mode: {rng_mode}")
    print(f"[INFO] seed value: {seed_value}")
    print("[INFO] This version is aligned with task6:")
    print("       - current col = left_col14")
    print("       - target col = left_col14")
    print("       - legal mask applied before argmax")
    print("       - safe rerank is applied on top legal NN candidates")
    print("       - control layer uses anchor_col converted from left_col14")
    print("       - direct mode: piece jumps to final target and locks immediately")
    print("       - default mode is RANDOM piece generation")
    print("       - use --fixed_seed --seed 42 for reproducible debug")

    while True:
        dt = clock.tick(FPS)
        auto_timer += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                if event.key == pygame.K_r:
                    if fixed_mode:
                        game_rng, rng_mode, seed_value = make_game_rng(
                            fixed_seed=True,
                            seed=args.seed,
                        )
                    else:
                        game_rng, rng_mode, seed_value = make_game_rng(
                            fixed_seed=False,
                            seed=None,
                        )

                    game = TetrisGame(
                        policy=policy,
                        rng=game_rng,
                        rng_mode=rng_mode,
                        seed_value=seed_value,
                    )
                    paused = False
                    auto_timer = 0

                    print(f"[RESET] piece rng mode: {rng_mode}")
                    print(f"[RESET] seed value: {seed_value}")

                if event.key == pygame.K_p:
                    paused = not paused

        if not paused and not game.game_over and auto_timer >= args.auto_step_ms:
            auto_timer = 0
            game.auto_step()

        renderer.draw(game, paused)


if __name__ == "__main__":
    main()
