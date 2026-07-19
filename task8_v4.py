"""Task8 legacy / experimental program.

用途：
- 早期 Task8 实验版本，包含搜索教师、样本收集、训练和评估相关逻辑。
- 用于保留历史方案和对比思路。

当前主线不推荐从这里继续改；正式训练请使用 task8_selfimit_train_v6.py。
"""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_f


ROWS = 18
COLS = 14
ACTION_CLASSES = 56
STATE_VEC_DIM = 7 + 4 + 14

KIND_LIST = ["I", "O", "T", "S", "Z", "J", "L"]
KIND_TO_ID = {kind: idx for idx, kind in enumerate(KIND_LIST)}
ID_TO_KIND = {idx: kind for kind, idx in KIND_TO_ID.items()}

NEG_INF = -1e9


def normalize_cells(cells: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    out = [(r - min_r, c - min_c) for r, c in cells]
    out.sort()
    return out


def rotate_cells_cw(cells: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    return normalize_cells([(c, -r) for r, c in cells])


def build_rotations(base_cells: Sequence[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    cur = normalize_cells(base_cells)
    rots: List[List[Tuple[int, int]]] = []
    for _ in range(4):
        rots.append(cur)
        cur = rotate_cells_cw(cur)
    return rots


PIECE_ROTATIONS: Dict[str, List[List[Tuple[int, int]]]] = {
    "I": build_rotations([(0, 0), (0, 1), (0, 2)]),
    "O": build_rotations([(0, 0), (0, 1), (1, 0), (1, 1)]),
    "T": build_rotations([(0, 1), (1, 0), (1, 1), (1, 2)]),
    "S": build_rotations([(0, 1), (0, 2), (1, 0), (1, 1)]),
    "Z": build_rotations([(0, 0), (0, 1), (1, 1), (1, 2)]),
    "J": build_rotations([(0, 0), (1, 0), (1, 1), (1, 2)]),
    "L": build_rotations([(0, 2), (1, 0), (1, 1), (1, 2)]),
}


def piece_cells(kind: str, rot: int) -> List[Tuple[int, int]]:
    return PIECE_ROTATIONS[kind][rot % 4]


def piece_hw(kind: str, rot: int) -> Tuple[int, int]:
    cells = piece_cells(kind, rot)
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    return h, w


def action_to_rot_col(action_idx: int) -> Tuple[int, int]:
    return int(action_idx // COLS), int(action_idx % COLS)


def rot_col_to_action(rot: int, col: int) -> int:
    return rot * COLS + col


def default_spawn_col(kind: str, rot: int = 0) -> int:
    _, w = piece_hw(kind, rot)
    return max(0, min(COLS - w, (COLS - w) // 2))


def one_hot(index: int, size: int) -> List[float]:
    out = [0.0] * size
    if 0 <= index < size:
        out[index] = 1.0
    return out


def build_state_vector(kind_id: int, cur_rot4: int, cur_col14: int) -> List[float]:
    return one_hot(kind_id, 7) + one_hot(cur_rot4, 4) + one_hot(cur_col14, COLS)


def compute_column_heights(board: np.ndarray) -> np.ndarray:
    heights = np.zeros((COLS,), dtype=np.int32)
    for c in range(COLS):
        filled = np.where(board[:, c] > 0)[0]
        heights[c] = 0 if filled.size == 0 else ROWS - int(filled[0])
    return heights


def aggregate_height(board: np.ndarray) -> int:
    return int(compute_column_heights(board).sum())


def max_height(board: np.ndarray) -> int:
    return int(compute_column_heights(board).max())


def count_holes(board: np.ndarray) -> int:
    holes = 0
    for c in range(COLS):
        seen = False
        for r in range(ROWS):
            if board[r, c]:
                seen = True
            elif seen:
                holes += 1
    return holes


def bumpiness(board: np.ndarray) -> int:
    heights = compute_column_heights(board)
    return int(np.abs(heights[1:] - heights[:-1]).sum())


def count_near_clear_rows(board: np.ndarray) -> int:
    cnt = 0
    for r in range(ROWS):
        filled = int(board[r].sum())
        if filled == COLS - 1:
            cnt += 1
    return cnt


def row_transitions(board: np.ndarray) -> int:
    total = 0
    for r in range(ROWS):
        prev = 1
        for c in range(COLS):
            cur = int(board[r, c])
            if cur != prev:
                total += 1
            prev = cur
        if prev == 0:
            total += 1
    return total


def col_transitions(board: np.ndarray) -> int:
    total = 0
    for c in range(COLS):
        prev = 1
        for r in range(ROWS):
            cur = int(board[r, c])
            if cur != prev:
                total += 1
            prev = cur
        if prev == 0:
            total += 1
    return total


def board_features(board: np.ndarray) -> Dict[str, float]:
    return {
        "agg_h": float(aggregate_height(board)),
        "max_h": float(max_height(board)),
        "holes": float(count_holes(board)),
        "bump": float(bumpiness(board)),
        "near": float(count_near_clear_rows(board)),
        "row_t": float(row_transitions(board)),
        "col_t": float(col_transitions(board)),
    }


def board_quality(board: np.ndarray) -> float:
    feat = board_features(board)

    quality = 0.0
    quality += 2.0 * feat["near"]
    quality -= 4.8 * feat["holes"]
    quality -= 0.11 * feat["agg_h"]
    quality -= 0.35 * feat["max_h"]
    quality -= 0.18 * feat["bump"]
    quality -= 0.12 * feat["row_t"]
    quality -= 0.10 * feat["col_t"]

    if feat["max_h"] >= 12:
        quality -= 3.0
    if feat["max_h"] >= 14:
        quality -= 6.0

    return float(quality)


def line_reward(lines: int) -> float:
    table = [0.0, 1.0, 3.0, 5.5, 8.5]
    return table[min(max(lines, 0), 4)]


def valid_at(board: np.ndarray, kind: str, rot: int, row: int, col: int) -> bool:
    for dr, dc in piece_cells(kind, rot):
        rr = row + dr
        cc = col + dc
        if cc < 0 or cc >= COLS or rr >= ROWS:
            return False
        if rr >= 0 and board[rr, cc]:
            return False
    return True


def compute_reachable_resting(
    board: np.ndarray,
    kind: str,
    start_rot: int = 0,
    start_col: Optional[int] = None,
) -> Dict[Tuple[int, int], int]:
    if start_col is None:
        start_col = default_spawn_col(kind, start_rot)

    start_h, _ = piece_hw(kind, start_rot)
    start_row = -start_h

    if not valid_at(board, kind, start_rot, start_row, start_col):
        return {}

    q = deque()
    q.append((start_rot, start_row, start_col))
    visited = {(start_rot, start_row, start_col)}
    resting: Dict[Tuple[int, int], int] = {}

    while q:
        rot, row, col = q.popleft()

        if not valid_at(board, kind, rot, row + 1, col):
            key = (rot, col)
            best_row = resting.get(key, -10**9)
            if row > best_row:
                resting[key] = row

        next_states = [
            (rot, row, col - 1),
            (rot, row, col + 1),
            ((rot + 1) % 4, row, col),
            (rot, row + 1, col),
        ]

        for nrot, nrow, ncol in next_states:
            state = (nrot, nrow, ncol)
            if state in visited:
                continue
            if not valid_at(board, kind, nrot, nrow, ncol):
                continue
            visited.add(state)
            q.append(state)

    return resting


def get_legal_mask(board: np.ndarray, kind: str) -> np.ndarray:
    mask = np.zeros((ACTION_CLASSES,), dtype=np.float32)
    for (rot, col), _row in compute_reachable_resting(board, kind).items():
        if 0 <= col < COLS:
            mask[rot_col_to_action(rot, col)] = 1.0
    return mask


def apply_action_to_board(
    board: np.ndarray,
    kind: str,
    action_idx: int,
) -> Tuple[np.ndarray, int, bool, bool]:
    resting = compute_reachable_resting(board, kind)
    rot, col = action_to_rot_col(action_idx)

    if (rot, col) not in resting:
        return board.copy(), 0, True, False

    row = resting[(rot, col)]
    out = board.copy()
    top_out = False

    for dr, dc in piece_cells(kind, rot):
        rr = row + dr
        cc = col + dc
        if rr < 0:
            top_out = True
        else:
            out[rr, cc] = 1

    if top_out:
        return out, 0, False, True

    full_rows = [r for r in range(ROWS) if int(out[r].sum()) == COLS]
    lines = len(full_rows)

    if lines > 0:
        keep_rows = [r for r in range(ROWS) if r not in full_rows]
        new_board = np.zeros_like(out)
        start = ROWS - len(keep_rows)
        new_board[start:] = out[keep_rows]
        out = new_board

    return out, lines, False, False


def can_spawn_any(board: np.ndarray, kind: str) -> bool:
    return bool(compute_reachable_resting(board, kind))


@dataclass
class SearchConfig:
    depth: int
    beam_width: int
    next_piece_samples: int
    gamma: float
    quality_coef: float
    dead_penalty: float


def transition_score(
    board_before: np.ndarray,
    board_after: np.ndarray,
    lines: int,
    quality_coef: float,
) -> float:
    q_before = board_quality(board_before)
    q_after = board_quality(board_after)
    score = line_reward(lines) + quality_coef * (q_after - q_before)
    return float(score)


def search_value(
    board: np.ndarray,
    kind: str,
    cfg: SearchConfig,
    rng: random.Random,
    memo: Dict[Tuple[bytes, str, int], float],
    depth: int,
) -> float:
    key = (board.tobytes(), kind, depth)
    if key in memo:
        return memo[key]

    legal_mask = get_legal_mask(board, kind)
    legal_actions = np.where(legal_mask > 0.5)[0].tolist()

    if not legal_actions:
        memo[key] = cfg.dead_penalty
        return cfg.dead_penalty

    scored: List[Tuple[float, int, np.ndarray, bool]] = []

    for action in legal_actions:
        board_after, lines, illegal, top_out = apply_action_to_board(board, kind, int(action))
        if illegal:
            continue
        if top_out:
            scored.append((cfg.dead_penalty, int(action), board_after, True))
            continue

        s = transition_score(board, board_after, lines, cfg.quality_coef)
        scored.append((s, int(action), board_after, False))

    if not scored:
        memo[key] = cfg.dead_penalty
        return cfg.dead_penalty

    scored.sort(key=lambda x: x[0], reverse=True)

    if depth <= 1:
        best = float(scored[0][0])
        memo[key] = best
        return best

    best_total = -1e18
    keep = scored[:max(1, cfg.beam_width)]

    for immediate_score, _action, board_after, is_dead in keep:
        if is_dead:
            total = cfg.dead_penalty
        else:
            future_vals: List[float] = []
            for _ in range(max(1, cfg.next_piece_samples)):
                next_kind = rng.choice(KIND_LIST)
                if not can_spawn_any(board_after, next_kind):
                    future_vals.append(cfg.dead_penalty)
                else:
                    v = search_value(
                        board=board_after,
                        kind=next_kind,
                        cfg=cfg,
                        rng=rng,
                        memo=memo,
                        depth=depth - 1,
                    )
                    future_vals.append(v)

            future_mean = float(np.mean(future_vals))
            total = immediate_score + cfg.gamma * future_mean

        if total > best_total:
            best_total = total

    memo[key] = float(best_total)
    return float(best_total)


def search_action_scores(
    board: np.ndarray,
    kind: str,
    cfg: SearchConfig,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.full((ACTION_CLASSES,), NEG_INF, dtype=np.float32)
    legal_mask = get_legal_mask(board, kind)

    legal_actions = np.where(legal_mask > 0.5)[0].tolist()
    if not legal_actions:
        return scores, legal_mask

    memo: Dict[Tuple[bytes, str, int], float] = {}

    for action in legal_actions:
        board_after, lines, illegal, top_out = apply_action_to_board(board, kind, int(action))
        if illegal:
            continue

        if top_out:
            total = cfg.dead_penalty
        else:
            immediate_score = transition_score(board, board_after, lines, cfg.quality_coef)

            if cfg.depth <= 1:
                total = immediate_score
            else:
                future_vals: List[float] = []
                for _ in range(max(1, cfg.next_piece_samples)):
                    next_kind = rng.choice(KIND_LIST)
                    if not can_spawn_any(board_after, next_kind):
                        future_vals.append(cfg.dead_penalty)
                    else:
                        v = search_value(
                            board=board_after,
                            kind=next_kind,
                            cfg=cfg,
                            rng=rng,
                            memo=memo,
                            depth=cfg.depth - 1,
                        )
                        future_vals.append(v)
                total = immediate_score + cfg.gamma * float(np.mean(future_vals))

        scores[int(action)] = float(total)

    return scores, legal_mask


class PlacementTetrisEnv:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_kind = "I"
        self.cur_rot4 = 0
        self.cur_col14 = default_spawn_col("I", 0)
        self.total_lines = 0
        self.total_pieces = 0
        self.done = False

    def get_active_mask(self) -> np.ndarray:
        mask = np.zeros((ROWS, COLS), dtype=np.float32)
        for dr, dc in piece_cells(self.current_kind, self.cur_rot4):
            rr = dr
            cc = self.cur_col14 + dc
            if 0 <= rr < ROWS and 0 <= cc < COLS:
                mask[rr, cc] = 1.0
        return mask

    def get_state_vec(self) -> np.ndarray:
        return np.array(
            build_state_vector(
                kind_id=KIND_TO_ID[self.current_kind],
                cur_rot4=self.cur_rot4,
                cur_col14=self.cur_col14,
            ),
            dtype=np.float32,
        )

    def get_obs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        board_ch = self.board.astype(np.float32)
        active_ch = self.get_active_mask()
        board_input = np.stack([board_ch, active_ch], axis=0)
        state_vec = self.get_state_vec()
        legal = get_legal_mask(self.board, self.current_kind).astype(np.float32)
        return board_input, state_vec, legal

    def spawn_piece(self, kind: Optional[str] = None) -> None:
        self.current_kind = kind if kind is not None else self.rng.choice(KIND_LIST)
        self.cur_rot4 = 0
        self.cur_col14 = default_spawn_col(self.current_kind, 0)
        if not can_spawn_any(self.board, self.current_kind):
            self.done = True

    def reset_empty(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.board.fill(0)
        self.total_lines = 0
        self.total_pieces = 0
        self.done = False
        self.spawn_piece()
        return self.get_obs()

    def build_curriculum_board(self, mode: str) -> np.ndarray:
        if mode == "empty":
            return np.zeros((ROWS, COLS), dtype=np.int8)

        board = np.zeros((ROWS, COLS), dtype=np.int8)

        if mode == "mid":
            lo, hi = 4, 8
            hole_prob = 0.05
            near_rows = self.rng.randint(1, 2)
        else:
            lo, hi = 8, 12
            hole_prob = 0.10
            near_rows = self.rng.randint(1, 3)

        heights = []
        cur = self.rng.randint(lo, hi)
        for _ in range(COLS):
            cur += self.rng.choice([-2, -1, 0, 1, 2])
            cur = max(lo, min(hi, cur))
            heights.append(cur)

        for c, h in enumerate(heights):
            if h <= 0:
                continue
            start_r = ROWS - h
            board[start_r:, c] = 1

        for c, h in enumerate(heights):
            if h <= 2:
                continue
            start_r = ROWS - h
            for r in range(start_r + 1, ROWS - 1):
                if self.rng.random() < hole_prob:
                    board[r, c] = 0

        candidate_rows = list(range(max(ROWS - hi - 2, 0), ROWS))
        self.rng.shuffle(candidate_rows)

        for r in candidate_rows[:near_rows]:
            board[r, :] = 1
            gaps = 1 if mode == "mid" else self.rng.choice([1, 1, 2])
            gap_cols = self.rng.sample(range(COLS), gaps)
            for gc in gap_cols:
                board[r, gc] = 0

        for r in range(ROWS):
            if int(board[r].sum()) == COLS:
                board[r, self.rng.randrange(COLS)] = 0

        return board.astype(np.int8)

    def reset_curriculum(self, mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if mode == "empty":
            return self.reset_empty()

        for _ in range(100):
            self.board = self.build_curriculum_board(mode)
            self.total_lines = 0
            self.total_pieces = 0
            self.done = False
            self.spawn_piece()
            if not self.done:
                return self.get_obs()

        return self.reset_empty()

    def reset_by_mix(self, p_empty: float, p_mid: float, p_hard: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.rng.random()
        if x < p_empty:
            return self.reset_curriculum("empty")
        if x < p_empty + p_mid:
            return self.reset_curriculum("mid")
        return self.reset_curriculum("hard")

    def step(self, action_idx: int) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], bool]:
        if self.done:
            return self.get_obs(), True

        board_after, lines, illegal, top_out = apply_action_to_board(
            self.board,
            self.current_kind,
            action_idx,
        )

        if illegal or top_out:
            self.done = True
            return self.get_obs(), True

        self.board = board_after
        self.total_lines += lines
        self.total_pieces += 1
        self.spawn_piece()
        return self.get_obs(), self.done


def masked_softmax_np(scores: np.ndarray, legal_mask: np.ndarray, temp: float) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float32)
    idx = np.where(legal_mask > 0.5)[0]
    if idx.size == 0:
        return out

    legal_scores = scores[idx].astype(np.float64) / max(temp, 1e-6)
    legal_scores = legal_scores - np.max(legal_scores)
    expv = np.exp(legal_scores)
    prob = expv / max(expv.sum(), 1e-12)
    out[idx] = prob.astype(np.float32)
    return out


def save_npz_dataset(
    out_path: str | Path,
    boards: List[np.ndarray],
    states: List[np.ndarray],
    legal_masks: List[np.ndarray],
    targets: List[int],
    score_maps: List[np.ndarray],
    prob_maps: List[np.ndarray],
) -> None:
    if len(targets) == 0:
        raise ValueError("no samples collected")

    boards_np = np.stack(boards).astype(np.float32)
    states_np = np.stack(states).astype(np.float32)
    legal_np = np.stack(legal_masks).astype(np.float32)
    targets_np = np.array(targets, dtype=np.int64)
    scores_np = np.stack(score_maps).astype(np.float32)
    probs_np = np.stack(prob_maps).astype(np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        boards=boards_np,
        states=states_np,
        legal_masks=legal_np,
        targets=targets_np,
        scores=scores_np,
        probs=probs_np,
    )

    print(f"[OK] saved dataset: {out_path}")
    print(f"[INFO] total_samples={len(targets_np)}")


def load_dataset_npz(path: str | Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    required = ["boards", "states", "legal_masks", "targets", "scores", "probs"]
    for key in required:
        if key not in data:
            raise ValueError(f"dataset missing key: {key}")
    return {key: data[key] for key in required}


def parse_data_paths(paths_str: str) -> List[str]:
    raw = str(paths_str).replace(";", ",").split(",")
    out = [s.strip() for s in raw if s.strip()]
    if not out:
        raise ValueError("empty --data")
    return out


def concat_datasets(paths: List[str]) -> Dict[str, np.ndarray]:
    chunks = [load_dataset_npz(p) for p in paths]
    if len(chunks) == 1:
        return chunks[0]

    merged: Dict[str, np.ndarray] = {}
    for key in ["boards", "states", "legal_masks", "targets", "scores", "probs"]:
        merged[key] = np.concatenate([x[key] for x in chunks], axis=0)

    print(f"[INFO] merged datasets={len(paths)} total_samples={len(merged['targets'])}")
    return merged


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

    def forward(self, board_tensor: torch.Tensor, state_tensor: torch.Tensor) -> torch.Tensor:
        board_feat = self.board_encoder(board_tensor)
        state_feat = self.state_encoder(state_tensor)
        fused = torch.cat([board_feat, state_feat], dim=1)
        logits = self.head(fused)
        return logits


def mask_logits_torch(logits: torch.Tensor, legal_masks: torch.Tensor) -> torch.Tensor:
    legal_bool = legal_masks > 0.5
    return torch.where(legal_bool, logits, torch.full_like(logits, -1e9))


def extract_state_dict(ckpt: object) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("checkpoint format not supported")


def safe_partial_load(
    model: nn.Module,
    ckpt_path: str,
    device: torch.device,
) -> Tuple[List[str], List[str]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = extract_state_dict(ckpt)
    model_state = model.state_dict()

    loaded_keys: List[str] = []
    skipped_keys: List[str] = []

    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            loaded_keys.append(k)
        else:
            skipped_keys.append(k)

    model.load_state_dict(model_state, strict=False)
    return loaded_keys, skipped_keys


def load_model(model_path: str, device: torch.device) -> ActionPolicyNet:
    model = ActionPolicyNet().to(device)

    ckpt = torch.load(model_path, map_location=device)
    state = extract_state_dict(ckpt)

    model_state = model.state_dict()
    all_match = True
    for k, v in state.items():
        if k not in model_state or model_state[k].shape != v.shape:
            all_match = False
            break

    if all_match:
        model.load_state_dict(state, strict=False)
        print("[LOAD] full load ok")
    else:
        loaded_keys, skipped_keys = safe_partial_load(model, model_path, device)
        print("[LOAD] partial load used")
        print(f"[LOAD] loaded_tensors={len(loaded_keys)}")
        print(f"[LOAD] first_loaded_keys={loaded_keys[:12]}")
        print(f"[LOAD] skipped_tensors={len(skipped_keys)}")
        print(f"[LOAD] first_skipped_keys={skipped_keys[:12]}")

    model.eval()
    return model


@torch.no_grad()
def choose_student_action(
    model: ActionPolicyNet,
    device: torch.device,
    board_input: np.ndarray,
    state_vec: np.ndarray,
    legal_mask: np.ndarray,
    rng: random.Random,
    epsilon: float,
) -> int:
    legal_idx = np.where(legal_mask > 0.5)[0]
    if legal_idx.size == 0:
        return 0

    if epsilon > 0.0 and rng.random() < epsilon:
        return int(rng.choice(legal_idx.tolist()))

    boards_t = torch.tensor(board_input[None], dtype=torch.float32, device=device)
    states_t = torch.tensor(state_vec[None], dtype=torch.float32, device=device)
    legal_t = torch.tensor(legal_mask[None], dtype=torch.float32, device=device)

    logits = model(boards_t, states_t)
    logits = mask_logits_torch(logits, legal_t)
    action = int(torch.argmax(logits, dim=1).item())
    return action


def make_sample_key(board: np.ndarray, kind: str, rot: int, col: int, target: int) -> Tuple[bytes, str, int, int, int]:
    return (board.tobytes(), kind, rot, col, target)


def maybe_add_sample(
    storage: Dict[str, List],
    seen_keys: set,
    board: np.ndarray,
    kind: str,
    rot: int,
    col: int,
    board_input: np.ndarray,
    state_vec: np.ndarray,
    legal_mask: np.ndarray,
    teacher_best: int,
    teacher_scores: np.ndarray,
    teacher_probs: np.ndarray,
) -> bool:
    key = make_sample_key(board, kind, rot, col, teacher_best)
    if key in seen_keys:
        return False
    seen_keys.add(key)
    storage["boards"].append(board_input.astype(np.float32))
    storage["states"].append(state_vec.astype(np.float32))
    storage["legal_masks"].append(legal_mask.astype(np.float32))
    storage["targets"].append(int(teacher_best))
    storage["scores"].append(teacher_scores.astype(np.float32))
    storage["probs"].append(teacher_probs.astype(np.float32))
    return True


def collect_dataset(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    env = PlacementTetrisEnv(seed=args.seed)

    cfg = SearchConfig(
        depth=args.search_depth,
        beam_width=args.search_beam_width,
        next_piece_samples=args.search_next_piece_samples,
        gamma=args.search_gamma,
        quality_coef=args.search_quality_coef,
        dead_penalty=args.search_dead_penalty,
    )

    storage = {
        "boards": [],
        "states": [],
        "legal_masks": [],
        "targets": [],
        "scores": [],
        "probs": [],
    }

    total_samples = 0
    t0 = time.time()

    for ep in range(1, args.collect_episodes + 1):
        env.reset_by_mix(
            p_empty=args.curriculum_empty_prob,
            p_mid=args.curriculum_mid_prob,
            p_hard=args.curriculum_hard_prob,
        )

        step_count = 0

        while not env.done and step_count < args.max_steps_per_episode:
            board_input, state_vec, _legal = env.get_obs()

            scores, legal_mask = search_action_scores(
                board=env.board,
                kind=env.current_kind,
                cfg=cfg,
                rng=rng,
            )

            legal_idx = np.where(legal_mask > 0.5)[0]
            if legal_idx.size == 0:
                break

            best_action = int(legal_idx[np.argmax(scores[legal_idx])])
            probs = masked_softmax_np(scores, legal_mask, args.teacher_temp)

            storage["boards"].append(board_input.astype(np.float32))
            storage["states"].append(state_vec.astype(np.float32))
            storage["legal_masks"].append(legal_mask.astype(np.float32))
            storage["targets"].append(best_action)
            storage["scores"].append(scores.astype(np.float32))
            storage["probs"].append(probs.astype(np.float32))

            total_samples += 1
            step_count += 1

            _, done = env.step(best_action)
            if done:
                break

        if ep % max(1, args.log_every_episodes) == 0:
            elapsed = time.time() - t0
            rate = total_samples / max(elapsed, 1e-6)
            print(
                f"[COLLECT] ep={ep:04d}/{args.collect_episodes} "
                f"samples={total_samples} rate={rate:.2f} samples/s"
            )

    save_npz_dataset(
        args.out_data,
        storage["boards"],
        storage["states"],
        storage["legal_masks"],
        storage["targets"],
        storage["scores"],
        storage["probs"],
    )


def collect_relabel_dataset(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    rng = random.Random(args.seed)
    env = PlacementTetrisEnv(seed=args.seed)
    student = load_model(args.student_model, device)

    cfg = SearchConfig(
        depth=args.search_depth,
        beam_width=args.search_beam_width,
        next_piece_samples=args.search_next_piece_samples,
        gamma=args.search_gamma,
        quality_coef=args.search_quality_coef,
        dead_penalty=args.search_dead_penalty,
    )

    storage = {
        "boards": [],
        "states": [],
        "legal_masks": [],
        "targets": [],
        "scores": [],
        "probs": [],
    }
    seen_keys: set = set()

    stats = {
        "online_added": 0,
        "tail_added": 0,
        "danger_hits": 0,
        "disagree_hits": 0,
        "near_hits": 0,
        "fail_episodes": 0,
    }

    t0 = time.time()

    for ep in range(1, args.collect_episodes + 1):
        env.reset_by_mix(
            p_empty=args.curriculum_empty_prob,
            p_mid=args.curriculum_mid_prob,
            p_hard=args.curriculum_hard_prob,
        )

        recent: Deque[Dict[str, object]] = deque(maxlen=max(1, args.fail_tail_keep))
        step_count = 0
        ep_added = 0

        while not env.done and step_count < args.max_steps_per_episode:
            board_input, state_vec, legal_mask = env.get_obs()
            legal_idx = np.where(legal_mask > 0.5)[0]
            if legal_idx.size == 0:
                break

            student_action = choose_student_action(
                model=student,
                device=device,
                board_input=board_input,
                state_vec=state_vec,
                legal_mask=legal_mask,
                rng=rng,
                epsilon=args.student_epsilon,
            )

            teacher_scores, teacher_legal = search_action_scores(
                board=env.board,
                kind=env.current_kind,
                cfg=cfg,
                rng=rng,
            )
            teacher_idx = np.where(teacher_legal > 0.5)[0]
            if teacher_idx.size == 0:
                break

            teacher_best = int(teacher_idx[np.argmax(teacher_scores[teacher_idx])])
            teacher_probs = masked_softmax_np(teacher_scores, teacher_legal, args.teacher_temp)

            feat = board_features(env.board)
            student_score = float(teacher_scores[student_action]) if legal_mask[student_action] > 0.5 else NEG_INF
            teacher_best_score = float(teacher_scores[teacher_best])
            score_gap = teacher_best_score - student_score

            is_danger = (
                feat["max_h"] >= args.fail_max_h
                or feat["holes"] >= args.fail_holes
                or feat["bump"] >= args.fail_bump
                or feat["agg_h"] >= args.fail_agg_h
            )
            is_near = (feat["near"] >= args.fail_near_rows) and (student_action != teacher_best)
            is_disagree = (student_action != teacher_best) and (score_gap >= args.fail_score_gap)

            if is_danger:
                stats["danger_hits"] += 1
            if is_near:
                stats["near_hits"] += 1
            if is_disagree:
                stats["disagree_hits"] += 1

            sample = {
                "board": env.board.copy(),
                "kind": env.current_kind,
                "rot": env.cur_rot4,
                "col": env.cur_col14,
                "board_input": board_input.copy(),
                "state_vec": state_vec.copy(),
                "legal_mask": teacher_legal.copy(),
                "teacher_best": teacher_best,
                "teacher_scores": teacher_scores.copy(),
                "teacher_probs": teacher_probs.copy(),
                "flagged": bool(is_danger or is_near or is_disagree),
            }
            recent.append(sample)

            should_add_now = False
            if is_disagree:
                should_add_now = True
            if is_near:
                should_add_now = True
            if is_danger and score_gap >= 0.0:
                should_add_now = True

            if should_add_now and rng.random() <= args.collect_online_prob:
                ok = maybe_add_sample(
                    storage=storage,
                    seen_keys=seen_keys,
                    board=sample["board"],
                    kind=sample["kind"],
                    rot=sample["rot"],
                    col=sample["col"],
                    board_input=sample["board_input"],
                    state_vec=sample["state_vec"],
                    legal_mask=sample["legal_mask"],
                    teacher_best=sample["teacher_best"],
                    teacher_scores=sample["teacher_scores"],
                    teacher_probs=sample["teacher_probs"],
                )
                if ok:
                    stats["online_added"] += 1
                    ep_added += 1

            _, done = env.step(student_action)
            step_count += 1

            if done:
                stats["fail_episodes"] += 1
                tail_items = list(recent)[-max(1, args.fail_tail_keep):]
                for sample_tail in tail_items:
                    if args.tail_only_flagged and not bool(sample_tail["flagged"]):
                        continue
                    ok = maybe_add_sample(
                        storage=storage,
                        seen_keys=seen_keys,
                        board=sample_tail["board"],
                        kind=sample_tail["kind"],
                        rot=sample_tail["rot"],
                        col=sample_tail["col"],
                        board_input=sample_tail["board_input"],
                        state_vec=sample_tail["state_vec"],
                        legal_mask=sample_tail["legal_mask"],
                        teacher_best=sample_tail["teacher_best"],
                        teacher_scores=sample_tail["teacher_scores"],
                        teacher_probs=sample_tail["teacher_probs"],
                    )
                    if ok:
                        stats["tail_added"] += 1
                        ep_added += 1
                break

        if ep % max(1, args.log_every_episodes) == 0:
            elapsed = time.time() - t0
            total_samples = len(storage["targets"])
            rate = total_samples / max(elapsed, 1e-6)
            print(
                f"[RELABEL] ep={ep:04d}/{args.collect_episodes} "
                f"samples={total_samples} ep_added={ep_added} "
                f"fail_eps={stats['fail_episodes']} rate={rate:.2f} samples/s"
            )

    print(
        f"[RELABEL STATS] online_added={stats['online_added']} "
        f"tail_added={stats['tail_added']} "
        f"danger_hits={stats['danger_hits']} "
        f"near_hits={stats['near_hits']} "
        f"disagree_hits={stats['disagree_hits']} "
        f"fail_episodes={stats['fail_episodes']}"
    )

    save_npz_dataset(
        args.out_data,
        storage["boards"],
        storage["states"],
        storage["legal_masks"],
        storage["targets"],
        storage["scores"],
        storage["probs"],
    )


def split_indices(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * 0.80)
    n_val = int(n * 0.10)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def make_batches(indices: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> List[np.ndarray]:
    idx = indices.copy()
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    batches = []
    for start in range(0, len(idx), batch_size):
        batches.append(idx[start:start + batch_size])
    return batches


@torch.no_grad()
def evaluate_split(
    model: ActionPolicyNet,
    arrays: Dict[str, np.ndarray],
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    soft_coef: float,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_ce = 0.0
    total_soft = 0.0
    total_score = 0.0
    total = 0
    correct = 0
    legal_ok = 0
    top3_ok = 0

    batches = make_batches(indices, batch_size, shuffle=False, seed=0)

    for batch_idx in batches:
        boards = torch.tensor(arrays["boards"][batch_idx], dtype=torch.float32, device=device)
        states = torch.tensor(arrays["states"][batch_idx], dtype=torch.float32, device=device)
        legal_masks = torch.tensor(arrays["legal_masks"][batch_idx], dtype=torch.float32, device=device)
        targets = torch.tensor(arrays["targets"][batch_idx], dtype=torch.long, device=device)
        teacher_probs = torch.tensor(arrays["probs"][batch_idx], dtype=torch.float32, device=device)
        teacher_scores = torch.tensor(arrays["scores"][batch_idx], dtype=torch.float32, device=device)

        logits = model(boards, states)
        logits_masked = mask_logits_torch(logits, legal_masks)

        ce = torch_f.cross_entropy(logits_masked, targets)
        log_probs = torch_f.log_softmax(logits_masked, dim=1)
        soft = -(teacher_probs * log_probs).sum(dim=1).mean()

        pred_score = (torch.softmax(logits_masked, dim=1) * teacher_scores).sum(dim=1).mean()
        loss = ce + soft_coef * soft

        pred = torch.argmax(logits_masked, dim=1)
        correct += int((pred == targets).sum().item())

        gathered = legal_masks.gather(1, pred.unsqueeze(1)).squeeze(1)
        legal_ok += int((gathered > 0.5).sum().item())

        top3 = torch.topk(logits_masked, k=3, dim=1).indices
        target_in_top3 = (top3 == targets.unsqueeze(1)).any(dim=1)
        top3_ok += int(target_in_top3.sum().item())

        bs = boards.shape[0]
        total += bs
        total_loss += float(loss.item()) * bs
        total_ce += float(ce.item()) * bs
        total_soft += float(soft.item()) * bs
        total_score += float(pred_score.item()) * bs

    return {
        "loss": total_loss / max(total, 1),
        "ce": total_ce / max(total, 1),
        "soft": total_soft / max(total, 1),
        "target_acc": correct / max(total, 1),
        "legal": legal_ok / max(total, 1),
        "top3": top3_ok / max(total, 1),
        "teacher_score_expectation": total_score / max(total, 1),
    }


@torch.no_grad()
def quick_eval_model(
    model: ActionPolicyNet,
    device: torch.device,
    episodes: int,
    max_pieces: int,
    seed: int,
) -> Dict[str, float]:
    env = PlacementTetrisEnv(seed=seed)
    model.eval()

    lines_list: List[int] = []
    pieces_list: List[int] = []
    illegal_total = 0

    for _ in range(episodes):
        env.reset_empty()

        while not env.done and env.total_pieces < max_pieces:
            boards, states, legal_masks = env.get_obs()

            boards_t = torch.tensor(boards[None], dtype=torch.float32, device=device)
            states_t = torch.tensor(states[None], dtype=torch.float32, device=device)
            legal_t = torch.tensor(legal_masks[None], dtype=torch.float32, device=device)

            logits = model(boards_t, states_t)
            logits = mask_logits_torch(logits, legal_t)
            action = int(torch.argmax(logits, dim=1).item())

            if legal_masks[action] < 0.5:
                illegal_total += 1
                legal_idx = np.where(legal_masks > 0.5)[0]
                if legal_idx.size == 0:
                    break
                action = int(legal_idx[0])

            _, done = env.step(action)
            if done:
                break

        lines_list.append(env.total_lines)
        pieces_list.append(env.total_pieces)

    arr_lines = np.array(lines_list, dtype=np.float32)
    arr_pieces = np.array(pieces_list, dtype=np.float32)

    return {
        "mean_survival": float(arr_pieces.mean()),
        "mean_lines": float(arr_lines.mean()),
        "max_lines": float(arr_lines.max()),
        "s10": float((arr_lines >= 10).mean()),
        "s20": float((arr_lines >= 20).mean()),
        "s30": float((arr_lines >= 30).mean()),
        "illegal": float(illegal_total / max(1, episodes)),
    }


def train_model(args: argparse.Namespace) -> None:
    arrays = concat_datasets(parse_data_paths(args.data))

    n = len(arrays["targets"])
    train_idx, val_idx, test_idx = split_indices(n, args.seed)

    print(
        f"[INFO] samples total={n} "
        f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
    )

    device = torch.device("cpu")
    print(f"[INFO] device={device}")

    model = ActionPolicyNet().to(device)

    if args.init_model:
        loaded_keys, skipped_keys = safe_partial_load(model, args.init_model, device)
        print(f"[INIT] loaded_tensors={len(loaded_keys)}")
        print(f"[INIT] first_loaded_keys={loaded_keys[:12]}")
        print(f"[INIT] skipped_tensors={len(skipped_keys)}")
        print(f"[INIT] first_skipped_keys={skipped_keys[:12]}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_eval_score = -1e18

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_batches = make_batches(train_idx, args.batch_size, shuffle=True, seed=args.seed + epoch)

        total_loss = 0.0
        total_ce = 0.0
        total_soft = 0.0
        total = 0
        correct = 0
        legal_ok = 0

        for batch_idx in train_batches:
            boards = torch.tensor(arrays["boards"][batch_idx], dtype=torch.float32, device=device)
            states = torch.tensor(arrays["states"][batch_idx], dtype=torch.float32, device=device)
            legal_masks = torch.tensor(arrays["legal_masks"][batch_idx], dtype=torch.float32, device=device)
            targets = torch.tensor(arrays["targets"][batch_idx], dtype=torch.long, device=device)
            teacher_probs = torch.tensor(arrays["probs"][batch_idx], dtype=torch.float32, device=device)

            logits = model(boards, states)
            logits_masked = mask_logits_torch(logits, legal_masks)

            ce = torch_f.cross_entropy(logits_masked, targets)
            log_probs = torch_f.log_softmax(logits_masked, dim=1)
            soft = -(teacher_probs * log_probs).sum(dim=1).mean()
            loss = ce + args.soft_coef * soft

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

            pred = torch.argmax(logits_masked, dim=1)
            correct += int((pred == targets).sum().item())

            gathered = legal_masks.gather(1, pred.unsqueeze(1)).squeeze(1)
            legal_ok += int((gathered > 0.5).sum().item())

            bs = boards.shape[0]
            total += bs
            total_loss += float(loss.item()) * bs
            total_ce += float(ce.item()) * bs
            total_soft += float(soft.item()) * bs

        train_metrics = {
            "loss": total_loss / max(total, 1),
            "ce": total_ce / max(total, 1),
            "soft": total_soft / max(total, 1),
            "target_acc": correct / max(total, 1),
            "legal": legal_ok / max(total, 1),
        }

        val_metrics = evaluate_split(
            model=model,
            arrays=arrays,
            indices=val_idx,
            batch_size=args.batch_size,
            device=device,
            soft_coef=args.soft_coef,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_metrics['loss']:.4f} "
            f"ce={train_metrics['ce']:.4f} "
            f"soft={train_metrics['soft']:.4f} "
            f"target_acc={train_metrics['target_acc']:.4f} "
            f"legal={train_metrics['legal']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} "
            f"ce={val_metrics['ce']:.4f} "
            f"soft={val_metrics['soft']:.4f} "
            f"target_acc={val_metrics['target_acc']:.4f} "
            f"legal={val_metrics['legal']:.4f} "
            f"top3={val_metrics['top3']:.4f} "
            f"scoreexp={val_metrics['teacher_score_expectation']:.4f}"
        )

        eval_metrics = None
        eval_score = None
        if args.save_best_by == "eval":
            eval_metrics = quick_eval_model(
                model=model,
                device=device,
                episodes=args.best_eval_episodes,
                max_pieces=args.eval_max_pieces,
                seed=args.seed + epoch * 17,
            )
            eval_score = (
                100.0 * eval_metrics["s30"]
                + 30.0 * eval_metrics["s20"]
                + eval_metrics["mean_lines"]
            )
            print(
                f"[BEST-EVAL] epoch={epoch:03d} "
                f"mean_lines={eval_metrics['mean_lines']:.2f} "
                f"s20={eval_metrics['s20']:.2f} "
                f"s30={eval_metrics['s30']:.2f} "
                f"mean_survival={eval_metrics['mean_survival']:.2f} "
                f"score={eval_score:.2f}"
            )

        improved_val = (
            val_metrics["target_acc"] > best_val_acc
            or (
                abs(val_metrics["target_acc"] - best_val_acc) < 1e-12
                and val_metrics["loss"] < best_val_loss
            )
        )

        improved_eval = eval_score is not None and eval_score > best_eval_score

        improved = improved_eval if args.save_best_by == "eval" else improved_val

        if improved:
            if improved_val:
                best_val_acc = val_metrics["target_acc"]
                best_val_loss = val_metrics["loss"]
            if improved_eval:
                best_eval_score = float(eval_score)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "best_eval_metrics": eval_metrics,
                    "args": vars(args),
                },
                out_dir / "model_best_action.pt",
            )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "best_eval_metrics": eval_metrics,
                "args": vars(args),
            },
            out_dir / "model_last_action.pt",
        )

    test_metrics = evaluate_split(
        model=model,
        arrays=arrays,
        indices=test_idx,
        batch_size=args.batch_size,
        device=device,
        soft_coef=args.soft_coef,
    )
    print(
        f"[TEST] loss={test_metrics['loss']:.4f} "
        f"ce={test_metrics['ce']:.4f} "
        f"soft={test_metrics['soft']:.4f} "
        f"target_acc={test_metrics['target_acc']:.4f} "
        f"legal={test_metrics['legal']:.4f} "
        f"top3={test_metrics['top3']:.4f} "
        f"scoreexp={test_metrics['teacher_score_expectation']:.4f}"
    )


@torch.no_grad()
def eval_model_policy(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    model = load_model(args.model, device)
    env = PlacementTetrisEnv(seed=args.seed)

    lines_list: List[int] = []
    pieces_list: List[int] = []
    illegal_total = 0

    for ep in range(1, args.eval_episodes + 1):
        env.reset_empty()

        while not env.done and env.total_pieces < args.eval_max_pieces:
            boards, states, legal_masks = env.get_obs()

            boards_t = torch.tensor(boards[None], dtype=torch.float32, device=device)
            states_t = torch.tensor(states[None], dtype=torch.float32, device=device)
            legal_t = torch.tensor(legal_masks[None], dtype=torch.float32, device=device)

            logits = model(boards_t, states_t)
            logits = mask_logits_torch(logits, legal_t)
            action = int(torch.argmax(logits, dim=1).item())

            if legal_masks[action] < 0.5:
                illegal_total += 1
                legal_idx = np.where(legal_masks > 0.5)[0]
                if legal_idx.size == 0:
                    break
                action = int(legal_idx[0])

            _, done = env.step(action)
            if done:
                break

        lines_list.append(env.total_lines)
        pieces_list.append(env.total_pieces)

        print(
            f"[EVAL] ep={ep:03d} pieces={env.total_pieces:04d} "
            f"lines={env.total_lines:04d}"
        )

    arr_lines = np.array(lines_list, dtype=np.float32)
    arr_pieces = np.array(pieces_list, dtype=np.float32)

    print(
        f"[EVAL SUMMARY] "
        f"mean_survival={arr_pieces.mean():.2f} "
        f"std_survival={arr_pieces.std():.2f} "
        f"min_survival={arr_pieces.min():.0f} "
        f"max_survival={arr_pieces.max():.0f} "
        f"mean_lines={arr_lines.mean():.2f} "
        f"max_lines={arr_lines.max():.0f} "
        f"s10={(arr_lines >= 10).mean():.2f} "
        f"s20={(arr_lines >= 20).mean():.2f} "
        f"s30={(arr_lines >= 30).mean():.2f} "
        f"illegal={illegal_total / max(1, len(lines_list)):.2f}"
    )


def eval_search_teacher(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    env = PlacementTetrisEnv(seed=args.seed)

    cfg = SearchConfig(
        depth=args.search_depth,
        beam_width=args.search_beam_width,
        next_piece_samples=args.search_next_piece_samples,
        gamma=args.search_gamma,
        quality_coef=args.search_quality_coef,
        dead_penalty=args.search_dead_penalty,
    )

    lines_list: List[int] = []
    pieces_list: List[int] = []

    for ep in range(1, args.eval_episodes + 1):
        env.reset_empty()

        while not env.done and env.total_pieces < args.eval_max_pieces:
            scores, legal_mask = search_action_scores(
                board=env.board,
                kind=env.current_kind,
                cfg=cfg,
                rng=rng,
            )
            legal_idx = np.where(legal_mask > 0.5)[0]
            if legal_idx.size == 0:
                break

            best_action = int(legal_idx[np.argmax(scores[legal_idx])])
            _, done = env.step(best_action)
            if done:
                break

        lines_list.append(env.total_lines)
        pieces_list.append(env.total_pieces)

        print(
            f"[SEARCH EVAL] ep={ep:03d} pieces={env.total_pieces:04d} "
            f"lines={env.total_lines:04d}"
        )

    arr_lines = np.array(lines_list, dtype=np.float32)
    arr_pieces = np.array(pieces_list, dtype=np.float32)

    print(
        f"[SEARCH SUMMARY] "
        f"mean_survival={arr_pieces.mean():.2f} "
        f"std_survival={arr_pieces.std():.2f} "
        f"min_survival={arr_pieces.min():.0f} "
        f"max_survival={arr_pieces.max():.0f} "
        f"mean_lines={arr_lines.mean():.2f} "
        f"max_lines={arr_lines.max():.0f} "
        f"s10={(arr_lines >= 10).mean():.2f} "
        f"s20={(arr_lines >= 20).mean():.2f} "
        f"s30={(arr_lines >= 30).mean():.2f}"
    )


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Task8 v4: student-failure + teacher relabel + supervised train"
    )

    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["collect", "collect_relabel", "train", "eval", "eval_search"],
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--data", type=str, default="out_task8_stage2/search_dataset_v4.npz")
    p.add_argument("--out_data", type=str, default="out_task8_stage2/search_dataset_v4.npz")
    p.add_argument("--out_dir", type=str, default="out_task8_stage2_model")
    p.add_argument("--model", type=str, default="")
    p.add_argument("--init_model", type=str, default="")
    p.add_argument("--student_model", type=str, default="")

    p.add_argument("--collect_episodes", type=int, default=200)
    p.add_argument("--max_steps_per_episode", type=int, default=120)
    p.add_argument("--log_every_episodes", type=int, default=10)

    p.add_argument("--curriculum_empty_prob", type=float, default=0.40)
    p.add_argument("--curriculum_mid_prob", type=float, default=0.35)
    p.add_argument("--curriculum_hard_prob", type=float, default=0.25)

    p.add_argument("--search_depth", type=int, default=2)
    p.add_argument("--search_beam_width", type=int, default=8)
    p.add_argument("--search_next_piece_samples", type=int, default=2)
    p.add_argument("--search_gamma", type=float, default=0.95)
    p.add_argument("--search_quality_coef", type=float, default=0.60)
    p.add_argument("--search_dead_penalty", type=float, default=-12.0)
    p.add_argument("--teacher_temp", type=float, default=0.90)

    p.add_argument("--student_epsilon", type=float, default=0.05)
    p.add_argument("--collect_online_prob", type=float, default=1.00)
    p.add_argument("--fail_tail_keep", type=int, default=8)
    p.add_argument("--tail_only_flagged", type=int, default=0)
    p.add_argument("--fail_max_h", type=float, default=10.0)
    p.add_argument("--fail_holes", type=float, default=4.0)
    p.add_argument("--fail_bump", type=float, default=18.0)
    p.add_argument("--fail_agg_h", type=float, default=70.0)
    p.add_argument("--fail_near_rows", type=float, default=1.0)
    p.add_argument("--fail_score_gap", type=float, default=0.75)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--soft_coef", type=float, default=0.35)

    p.add_argument("--save_best_by", type=str, default="eval", choices=["eval", "val"])
    p.add_argument("--best_eval_episodes", type=int, default=20)

    p.add_argument("--eval_episodes", type=int, default=50)
    p.add_argument("--eval_max_pieces", type=int, default=500)

    return p


def main() -> None:
    args = build_argparser().parse_args()

    total_prob = (
        args.curriculum_empty_prob
        + args.curriculum_mid_prob
        + args.curriculum_hard_prob
    )
    if total_prob <= 0:
        raise ValueError("curriculum probabilities must sum to positive value")

    args.curriculum_empty_prob /= total_prob
    args.curriculum_mid_prob /= total_prob
    args.curriculum_hard_prob /= total_prob
    args.tail_only_flagged = bool(args.tail_only_flagged)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()

    if args.mode == "collect":
        collect_dataset(args)
    elif args.mode == "collect_relabel":
        if not args.student_model:
            raise ValueError("collect_relabel mode requires --student_model")
        collect_relabel_dataset(args)
    elif args.mode == "train":
        train_model(args)
    elif args.mode == "eval":
        if not args.model:
            raise ValueError("eval mode requires --model")
        eval_model_policy(args)
    elif args.mode == "eval_search":
        eval_search_teacher(args)
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    print(f"[DONE] elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
