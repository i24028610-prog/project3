from __future__ import annotations

import argparse
import ctypes
import json
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import mss
import numpy as np
import torch
import torch.nn as nn

try:
    import pydirectinput as pdi
except Exception:
    pdi = None

try:
    import win32con
    import win32gui
except Exception:
    win32con = None
    win32gui = None


ROWS = 18
COLS = 14
ACTION_CLASSES = 56
STATE_VEC_DIM = 7 + 4 + 14

ROI_FILE = Path("roi.json")
NEXT_ROI_FILE = Path("next_roi.json")

WIN_HUD = "Task1 HUD - Real EXE + Pure NN"
WIN_PREVIEW = "ROI Preview"
WIN_NEXT = "Next Preview"
WIN_SELECT = "Select ROI on SCREEN"

GAME_WINDOW_KEYWORD = "俄罗斯方块"

KEY_LEFT = "left"
KEY_RIGHT = "right"
KEY_ROTATE = "up"
KEY_DROP = "down"
KEY_HARD_DROP = None

PRESS_DELAY = 0.03
ROTATE_INTERVAL = 0.09
MOVE_INTERVAL = 0.07
DROP_INTERVAL = 0.04

SPAWN_COOLDOWN = 0.35
STABLE_SPAWN_FRAMES = 3
ALIGN_STABLE_FRAMES = 2
CLEAR_TARGET_NO_ACTIVE_FRAMES = 2

# 新增：控制前，当前 piece/rot/left 连续稳定多少帧
DETECT_STABLE_FRAMES = 2

# 新增：旧红框过期强制清除参数
STALE_TARGET_STABLE_FRAMES = 2
STALE_TARGET_TOP_MAX_ROW = 2
STALE_TARGET_MIN_AGE_SEC = 0.80
STALE_TARGET_MAX_OVERLAP = 0.34

CENTER_SAMPLE_RATIO = 0.55
BOARD_TEMPORAL_WINDOW = 3
NEXT_TEMPORAL_WINDOW = 4

ACTIVE_MIN_SAT = 40
ACTIVE_MIN_VAL = 70
OCCUPANCY_GRAY_MAX = 246

PREVIEW_MIN_SAT = 35
PREVIEW_MIN_VAL = 70
PREVIEW_MIN_AREA = 40

SPAWN_TOP_MAX_ROW = 6

KIND_LIST = ["I", "O", "T", "S", "Z", "J", "L"]
KIND_TO_ID = {k: i for i, k in enumerate(KIND_LIST)}
ID_TO_KIND = {i: k for i, k in enumerate(KIND_LIST)}

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


@dataclass
class Roi:
    x: int
    y: int
    w: int
    h: int
    monitor_index: int = 1

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def is_valid(self, min_size: int = 30) -> bool:
        return self.w >= min_size and self.h >= min_size


@dataclass
class ActiveInfo:
    mask: np.ndarray
    piece: Optional[str]
    rot_idx: Optional[int]
    left_col: Optional[int]
    top_row: Optional[int]
    cells: List[Tuple[int, int]]


@dataclass
class Target:
    piece: str
    rot_idx: int
    left_col: int
    anchor_col: int
    row: int
    cells: List[Tuple[int, int]]
    action_idx: int
    logit: float
    lines_cleared: int


@dataclass
class FrozenState:
    board_before: np.ndarray
    active_mask: np.ndarray
    piece: str
    cur_rot: int
    cur_col: int


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


def one_hot(index: int, size: int) -> List[float]:
    out = [0.0] * size
    if 0 <= index < size:
        out[index] = 1.0
    return out


def build_state_vector(kind_id: int, cur_rot4: int, cur_col14: int) -> List[float]:
    return one_hot(kind_id, 7) + one_hot(cur_rot4, 4) + one_hot(cur_col14, COLS)


def decode_action(action_idx: int) -> Tuple[int, int]:
    rot4 = int(action_idx // COLS)
    col14 = int(action_idx % COLS)
    return rot4, col14


def flatten_4x14(matrix_4x14: Sequence[Sequence[float]]) -> List[float]:
    out: List[float] = []
    for rot in range(4):
        for col in range(COLS):
            out.append(float(matrix_4x14[rot][col]))
    return out


def occ_board(board: Sequence[Sequence[int]]) -> List[List[int]]:
    return [[1 if safe_int(cell) != 0 else 0 for cell in row] for row in board]


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


def apply_legal_mask(
    logits: torch.Tensor,
    legal_mask_flat: torch.Tensor,
    large_negative: float = -1e9,
) -> torch.Tensor:
    if legal_mask_flat.dim() == 1:
        legal_mask_flat = legal_mask_flat.unsqueeze(0)
    return logits.masked_fill(legal_mask_flat <= 0.5, large_negative)


def left_col14_to_anchor_col(kind: str, rot4: int, left_col14: int) -> int:
    rot_idx = rot4 % len(SHAPES[kind])
    min_c = min(c for _, c in SHAPES[kind][rot_idx])
    return int(left_col14 - min_c)


def is_valid_position_on_board(board_occ: Sequence[Sequence[int]], piece: Piece) -> bool:
    for row_idx, col_idx in piece.cells():
        if not (0 <= row_idx < ROWS and 0 <= col_idx < COLS):
            return False
        if safe_int(board_occ[row_idx][col_idx]) != 0:
            return False
    return True


def hard_drop_row_on_board(board_occ: Sequence[Sequence[int]], piece: Piece) -> int:
    test = Piece(piece.kind, piece.rotation, piece.row, piece.col)
    while True:
        nxt = Piece(test.kind, test.rotation, test.row + 1, test.col)
        if is_valid_position_on_board(board_occ, nxt):
            test = nxt
        else:
            return test.row


def build_legal_mask_4x14_from_board(
    board_occ: Sequence[Sequence[int]],
    kind: str,
) -> List[List[int]]:
    legal = [[0 for _ in range(COLS)] for _ in range(4)]
    rot_total = len(SHAPES[kind])
    for rot4 in range(4):
        if rot4 >= rot_total:
            continue
        for left_col14 in range(COLS):
            anchor_col = left_col14_to_anchor_col(kind, rot4, left_col14)
            spawn_piece = Piece(kind=kind, rotation=rot4, row=0, col=anchor_col)
            if is_valid_position_on_board(board_occ, spawn_piece):
                legal[rot4][left_col14] = 1
    return legal


def place_piece_on_board(board_occ: Sequence[Sequence[int]], piece: Piece) -> np.ndarray:
    out = np.array(occ_board(board_occ), dtype=np.uint8)
    for r, c in piece.cells():
        if 0 <= r < ROWS and 0 <= c < COLS:
            out[r, c] = 1
    return out


def make_process_dpi_aware() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def load_roi_from_file(path: Path) -> Optional[Roi]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        roi = Roi(
            x=int(data["x"]),
            y=int(data["y"]),
            w=int(data["w"]),
            h=int(data["h"]),
            monitor_index=int(data.get("monitor_index", 1)),
        )
        return roi if roi.is_valid() else None
    except Exception:
        return None


def save_roi_to_file(path: Path, roi: Roi) -> None:
    path.write_text(
        json.dumps(
            {
                "x": roi.x,
                "y": roi.y,
                "w": roi.w,
                "h": roi.h,
                "monitor_index": roi.monitor_index,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_roi() -> Optional[Roi]:
    return load_roi_from_file(ROI_FILE)


def load_next_roi() -> Optional[Roi]:
    return load_roi_from_file(NEXT_ROI_FILE)


def save_roi(roi: Roi) -> None:
    save_roi_to_file(ROI_FILE, roi)


def save_next_roi(roi: Roi) -> None:
    save_roi_to_file(NEXT_ROI_FILE, roi)


def clamp_roi(roi: Roi, frame_w: int, frame_h: int) -> Roi:
    x = max(0, min(roi.x, frame_w - 1))
    y = max(0, min(roi.y, frame_h - 1))
    w = max(1, min(roi.w, frame_w - x))
    h = max(1, min(roi.h, frame_h - y))
    return Roi(x=x, y=y, w=w, h=h, monitor_index=roi.monitor_index)


def hide_windows() -> None:
    for name in (WIN_HUD, WIN_PREVIEW, WIN_NEXT, WIN_SELECT):
        try:
            cv2.destroyWindow(name)
        except cv2.error:
            pass


def grab_screen(monitor_index: int) -> Tuple[np.ndarray, int, int, int]:
    with mss.mss() as sct:
        monitor_count = len(sct.monitors) - 1
        monitor_index = max(1, min(monitor_index, max(1, monitor_count)))
        mon = sct.monitors[monitor_index]
        shot = sct.grab(mon)
        img = np.array(shot, dtype=np.uint8)
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return frame, int(mon["width"]), int(mon["height"]), int(monitor_count)


def select_roi_on_clean_screenshot(
    monitor_index: int,
) -> Optional[Tuple[Roi, int, int, int]]:
    hide_windows()
    time.sleep(0.35)
    frame, screen_w, screen_h, monitor_count = grab_screen(monitor_index)

    rect = cv2.selectROI(
        WIN_SELECT,
        frame,
        showCrosshair=False,
        fromCenter=False,
    )
    cv2.destroyWindow(WIN_SELECT)

    x, y, w, h = [int(v) for v in rect]
    roi = Roi(x=x, y=y, w=w, h=h, monitor_index=monitor_index)
    if not roi.is_valid():
        return None
    roi = clamp_roi(roi, screen_w, screen_h)
    return roi, screen_w, screen_h, monitor_count


def find_game_window(keyword: str) -> Optional[int]:
    if win32gui is None:
        return None

    result: List[int] = []

    def _enum_cb(hwnd, _):
        try:
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if keyword in title:
                    result.append(hwnd)
        except Exception:
            pass

    try:
        win32gui.EnumWindows(_enum_cb, None)
    except Exception:
        return None

    return result[0] if result else None


def focus_game_window() -> bool:
    if win32gui is None or win32con is None:
        return False
    hwnd = find_game_window(GAME_WINDOW_KEYWORD)
    if hwnd is None:
        return False
    try:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.02)
        return True
    except Exception:
        return False


def compute_edges(total: int, parts: int) -> np.ndarray:
    edges = np.linspace(0, total, parts + 1)
    return np.rint(edges).astype(int)


def center_subregion(
    x1: int,
    x2: int,
    y1: int,
    y2: int,
    ratio: float,
) -> Tuple[int, int, int, int]:
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    sw = max(2, int(round(w * ratio)))
    sh = max(2, int(round(h * ratio)))
    cx1 = x1 + (w - sw) // 2
    cy1 = y1 + (h - sh) // 2
    cx2 = min(x2, cx1 + sw)
    cy2 = min(y2, cy1 + sh)
    return cx1, cx2, cy1, cy2


def roi_to_masks(roi_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    x_edges = compute_edges(w, COLS)
    y_edges = compute_edges(h, ROWS)

    occupancy = np.zeros((ROWS, COLS), dtype=np.uint8)
    active = np.zeros((ROWS, COLS), dtype=np.uint8)

    for r in range(ROWS):
        y1, y2 = int(y_edges[r]), int(y_edges[r + 1])
        for c in range(COLS):
            x1, x2 = int(x_edges[c]), int(x_edges[c + 1])
            cx1, cx2, cy1, cy2 = center_subregion(x1, x2, y1, y2, CENTER_SAMPLE_RATIO)
            if cx2 <= cx1 or cy2 <= cy1:
                continue

            gray_patch = gray[cy1:cy2, cx1:cx2]
            hsv_patch = hsv[cy1:cy2, cx1:cx2]

            mean_gray = float(np.mean(gray_patch))
            mean_sat = float(np.mean(hsv_patch[:, :, 1]))
            mean_val = float(np.mean(hsv_patch[:, :, 2]))

            if mean_gray < OCCUPANCY_GRAY_MAX:
                occupancy[r, c] = 1
            if mean_sat >= ACTIVE_MIN_SAT and mean_val >= ACTIVE_MIN_VAL:
                active[r, c] = 1

    return occupancy, active


def majority_vote(grids: Deque[np.ndarray]) -> np.ndarray:
    if not grids:
        return np.zeros((ROWS, COLS), dtype=np.uint8)
    stack = np.stack(list(grids), axis=0).astype(np.uint8)
    votes = np.sum(stack, axis=0)
    return (votes >= (len(grids) // 2 + 1)).astype(np.uint8)


def subtract_active(occupancy: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    board_before = occupancy.copy().astype(np.uint8)
    board_before[active_mask > 0] = 0
    return board_before


def normalize_cells(cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted([(r - min_r, c - min_c) for r, c in cells])


def mask_to_cells(mask: np.ndarray) -> List[Tuple[int, int]]:
    ys, xs = np.where(mask > 0)
    return list(zip(ys.tolist(), xs.tolist()))


def exact_match_piece_and_rot(cells: List[Tuple[int, int]]) -> Tuple[Optional[str], Optional[int]]:
    norm = normalize_cells(cells)
    for piece, rots in SHAPES.items():
        for rot_idx, shape in enumerate(rots):
            if len(shape) != len(cells):
                continue
            if norm == normalize_cells(shape):
                return piece, rot_idx
    return None, None


def try_recover_component(
    cells: List[Tuple[int, int]],
    active_grid: np.ndarray,
    occupancy_grid: np.ndarray,
) -> Tuple[List[Tuple[int, int]], Optional[str], Optional[int]]:
    if len(cells) not in (3, 4):
        return cells, None, None

    piece, rot_idx = exact_match_piece_and_rot(cells)
    if piece is not None:
        return cells, piece, rot_idx

    if len(cells) != 3:
        return cells, None, None

    cell_set = set(cells)
    candidates = set()
    for r, c in cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < ROWS and 0 <= cc < COLS and (rr, cc) not in cell_set:
                if occupancy_grid[rr, cc] == 1 and active_grid[rr, cc] == 0:
                    candidates.add((rr, cc))

    matches: List[Tuple[List[Tuple[int, int]], str, int]] = []
    for add_cell in candidates:
        test_cells = cells + [add_cell]
        p, r = exact_match_piece_and_rot(test_cells)
        if p is not None and r is not None:
            matches.append((test_cells, p, r))

    if len(matches) == 1:
        return matches[0]
    return cells, None, None


def choose_active_component(active_grid: np.ndarray, occupancy_grid: np.ndarray) -> ActiveInfo:
    mask = (active_grid > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)

    candidates: List[Tuple[Tuple[int, int, int], List[Tuple[int, int]], str, int]] = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 3 or area > 5:
            continue

        comp = (labels == label).astype(np.uint8)
        cells = mask_to_cells(comp)
        if not cells:
            continue

        recovered_cells, piece, rot_idx = try_recover_component(cells, active_grid, occupancy_grid)
        if piece is None or rot_idx is None:
            continue

        top_row = min(r for r, _ in recovered_cells)
        left_col = min(c for _, c in recovered_cells)
        score = (top_row, left_col, abs(len(recovered_cells) - len(cells)))
        candidates.append((score, recovered_cells, piece, rot_idx))

    if not candidates:
        return ActiveInfo(
            mask=np.zeros((ROWS, COLS), dtype=np.uint8),
            piece=None,
            rot_idx=None,
            left_col=None,
            top_row=None,
            cells=[],
        )

    candidates.sort(key=lambda x: x[0])
    _, best_cells, piece, rot_idx = candidates[0]

    full_mask = np.zeros((ROWS, COLS), dtype=np.uint8)
    for rr, cc in best_cells:
        if 0 <= rr < ROWS and 0 <= cc < COLS:
            full_mask[rr, cc] = 1

    return ActiveInfo(
        mask=full_mask,
        piece=piece,
        rot_idx=rot_idx,
        left_col=min(c for _, c in best_cells),
        top_row=min(r for r, _ in best_cells),
        cells=best_cells,
    )


def _largest_component_mask(bin_img: np.ndarray, min_area: int) -> Optional[np.ndarray]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    best_label = -1
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_label = label
            best_area = area
    if best_label <= 0:
        return None
    out = np.zeros_like(bin_img)
    out[labels == best_label] = 255
    return out


def _normalize_binary_patch(mask: np.ndarray, size: int = 64) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((size, size), dtype=np.uint8)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    crop = mask[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((size, size), dtype=np.uint8)
    ox = (size - new_w) // 2
    oy = (size - new_h) // 2
    out[oy:oy + new_h, ox:ox + new_w] = resized
    return out


def _render_shape_mask(shape: List[Tuple[int, int]], size: int = 64) -> np.ndarray:
    cell = size // 4
    canvas = np.zeros((size, size), dtype=np.uint8)
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    sh = max_r - min_r + 1
    sw = max_c - min_c + 1
    ox = (size - sw * cell) // 2
    oy = (size - sh * cell) // 2
    for r, c in shape:
        rr = oy + (r - min_r) * cell
        cc = ox + (c - min_c) * cell
        canvas[rr:rr + cell, cc:cc + cell] = 255
    return canvas


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = (a > 0).astype(np.uint8)
    b_bin = (b > 0).astype(np.uint8)
    inter = int(np.sum((a_bin & b_bin) > 0))
    union = int(np.sum((a_bin | b_bin) > 0))
    if union <= 0:
        return 0.0
    return inter / float(union)


def detect_next_piece_kind(preview_bgr: np.ndarray) -> Tuple[Optional[str], Optional[np.ndarray], float]:
    hsv = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    bin_img = np.zeros(preview_bgr.shape[:2], dtype=np.uint8)
    bin_img[(sat >= PREVIEW_MIN_SAT) & (val >= PREVIEW_MIN_VAL)] = 255

    kernel = np.ones((3, 3), dtype=np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    comp = _largest_component_mask(bin_img, PREVIEW_MIN_AREA)
    if comp is None:
        return None, bin_img, 0.0

    norm = _normalize_binary_patch(comp, size=64)
    best_piece = None
    best_score = -1.0
    for piece, rots in SHAPES.items():
        local_best = -1.0
        for rot in range(len(rots)):
            tmpl = _render_shape_mask(rots[rot], size=64)
            score = _iou(norm, tmpl)
            if score > local_best:
                local_best = score
        if local_best > best_score:
            best_score = local_best
            best_piece = piece

    if best_score < 0.18:
        return None, comp, best_score
    return best_piece, comp, best_score


def majority_kind(kinds: Deque[Optional[str]]) -> Optional[str]:
    vals = [k for k in kinds if k is not None]
    if not vals:
        return None
    counter = Counter(vals)
    return counter.most_common(1)[0][0]


class Task6RealtimePolicy:
    def __init__(self, model_path: str, cpu: bool = False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
        ckpt = torch.load(model_path, map_location=self.device)
        self.model = ActionPolicyNet().to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.ckpt = ckpt

    @torch.no_grad()
    def predict_target(self, frozen: FrozenState) -> Optional[Target]:
        board_before = occ_board(frozen.board_before.tolist())
        active_mask = occ_board(frozen.active_mask.tolist())
        kind = frozen.piece
        kind_id = KIND_TO_ID[kind]

        legal_mask_4x14 = build_legal_mask_4x14_from_board(board_before, kind)
        legal_flat_list = flatten_4x14(legal_mask_4x14)
        if sum(1 for x in legal_flat_list if x > 0.5) <= 0:
            return None

        board_tensor = torch.tensor(
            [[board_before, active_mask]],
            dtype=torch.float32,
            device=self.device,
        )
        state_tensor = torch.tensor(
            [build_state_vector(kind_id, frozen.cur_rot, frozen.cur_col)],
            dtype=torch.float32,
            device=self.device,
        )
        legal_flat = torch.tensor(
            [legal_flat_list],
            dtype=torch.float32,
            device=self.device,
        )

        logits = self.model(board_tensor, state_tensor)
        masked_logits = apply_legal_mask(logits, legal_flat)

        action_idx = int(masked_logits.argmax(dim=1).item())
        pred_rot4, pred_col14 = decode_action(action_idx)

        if pred_rot4 >= len(SHAPES[kind]):
            return None

        anchor_col = left_col14_to_anchor_col(kind, pred_rot4, pred_col14)
        spawn_piece = Piece(kind=kind, rotation=pred_rot4, row=0, col=anchor_col)
        if not is_valid_position_on_board(board_before, spawn_piece):
            return None

        drop_row = hard_drop_row_on_board(board_before, spawn_piece)
        final_piece = Piece(kind=kind, rotation=pred_rot4, row=drop_row, col=anchor_col)

        return Target(
            piece=kind,
            rot_idx=pred_rot4,
            left_col=pred_col14,
            anchor_col=anchor_col,
            row=drop_row,
            cells=final_piece.cells(),
            action_idx=action_idx,
            logit=float(masked_logits[0, action_idx].item()),
            lines_cleared=0,
        )


def choose_control_action(
    current_piece: str,
    cur_rot: int,
    cur_left: int,
    target: Target,
    aligned_frames: int,
) -> str:
    rot_total = len(SHAPES[current_piece])
    rot_gap = (target.rot_idx - cur_rot) % rot_total

    if rot_gap != 0:
        return "ROTATE"
    if cur_left > target.left_col:
        return "LEFT"
    if cur_left < target.left_col:
        return "RIGHT"
    if aligned_frames >= ALIGN_STABLE_FRAMES:
        return "DROP"
    return "NONE"


def control_interval(action: str) -> float:
    if action == "ROTATE":
        return ROTATE_INTERVAL
    if action in ("LEFT", "RIGHT"):
        return MOVE_INTERVAL
    return DROP_INTERVAL


def press_key(key: str) -> None:
    if pdi is None:
        return
    focus_game_window()
    pdi.keyDown(key)
    time.sleep(PRESS_DELAY)
    pdi.keyUp(key)


def target_cells_filled(occupancy: np.ndarray, target: Target) -> bool:
    for r, c in target.cells:
        if not (0 <= r < ROWS and 0 <= c < COLS):
            return False
        if occupancy[r, c] == 0:
            return False
    return True


def active_overlap_ratio(active_cells: List[Tuple[int, int]], target: Target) -> float:
    if not active_cells:
        return 0.0
    active_set = set(active_cells)
    target_set = set(target.cells)
    inter = len(active_set & target_set)
    denom = max(1, len(target_set))
    return inter / float(denom)


def active_cells_match_target(active_cells: List[Tuple[int, int]], target: Target) -> bool:
    return active_overlap_ratio(active_cells, target) >= 0.67


def draw_overlay(
    roi_bgr: np.ndarray,
    active_mask: np.ndarray,
    target_cells: Optional[List[Tuple[int, int]]],
) -> np.ndarray:
    out = roi_bgr.copy()
    h, w = out.shape[:2]
    x_edges = compute_edges(w, COLS)
    y_edges = compute_edges(h, ROWS)

    for c in range(1, COLS):
        x = int(x_edges[c])
        cv2.line(out, (x, 0), (x, h), (70, 70, 70), 1)
    for r in range(1, ROWS):
        y = int(y_edges[r])
        cv2.line(out, (0, y), (w, y), (70, 70, 70), 1)

    ys, xs = np.where(active_mask > 0)
    for r, c in zip(ys.tolist(), xs.tolist()):
        x1, x2 = int(x_edges[c]), int(x_edges[c + 1])
        y1, y2 = int(y_edges[r]), int(y_edges[r + 1])
        cv2.rectangle(out, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), (0, 255, 0), 2)

    if target_cells:
        for r, c in target_cells:
            x1, x2 = int(x_edges[c]), int(x_edges[c + 1])
            y1, y2 = int(y_edges[r]), int(y_edges[r + 1])
            cv2.rectangle(out, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), (0, 0, 255), 2)

    return out


def draw_next_overlay(
    preview_bgr: np.ndarray,
    mask: Optional[np.ndarray],
    piece: Optional[str],
    score: float,
) -> np.ndarray:
    out = preview_bgr.copy()
    if mask is not None:
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
    text = f"Next: {piece}  score={score:.2f}" if piece else f"Next: None  score={score:.2f}"
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2, cv2.LINE_AA)
    return out


def make_hud(
    roi: Optional[Roi],
    next_roi: Optional[Roi],
    active: ActiveInfo,
    target: Optional[Target],
    model_ok: bool,
    device_name: str,
    next_piece: Optional[str],
    next_score: float,
    last_action: str,
    target_locked_now: bool,
    active_overlap_now: bool,
    detect_stable: bool,
) -> np.ndarray:
    hud = np.zeros((472, 620, 3), dtype=np.uint8)
    lines = [
        "Task1 Real EXE + Pure Neural Network Decision",
        f"Win focus: {'ON' if win32gui is not None else 'OFF (pip install pywin32)'}",
        f"Model: {'OK' if model_ok else 'FAILED'} | Device: {device_name}",
        "",
        f"Board ROI: {roi}" if roi else "Board ROI: (not set)",
        f"Next ROI: {next_roi}" if next_roi else "Next ROI: (not set)",
        f"Detected piece: {active.piece}",
        f"Detected rot: {active.rot_idx}",
        f"Detected left col: {active.left_col}",
        f"Active cells: {len(active.cells)}",
        f"Detect stable: {detect_stable}",
        f"Detected next: {next_piece}",
        f"Next match score: {next_score:.2f}",
        f"Last action: {last_action}",
        f"Target filled on board: {target_locked_now}",
        f"Active overlap target: {active_overlap_now}",
    ]

    if target is not None:
        lines.append(
            f"Target: piece={target.piece} rot={target.rot_idx} left={target.left_col} row={target.row} logit={target.logit:.3f}"
        )
    else:
        lines.append("Target: None")

    y = 26
    for i, text in enumerate(lines):
        scale = 0.60 if i == 0 else 0.56
        thickness = 2 if i == 0 else 1
        cv2.putText(
            hud,
            text,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (230, 230, 230),
            thickness,
            cv2.LINE_AA,
        )
        y += 22
    return hud


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=r"C:\Users\25361\PycharmProjects\pythonProject3\out_task6\model_best_action.pt",
    )
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    make_process_dpi_aware()

    if pdi is not None:
        try:
            pdi.FAILSAFE = False
            pdi.PAUSE = 0.0
        except Exception:
            pass

    roi = load_roi()
    next_roi = load_next_roi()
    monitor_index = roi.monitor_index if roi else (next_roi.monitor_index if next_roi else 1)

    target: Optional[Target] = None
    target_cells: Optional[List[Tuple[int, int]]] = None
    frozen: Optional[FrozenState] = None
    aligned_frames = 0

    spawn_candidate: Optional[Tuple[str, int, int]] = None
    spawn_stable_count = 0
    no_active_count = 0
    target_piece_disappeared = False
    target_contact_latched = False

    waiting_new_piece_after_old = False
    new_piece_candidate: Optional[Tuple[str, int, int]] = None
    new_piece_stable_count = 0

    # 新增：旧红框过期强制清除
    stale_target_candidate: Optional[Tuple[str, int, int]] = None
    stale_target_count = 0

    # 新增：控制前 2 帧稳定确认
    detect_candidate: Optional[Tuple[str, int, int]] = None
    detect_stable_count = 0

    last_spawn_time = 0.0
    last_action_time = 0.0
    ignore_esc_until = 0.0
    last_action = "NONE"

    board_hist: Deque[np.ndarray] = deque(maxlen=BOARD_TEMPORAL_WINDOW)
    next_hist: Deque[Optional[str]] = deque(maxlen=NEXT_TEMPORAL_WINDOW)

    next_piece_now: Optional[str] = None
    next_piece_score: float = 0.0
    next_piece_mask: Optional[np.ndarray] = None

    model_ok = True
    policy: Optional[Task6RealtimePolicy] = None
    device_name = "none"

    try:
        policy = Task6RealtimePolicy(args.model, cpu=args.cpu)
        device_name = str(policy.device)
        print(f"[INFO] model loaded: {args.model}")
        print(f"[INFO] device: {device_name}")
    except Exception as e:
        model_ok = False
        print(f"[WARN] model load failed: {e}")

    while True:
        frame, screen_w, screen_h, monitor_count = grab_screen(monitor_index)
        active_overlap_now = False
        detect_stable = False

        if roi is None:
            hud = make_hud(
                roi=None,
                next_roi=next_roi,
                active=ActiveInfo(
                    mask=np.zeros((ROWS, COLS), dtype=np.uint8),
                    piece=None,
                    rot_idx=None,
                    left_col=None,
                    top_row=None,
                    cells=[],
                ),
                target=None,
                model_ok=model_ok,
                device_name=device_name,
                next_piece=next_piece_now,
                next_score=next_piece_score,
                last_action=last_action,
                target_locked_now=False,
                active_overlap_now=False,
                detect_stable=False,
            )
            cv2.imshow(WIN_HUD, hud)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("e"):
                result = select_roi_on_clean_screenshot(monitor_index)
                if result is not None:
                    roi, _, _, _ = result
                    ignore_esc_until = time.time() + 0.6
            if key == ord("n"):
                result = select_roi_on_clean_screenshot(monitor_index)
                if result is not None:
                    next_roi, _, _, _ = result
                    ignore_esc_until = time.time() + 0.6
            if key == ord("m"):
                monitor_index += 1
                if monitor_index > max(1, monitor_count):
                    monitor_index = 1
            continue

        roi.monitor_index = monitor_index
        roi = clamp_roi(roi, screen_w, screen_h)
        if next_roi is not None:
            next_roi.monitor_index = monitor_index
            next_roi = clamp_roi(next_roi, screen_w, screen_h)

        x, y, rw, rh = roi.as_tuple()
        roi_bgr = frame[y:y + rh, x:x + rw]

        if next_roi is not None:
            nx, ny, nw, nh = next_roi.as_tuple()
            next_roi_bgr = frame[ny:ny + nh, nx:nx + nw]
            next_piece_raw, next_piece_mask, next_piece_score = detect_next_piece_kind(next_roi_bgr)
            next_hist.append(next_piece_raw)
            next_piece_now = majority_kind(next_hist)
            next_preview = draw_next_overlay(
                next_roi_bgr,
                next_piece_mask,
                next_piece_now,
                next_piece_score,
            )
            cv2.imshow(WIN_NEXT, next_preview)
        else:
            next_piece_now = None
            next_piece_score = 0.0
            next_piece_mask = None
            next_hist.clear()

        occupancy_grid, active_color_grid = roi_to_masks(roi_bgr)
        board_hist.append(occupancy_grid)
        occupancy_s = majority_vote(board_hist)
        active_s = active_color_grid.copy()
        active = choose_active_component(active_s, occupancy_s)
        board_before = subtract_active(occupancy_s, active.mask)
        now = time.time()

        # ------------------------------------------------------------
        # 新增：当前检测值稳定确认
        # ------------------------------------------------------------
        if (
            active.piece is not None
            and active.rot_idx is not None
            and active.left_col is not None
            and len(active.cells) in (3, 4)
        ):
            detect_sig = (active.piece, int(active.rot_idx), int(active.left_col))
            if detect_sig == detect_candidate:
                detect_stable_count += 1
            else:
                detect_candidate = detect_sig
                detect_stable_count = 1
        else:
            detect_candidate = None
            detect_stable_count = 0

        detect_stable = detect_stable_count >= DETECT_STABLE_FRAMES

        target_locked_now = False
        active_overlap_now = False

        # ------------------------------------------------------------
        # target 生命周期
        # ------------------------------------------------------------
        if target is not None:
            target_locked_now = target_cells_filled(occupancy_s, target)
            overlap_ratio_now = active_overlap_ratio(active.cells, target)
            active_overlap_now = overlap_ratio_now >= 0.67

            if active_overlap_now:
                target_contact_latched = True

            # --------------------------------------------------------
            # 新增：旧红框过期强制清除
            # 条件：
            # 1) target 已经存在一段时间
            # 2) 顶部出现一个稳定的新块
            # 3) 这个新块和旧红框几乎不重合
            # --------------------------------------------------------
            if (
                (now - last_spawn_time) >= STALE_TARGET_MIN_AGE_SEC
                and active.piece is not None
                and active.rot_idx is not None
                and active.left_col is not None
                and active.top_row is not None
                and len(active.cells) in (3, 4)
                and active.top_row <= STALE_TARGET_TOP_MAX_ROW
            ):
                overlap_now = active_overlap_ratio(active.cells, target)
                if overlap_now < STALE_TARGET_MAX_OVERLAP:
                    sig_now = (active.piece, int(active.rot_idx), int(active.left_col))
                    if sig_now == stale_target_candidate:
                        stale_target_count += 1
                    else:
                        stale_target_candidate = sig_now
                        stale_target_count = 1

                    if stale_target_count >= STALE_TARGET_STABLE_FRAMES:
                        target = None
                        target_cells = None
                        frozen = None
                        aligned_frames = 0
                        no_active_count = 0
                        target_piece_disappeared = False
                        target_contact_latched = False
                        waiting_new_piece_after_old = False
                        new_piece_candidate = None
                        new_piece_stable_count = 0
                        stale_target_candidate = None
                        stale_target_count = 0
                        spawn_candidate = None
                        spawn_stable_count = 0
                        last_action = "NONE"
                        last_spawn_time = now
                else:
                    stale_target_candidate = None
                    stale_target_count = 0
            else:
                stale_target_candidate = None
                stale_target_count = 0

        if target is not None:
            if target_locked_now:
                target = None
                target_cells = None
                frozen = None
                aligned_frames = 0
                no_active_count = 0
                target_piece_disappeared = False
                target_contact_latched = False
                waiting_new_piece_after_old = False
                new_piece_candidate = None
                new_piece_stable_count = 0
                stale_target_candidate = None
                stale_target_count = 0
                spawn_candidate = None
                spawn_stable_count = 0
                last_action = "NONE"
                last_spawn_time = now
            else:
                if len(active.cells) == 0:
                    no_active_count += 1
                    if no_active_count >= CLEAR_TARGET_NO_ACTIVE_FRAMES:
                        target_piece_disappeared = True
                        waiting_new_piece_after_old = True
                else:
                    no_active_count = 0

                if waiting_new_piece_after_old:
                    if (
                        active.piece is not None
                        and active.rot_idx is not None
                        and active.left_col is not None
                        and active.top_row is not None
                        and len(active.cells) in (3, 4)
                        and active.top_row <= SPAWN_TOP_MAX_ROW
                    ):
                        sig_new = (active.piece, active.rot_idx, active.left_col)
                        if sig_new == new_piece_candidate:
                            new_piece_stable_count += 1
                        else:
                            new_piece_candidate = sig_new
                            new_piece_stable_count = 1

                        if new_piece_stable_count >= STABLE_SPAWN_FRAMES:
                            target = None
                            target_cells = None
                            frozen = None
                            aligned_frames = 0
                            no_active_count = 0
                            target_piece_disappeared = False
                            target_contact_latched = False
                            waiting_new_piece_after_old = False
                            new_piece_candidate = None
                            new_piece_stable_count = 0
                            stale_target_candidate = None
                            stale_target_count = 0
                            spawn_candidate = None
                            spawn_stable_count = 0
                            last_action = "NONE"
                            last_spawn_time = now
                    else:
                        new_piece_candidate = None
                        new_piece_stable_count = 0

        # ------------------------------------------------------------
        # 重新规划 target
        # ------------------------------------------------------------
        if target is None:
            if (
                policy is not None
                and active.piece is not None
                and active.rot_idx is not None
                and active.left_col is not None
                and active.top_row is not None
                and len(active.cells) in (3, 4)
                and active.top_row <= SPAWN_TOP_MAX_ROW
                and (now - last_spawn_time) > SPAWN_COOLDOWN
            ):
                sig = (active.piece, active.rot_idx, active.left_col)
                if sig == spawn_candidate:
                    spawn_stable_count += 1
                else:
                    spawn_candidate = sig
                    spawn_stable_count = 1

                if spawn_stable_count >= STABLE_SPAWN_FRAMES:
                    frozen = FrozenState(
                        board_before=board_before.copy(),
                        active_mask=active.mask.copy(),
                        piece=active.piece,
                        cur_rot=int(active.rot_idx),
                        cur_col=int(active.left_col),
                    )
                    try:
                        best = policy.predict_target(frozen)
                    except Exception as e:
                        print(f"[WARN] model predict failed: {e}")
                        best = None

                    if best is not None:
                        target = best
                        target_cells = best.cells
                        aligned_frames = 0
                        target_piece_disappeared = False
                        target_contact_latched = False
                        waiting_new_piece_after_old = False
                        new_piece_candidate = None
                        new_piece_stable_count = 0
                        stale_target_candidate = None
                        stale_target_count = 0
                        no_active_count = 0
                        last_spawn_time = now
                        print(
                            f"[PLAN] piece={best.piece} rot={best.rot_idx} "
                            f"left={best.left_col} row={best.row} logit={best.logit:.3f}"
                        )
                    else:
                        print("[WARN] invalid target from model.")

                    spawn_candidate = None
                    spawn_stable_count = 0
            else:
                spawn_candidate = None
                spawn_stable_count = 0

        # ------------------------------------------------------------
        # 控制当前块朝 target 走
        # 新增：必须先 detect_stable 才允许发控制动作
        # ------------------------------------------------------------
        if (
            target is not None
            and active.piece is not None
            and active.rot_idx is not None
            and active.left_col is not None
            and active.piece == target.piece
        ):
            if detect_stable:
                if active.rot_idx == target.rot_idx and active.left_col == target.left_col:
                    aligned_frames += 1
                else:
                    aligned_frames = 0

                action = choose_control_action(
                    current_piece=active.piece,
                    cur_rot=int(active.rot_idx),
                    cur_left=int(active.left_col),
                    target=target,
                    aligned_frames=aligned_frames,
                )

                if action != "NONE":
                    wait_t = control_interval(action)
                    if pdi is not None and (now - last_action_time) >= wait_t:
                        if action == "ROTATE":
                            press_key(KEY_ROTATE)
                        elif action == "LEFT":
                            press_key(KEY_LEFT)
                        elif action == "RIGHT":
                            press_key(KEY_RIGHT)
                        elif action == "DROP":
                            if KEY_HARD_DROP is not None:
                                press_key(KEY_HARD_DROP)
                            else:
                                press_key(KEY_DROP)
                        last_action = action
                        last_action_time = now
                else:
                    last_action = "HOLD"
            else:
                aligned_frames = 0
                last_action = "WAIT_STABLE"

            if target is not None:
                target_locked_now = target_cells_filled(occupancy_s, target)
                active_overlap_now = active_cells_match_target(active.cells, target)
        else:
            aligned_frames = 0

        preview = draw_overlay(roi_bgr, active.mask, target_cells)
        cv2.imshow(WIN_PREVIEW, preview)

        hud = make_hud(
            roi=roi,
            next_roi=next_roi,
            active=active,
            target=target,
            model_ok=model_ok,
            device_name=device_name,
            next_piece=next_piece_now,
            next_score=next_piece_score,
            last_action=last_action,
            target_locked_now=target_locked_now,
            active_overlap_now=active_overlap_now,
            detect_stable=detect_stable,
        )
        cv2.imshow(WIN_HUD, hud)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 and time.time() < ignore_esc_until:
            key = 0

        if key == ord("q"):
            break
        if key == ord("m"):
            monitor_index += 1
            if monitor_index > max(1, monitor_count):
                monitor_index = 1
        if key == ord("e"):
            result = select_roi_on_clean_screenshot(monitor_index)
            if result is not None:
                roi, _, _, _ = result
                target = None
                target_cells = None
                frozen = None
                aligned_frames = 0
                spawn_candidate = None
                spawn_stable_count = 0
                no_active_count = 0
                target_piece_disappeared = False
                target_contact_latched = False
                waiting_new_piece_after_old = False
                new_piece_candidate = None
                new_piece_stable_count = 0
                stale_target_candidate = None
                stale_target_count = 0
                detect_candidate = None
                detect_stable_count = 0
                board_hist.clear()
                ignore_esc_until = time.time() + 0.6
        if key == ord("n"):
            result = select_roi_on_clean_screenshot(monitor_index)
            if result is not None:
                next_roi, _, _, _ = result
                next_hist.clear()
                ignore_esc_until = time.time() + 0.6
        if key == ord("s"):
            save_roi(roi)
            print(f"[INFO] board ROI saved -> {ROI_FILE}")
            if next_roi is not None:
                save_next_roi(next_roi)
                print(f"[INFO] next ROI saved -> {NEXT_ROI_FILE}")

    hide_windows()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()