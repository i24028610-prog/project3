from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np

try:
    import pydirectinput as pdi
except Exception:
    pdi = None


# =========================
# Basic config
# =========================
ROWS = 18
COLS = 14
ROI_FILE = Path("roi.json")

WIN_HUD = "Task4 HUD"
WIN_PREVIEW = "ROI Preview"
WIN_SELECT = "Select ROI on SCREEN (ENTER/SPACE confirm, C cancel)"

# -------- key mapping --------
KEY_LEFT = "left"
KEY_RIGHT = "right"
KEY_ROTATE = "up"      # 你的游戏旋转键不同就改这里
KEY_DROP = "down"      # soft drop（连续按多次模拟快速下落）

# -------- timing --------
PRESS_DELAY = 0.03
ACTION_INTERVAL = 0.05
SPAWN_COOLDOWN = 0.35
STABLE_SPAWN_FRAMES = 3
DROP_REPEAT = 18

# -------- cell sampling --------
CENTER_SAMPLE_RATIO = 0.55

# -------- temporal smoothing --------
TEMPORAL_WINDOW = 3  # 3帧多数投票

# -------- active colored piece detection (HSV) --------
ACTIVE_MIN_SAT = 40
ACTIVE_MIN_VAL = 70

# -------- board occupancy detection (gray + color) --------
OCCUPANCY_GRAY_MAX = 246

# 顶部生成区约束
SPAWN_TOP_MAX_ROW = 6


# =========================
# Data classes
# =========================
@dataclass
class Roi:
    x: int
    y: int
    w: int
    h: int
    monitor_index: int = 1

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def is_valid(self, min_size: int = 50) -> bool:
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
    x: int
    y: int
    cells: List[Tuple[int, int]]


# =========================
# Piece templates
# =========================
PIECES: Dict[str, List[List[Tuple[int, int]]]] = {
    "I": [
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
         ],
    "O": [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
    "T": [
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (1, 1), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 1)],
    ],
    "S": [
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
    ],
    "Z": [
        [(0, 0), (0, 1), (1, 1), (1, 2)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
    ],
    "J": [
        [(0, 0), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2), (1, 2)],
        [(0, 1), (1, 1), (2, 0), (2, 1)],
    ],
    "L": [
        [(0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 0), (0, 1), (0, 2), (1, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1)],
    ],
}


# =========================
# ROI / screen
# =========================
def load_roi() -> Optional[Roi]:
    if not ROI_FILE.exists():
        return None
    try:
        data = json.loads(ROI_FILE.read_text(encoding="utf-8"))
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


def save_roi(roi: Roi) -> None:
    ROI_FILE.write_text(
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


def clamp_roi(roi: Roi, frame_w: int, frame_h: int) -> Roi:
    x = max(0, min(roi.x, frame_w - 1))
    y = max(0, min(roi.y, frame_h - 1))
    w = max(1, min(roi.w, frame_w - x))
    h = max(1, min(roi.h, frame_h - y))
    return Roi(x=x, y=y, w=w, h=h, monitor_index=roi.monitor_index)


def hide_windows() -> None:
    for name in (WIN_HUD, WIN_PREVIEW, WIN_SELECT):
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


# =========================
# Grid splitting (linspace, stable)
# =========================
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


# =========================
# Active piece helpers
# =========================
def normalize_cells(cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted([(r - min_r, c - min_c) for r, c in cells])


def mask_to_cells(mask: np.ndarray) -> List[Tuple[int, int]]:
    ys, xs = np.where(mask > 0)
    return list(zip(ys.tolist(), xs.tolist()))


def detect_piece_and_rotation(
    cells: List[Tuple[int, int]],
) -> Tuple[Optional[str], Optional[int]]:
    # 允许 3 格 or 4 格都直接匹配模板
    if len(cells) not in (3, 4):
        return None, None

    norm = normalize_cells(cells)
    for piece, rots in PIECES.items():
        for rot_idx, shape in enumerate(rots):
            if len(shape) != len(cells):
                continue
            if norm == normalize_cells(shape):
                return piece, rot_idx

    return None, None

    # 3格：只做“顶部裁切(row=-1)”补全匹配（几何补I缺一格在 try_recover_* 里做）
    if len(cells) != 3:
        return None, None

    top_row = min(r for r, _ in cells)
    if top_row > 0:
        return None, None

    cell_set = set(cells)
    candidates = set()
    for r, c in cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if -1 <= rr < ROWS and 0 <= cc < COLS and (rr, cc) not in cell_set:
                candidates.add((rr, cc))

    matches: List[Tuple[str, int]] = []
    for add_cell in candidates:
        test_cells = cells + [add_cell]
        norm = normalize_cells(test_cells)
        for piece, rots in PIECES.items():
            for rot_idx, shape in enumerate(rots):
                if norm == normalize_cells(shape):
                    matches.append((piece, rot_idx))

    if len(matches) == 1:
        return matches[0][0], matches[0][1]

    return None, None


def try_recover_to_four_by_occupancy(
    cells: List[Tuple[int, int]],
    active_grid: np.ndarray,
    occupancy_grid: np.ndarray,
) -> List[Tuple[int, int]]:
    """3格时：优先从 occupancy 里补第4格（解决 HSV 漏检一格）。"""
    if len(cells) != 3:
        return cells

    cell_set = set(cells)
    candidates = set()

    for r, c in cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < ROWS and 0 <= cc < COLS and (rr, cc) not in cell_set:
                if occupancy_grid[rr, cc] == 1 and active_grid[rr, cc] == 0:
                    candidates.add((rr, cc))

    if not candidates:
        return cells

    matches: List[List[Tuple[int, int]]] = []
    for add_cell in candidates:
        test_cells = cells + [add_cell]
        piece, rot_idx = detect_piece_and_rotation(test_cells)  # 走4格匹配
        if piece is not None and rot_idx is not None:
            matches.append(test_cells)

    if len(matches) == 1:
        return matches[0]

    return cells


def try_recover_i_line_geometry(
    cells: List[Tuple[int, int]],
    occupancy_grid: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    终极兜底：专门修复你截图这种情况——I竖条永远只有3格。
    规则：
    - 如果3格严格同一列，且行号是连续的 (r, r+1, r+2)
      -> 直接补成4格：补在上端(rmin-1)或下端(rmax+1)
    - 优先补 occupancy=1 的那一端；都不是/都不行则选更“合理”的一端
    """
    if len(cells) != 3:
        return cells

    cols = [c for _, c in cells]
    if len(set(cols)) != 1:
        return cells

    col = cols[0]
    rows = sorted([r for r, _ in cells])

    # 必须是连续三格
    if not (rows[1] == rows[0] + 1 and rows[2] == rows[1] + 1):
        return cells

    r_min, r_max = rows[0], rows[2]
    up = (r_min - 1, col)
    down = (r_max + 1, col)

    up_ok = (0 <= up[0] < ROWS)
    down_ok = (0 <= down[0] < ROWS)

    # 候选：能放进棋盘的才算
    candidates: List[Tuple[int, int]] = []
    if up_ok:
        candidates.append(up)
    if down_ok:
        candidates.append(down)

    if not candidates:
        return cells

    # 优先选 occupancy=1 的那一端（说明那格确实“被占用”只是没被判成active）
    occ_scored = []
    for cand in candidates:
        occ_scored.append((int(occupancy_grid[cand[0], cand[1]]), cand))
    occ_scored.sort(reverse=True, key=lambda x: x[0])

    best_cand = occ_scored[0][1]
    test_cells = cells + [best_cand]

    # 验证一下：补完必须能匹配到 I（竖）或至少匹配某块（这里会是I）
    piece, rot_idx = detect_piece_and_rotation(test_cells)
    if piece is not None and rot_idx is not None:
        return test_cells

    # 如果 occupancy 引导的那端不匹配，再尝试另一端
    for _, cand in occ_scored[1:]:
        test_cells = cells + [cand]
        piece, rot_idx = detect_piece_and_rotation(test_cells)
        if piece is not None and rot_idx is not None:
            return test_cells

    # 若仍不匹配，保守返回原cells
    return cells


def choose_active_component(active_grid: np.ndarray, occupancy_grid: np.ndarray) -> ActiveInfo:
    mask = (active_grid > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)

    best_cells: List[Tuple[int, int]] = []
    best_mask = np.zeros_like(mask)
    best_score = None

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 3 or area > 5:
            continue

        comp = (labels == label).astype(np.uint8)
        cells = mask_to_cells(comp)
        if not cells:
            continue

        top_row = min(r for r, _ in cells)
        score = (top_row, abs(area - 4))

        if best_score is None or score < best_score:
            best_score = score
            best_cells = cells
            best_mask = comp

    if not best_cells:
        return ActiveInfo(
            mask=np.zeros_like(mask),
            piece=None,
            rot_idx=None,
            left_col=None,
            top_row=None,
            cells=[],
        )

    recovered = best_cells

    # mask：如果补到了4格，用新mask；否则保持原mask（row=-1不可画）
    if len(recovered) == 4:
        full_mask = np.zeros_like(mask)
        for rr, cc in recovered:
            full_mask[rr, cc] = 1
    else:
        full_mask = best_mask

    piece, rot_idx = detect_piece_and_rotation(recovered)
    left_col = min(c for _, c in recovered)
    top_row = min(r for r, _ in recovered)

    return ActiveInfo(
        mask=full_mask.astype(np.uint8),
        piece=piece,
        rot_idx=rot_idx,
        left_col=left_col,
        top_row=top_row,
        cells=recovered,
    )


# =========================
# Dellacherie
# =========================
def subtract_active(occupancy: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    board = occupancy.copy()
    board[active_mask > 0] = 0
    return board


def collides(board: np.ndarray, shape: List[Tuple[int, int]], x: int, y: int) -> bool:
    for dr, dc in shape:
        rr = y + dr
        cc = x + dc
        if rr < 0 or rr >= ROWS or cc < 0 or cc >= COLS:
            return True
        if board[rr, cc] == 1:
            return True
    return False


def drop_height(board: np.ndarray, shape: List[Tuple[int, int]], x: int) -> Optional[int]:
    max_c = max(c for _, c in shape)
    if x < 0 or x + max_c >= COLS:
        return None

    y = 0
    while True:
        if collides(board, shape, x, y + 1):
            if collides(board, shape, x, y):
                return None
            return y
        y += 1
        if y > ROWS:
            return None


def place_piece(board: np.ndarray, shape: List[Tuple[int, int]], x: int, y: int) -> np.ndarray:
    out = board.copy()
    for dr, dc in shape:
        out[y + dr, x + dc] = 1
    return out


def count_complete_lines(board: np.ndarray) -> int:
    return int(np.sum(np.all(board == 1, axis=1)))


def aggregate_height(board: np.ndarray) -> int:
    heights = []
    for c in range(COLS):
        col = board[:, c]
        filled = np.where(col == 1)[0]
        h = ROWS - int(filled[0]) if filled.size > 0 else 0
        heights.append(h)
    return int(sum(heights))


def holes(board: np.ndarray) -> int:
    total = 0
    for c in range(COLS):
        col = board[:, c]
        filled = np.where(col == 1)[0]
        if filled.size == 0:
            continue
        top = int(filled[0])
        total += int(np.sum(col[top:] == 0))
    return total


def bumpiness(board: np.ndarray) -> int:
    heights = []
    for c in range(COLS):
        col = board[:, c]
        filled = np.where(col == 1)[0]
        h = ROWS - int(filled[0]) if filled.size > 0 else 0
        heights.append(h)
    return int(sum(abs(heights[i] - heights[i + 1]) for i in range(COLS - 1)))


def dellacherie_score(board_after: np.ndarray) -> float:
    lines = count_complete_lines(board_after)
    agg = aggregate_height(board_after)
    hol = holes(board_after)
    bump = bumpiness(board_after)
    return 0.76 * lines - 0.51 * agg - 0.36 * hol - 0.18 * bump


def compute_best_target(board_locked: np.ndarray, piece: str) -> Optional[Target]:
    best: Optional[Target] = None
    best_score = -1e18

    for rot_idx, shape in enumerate(PIECES[piece]):
        max_c = max(c for _, c in shape)
        for x in range(0, (COLS - max_c)):
            y = drop_height(board_locked, shape, x)
            if y is None:
                continue

            board_after = place_piece(board_locked, shape, x, y)
            score = dellacherie_score(board_after)

            if score > best_score:
                best_score = score
                cells = [(y + dr, x + dc) for dr, dc in shape]
                best = Target(piece=piece, rot_idx=rot_idx, x=x, y=y, cells=cells)

    return best


# =========================
# Actions
# =========================
def build_plan(piece: str, cur_rot: int, cur_left: int, target: Target) -> List[str]:
    actions: List[str] = []

    rot_total = len(PIECES[piece])
    rot_count = (target.rot_idx - cur_rot) % rot_total
    for _ in range(rot_count):
        actions.append("ROTATE")

    dx = target.x - cur_left
    if dx < 0:
        actions.extend(["LEFT"] * abs(dx))
    elif dx > 0:
        actions.extend(["RIGHT"] * dx)

    actions.extend(["DROP"] * DROP_REPEAT)
    return actions


def press_key(key: str) -> None:
    if pdi is None:
        return
    pdi.keyDown(key)
    time.sleep(PRESS_DELAY)
    pdi.keyUp(key)


# =========================
# Drawing
# =========================
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
        cv2.line(out, (x, 0), (x, h - 1), (0, 255, 255), 1)
    for r in range(1, ROWS):
        y = int(y_edges[r])
        cv2.line(out, (0, y), (w - 1, y), (0, 255, 255), 1)

    ys, xs = np.where(active_mask > 0)
    for rr, cc in zip(ys.tolist(), xs.tolist()):
        x1, x2 = int(x_edges[cc]), int(x_edges[cc + 1])
        y1, y2 = int(y_edges[rr]), int(y_edges[rr + 1])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if target_cells:
        for rr, cc in target_cells:
            if 0 <= rr < ROWS and 0 <= cc < COLS:
                x1, x2 = int(x_edges[cc]), int(x_edges[cc + 1])
                y1, y2 = int(y_edges[rr]), int(y_edges[rr + 1])
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return out


def make_hud(
    roi: Optional[Roi],
    active: ActiveInfo,
    target: Optional[Target],
    plan: List[str],
) -> np.ndarray:
    hud = np.zeros((260, 980, 3), dtype=np.uint8)
    hud[:] = (30, 30, 30)

    lines = [
        "Task4 - Stable ROI->Grid + Spawn->Plan (single file)",
        "Keys: e=select ROI, s=save ROI, m=switch monitor, q=quit",
        f"Auto key: {'ON' if pdi is not None else 'OFF (pip install pydirectinput)'}",
        "",
        f"ROI: {roi}" if roi else "ROI: (not set)  -> press 'e' to select",
        f"Detected piece: {active.piece}",
        f"Detected rot: {active.rot_idx}",
        f"Detected left col: {active.left_col}",
        f"Active cells: {len(active.cells)}",
    ]

    if target:
        lines.append(f"Target: piece={target.piece} rot={target.rot_idx} x={target.x} y={target.y}")
    else:
        lines.append("Target: None")

    lines.append(f"Plan length: {len(plan)}")

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


# =========================
# Main
# =========================
def main() -> None:
    roi = load_roi()
    monitor_index = roi.monitor_index if roi else 1
    last_active_cells: List[Tuple[int, int]] = []
    occ_before_lock = None  # 锁定前的 occupancy 快照
    ignore_esc_until = 0.0

    target: Optional[Target] = None
    target_cells: Optional[List[Tuple[int, int]]] = None
    plan: List[str] = []

    spawn_candidate: Optional[Tuple[str, int, int]] = None
    spawn_stable_count = 0
    no_active_count = 0
    last_spawn_time = 0.0
    last_action_time = 0.0

    occ_hist: Deque[np.ndarray] = deque(maxlen=TEMPORAL_WINDOW)
    act_hist: Deque[np.ndarray] = deque(maxlen=TEMPORAL_WINDOW)

    while True:
        frame, screen_w, screen_h, monitor_count = grab_screen(monitor_index)

        if roi is None:
            hud = make_hud(
                roi=None,
                active=ActiveInfo(
                    mask=np.zeros((ROWS, COLS), dtype=np.uint8),
                    piece=None,
                    rot_idx=None,
                    left_col=None,
                    top_row=None,
                    cells=[],
                ),
                target=None,
                plan=[],
            )
            cv2.imshow(WIN_HUD, hud)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("e"):
                result = select_roi_on_clean_screenshot(monitor_index)
                if result is not None:
                    roi, _, _, _ = result
                    target = None
                    target_cells = None
                    plan = []
                    spawn_candidate = None
                    spawn_stable_count = 0
                    no_active_count = 0
                    occ_hist.clear()
                    act_hist.clear()
                    ignore_esc_until = time.time() + 0.6
            if key == ord("m"):
                monitor_index += 1
                if monitor_index > max(1, monitor_count):
                    monitor_index = 1
            continue

        monitor_index = roi.monitor_index
        roi = clamp_roi(roi, screen_w, screen_h)

        x, y, rw, rh = roi.as_tuple()
        roi_bgr = frame[y:y + rh, x:x + rw]

        occupancy_grid, active_color_grid = roi_to_masks(roi_bgr)

        occ_hist.append(occupancy_grid)
        act_hist.append(active_color_grid)
        occupancy_s = majority_vote(occ_hist)
        active_s = majority_vote(act_hist)

        active = choose_active_component(active_s, occupancy_s)
        if len(active.cells) in (3, 4):
            last_active_cells = list(active.cells)
            occ_before_lock = occupancy_s.copy()  # ← 新增：保存“锁定前堆叠”快照（不含active）
        board_locked = subtract_active(occupancy_s, active.mask)

        now = time.time()

        # ========= spawn detection =========
        if target is None and len(plan) == 0:
            if (
                active.piece is not None
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
                    best = compute_best_target(board_locked, active.piece)
                    if best is not None:
                        target = best
                        target_cells = best.cells
                        plan = build_plan(
                            piece=active.piece,
                            cur_rot=active.rot_idx,
                            cur_left=active.left_col,
                            target=best,
                        )
                        last_spawn_time = now
                    spawn_candidate = None
                    spawn_stable_count = 0
            else:
                spawn_candidate = None
                spawn_stable_count = 0

        # ========= execute plan =========
        # ========= execute plan =========
        if target is not None and plan and pdi is not None:
            # 当“方块已经对齐红框（目标列+目标旋转）”时，加速下落
            fast_drop = False
            if (
                    target_cells is not None
                    and active.piece == target.piece
                    and active.rot_idx == target.rot_idx
                    and active.left_col == target.x
                    and active.top_row is not None
            ):
                # 可选：越接近目标 y 越快（这里用 4 格内触发）
                if (target.y - active.top_row) <= 4:
                    fast_drop = True

            action_interval = 0.02 if fast_drop else ACTION_INTERVAL  # 0.02更快，你也可改成0.03

            if now - last_action_time >= action_interval:
                action = plan.pop(0)

                if action == "ROTATE":
                    press_key(KEY_ROTATE)
                elif action == "LEFT":
                    press_key(KEY_LEFT)
                elif action == "RIGHT":
                    press_key(KEY_RIGHT)
                elif action == "DROP":
                    press_key(KEY_DROP)

                last_action_time = now

        # ========= clear target (judge success by newly-occupied cells after lock) =========
        if target is not None:
            if len(active.cells) == 0:
                no_active_count += 1
            else:
                no_active_count = 0

            # “这块结束/锁定”判定：连续2帧看不到 active
            if no_active_count >= 2:
                success = False

                # 用“锁定后新增的 occupancy 格子”去比对 target_cells
                if target_cells is not None and occ_before_lock is not None:
                    new_cells = set()
                    for rr in range(ROWS):
                        for cc in range(COLS):
                            if occ_before_lock[rr, cc] == 0 and occupancy_s[rr, cc] == 1:
                                new_cells.add((rr, cc))

                    t = set(target_cells)

                    # 理想：新增格子与目标完全一致
                    if new_cells == t:
                        success = True
                    # 容错：偶发漏检/抖动时，允许新增是目标的子集（常见 3/4 格）
                    elif len(new_cells) in (3, 4) and new_cells.issubset(t):
                        success = True

                # 关键：无论成功与否，都必须放行下一块规划（否则会卡住导致“降落错误”）
                target = None
                plan = []
                no_active_count = 0
                occ_before_lock = None

                # 只有成功，红框才消失
                if success:
                    target_cells = None
        preview = draw_overlay(roi_bgr, active.mask, target_cells)
        cv2.imshow(WIN_PREVIEW, preview)

        hud = make_hud(roi, active, target, plan)
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
                plan = []
                spawn_candidate = None
                spawn_stable_count = 0
                no_active_count = 0
                occ_hist.clear()
                act_hist.clear()
                ignore_esc_until = time.time() + 0.6

        if key == ord("s"):
            save_roi(roi)

    hide_windows()


if __name__ == "__main__":
    main()