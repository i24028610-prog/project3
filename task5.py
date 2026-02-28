import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

# =========================
# Config
# =========================
CELL_SIZE = 28
BOARD_ROWS = 18
BOARD_COLS = 10

EXPORT_ROWS = 18
EXPORT_COLS = 14
SIDE_PADDING_COLS = 2  # 10 -> 14 shift (左右各+2 padding)

FPS = 60

# Frame-based control speeds (ms)
MOVE_STEP_MS_DEFAULT = 45
GRAVITY_STEP_MS_DEFAULT = 80

WINDOW_BG = (245, 247, 250)
GRID_LINE_COLOR = (210, 214, 220)
TEXT_COLOR = (35, 42, 52)
PANEL_BG = (232, 236, 242)
BORDER_COLOR = (60, 70, 85)

# Overlay colors
TARGET_OVERLAY_COLOR = (235, 80, 80)  # red-ish
GHOST_OVERLAY_COLOR = (70, 180, 90)  # green-ish
OVERLAY_STROKE = 3

TOP_MARGIN = 92
BOTTOM_MARGIN = 120
LEFT_MARGIN = 40
RIGHT_PANEL_WIDTH = 300  # slightly wider for overlay texts

WINDOW_WIDTH = LEFT_MARGIN + BOARD_COLS * CELL_SIZE + 40 + RIGHT_PANEL_WIDTH + 80
WINDOW_HEIGHT = TOP_MARGIN + BOARD_ROWS * CELL_SIZE + BOTTOM_MARGIN

PIECE_ORDER = ["I", "O", "T", "S", "Z", "J", "L"]
PIECE_TO_INDEX = {kind: idx for idx, kind in enumerate(PIECE_ORDER)}

# =========================
# Tetromino definitions
# =========================
SHAPES: Dict[str, List[List[Tuple[int, int]]]] = {
    "I": [
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 1), (1, 1), (2, 1), (3, 1)],
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


# =========================
# Data structures
# =========================
@dataclass
class Piece:
    kind: str
    rotation: int
    row: int
    col: int

    def cells(self) -> List[Tuple[int, int]]:
        rotation_count = len(SHAPES[self.kind])
        r_idx = self.rotation % rotation_count
        offsets = SHAPES[self.kind][r_idx]
        return [(self.row + r, self.col + c) for r, c in offsets]


# =========================
# Encoding helpers
# =========================
def flatten_7x4x14(tensor: List[List[List[int]]]) -> np.ndarray:
    arr = np.array(tensor, dtype=np.int8)
    return arr.reshape(-1)


def flatten_14x4(tensor: List[List[int]]) -> np.ndarray:
    arr = np.array(tensor, dtype=np.int8)
    return arr.reshape(-1)


def assert_one_hot(vec: np.ndarray, name: str) -> None:
    if vec.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={vec.shape}")
    if not np.all((vec == 0) | (vec == 1)):
        raise ValueError(f"{name} must be binary 0/1.")
    if int(vec.sum()) != 1:
        raise ValueError(f"{name} must be one-hot (sum==1), got sum={int(vec.sum())}.")


def decode_x_index(x_idx: int) -> Tuple[int, int, int]:
    kind = x_idx // (4 * 14)
    rem = x_idx % (4 * 14)
    rot4 = rem // 14
    col14 = rem % 14
    return kind, rot4, col14


def decode_y_index(y_idx: int) -> Tuple[int, int]:
    col14 = y_idx // 4
    rot4 = y_idx % 4
    return col14, rot4


def encode_x_index(kind_idx: int, rot4: int, col14: int) -> int:
    return kind_idx * (4 * 14) + rot4 * 14 + col14


def encode_y_index(col14: int, rot4: int) -> int:
    return col14 * 4 + rot4


# =========================
# Recorder
# =========================
class Task5Recorder:
    """
    保存两套数据：
    1) 整段保存：task5_frames_时间戳.npz + task5_frames_时间戳_meta.csv
    2) 逐帧保存：out_dir/per_frame_dir/frame_000001.npz ...
    """

    def __init__(
        self,
        out_dir: str,
        per_frame_save: bool = True,
        per_frame_dir: str = "frames_x_y",
    ) -> None:
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.per_frame_save = per_frame_save
        self.per_frame_root = os.path.join(self.out_dir, per_frame_dir)

        # 关键：无论如何都先创建目录（你要“必定生效”）
        if self.per_frame_save:
            os.makedirs(self.per_frame_root, exist_ok=True)

        # 用于整段保存
        self.x_list: List[np.ndarray] = []
        self.y_list: List[np.ndarray] = []
        self.meta_rows: List[dict] = []

    def add(self, x_392: np.ndarray, y_56: np.ndarray, meta: dict) -> None:
        assert_one_hot(x_392, "X(392)")
        assert_one_hot(y_56, "Y(56)")

        # 1) 内存累积（整段）
        self.x_list.append(x_392.astype(np.int8))
        self.y_list.append(y_56.astype(np.int8))
        self.meta_rows.append(meta)

        # 2) 逐帧保存（每帧一个文件）
        if self.per_frame_save:
            idx = int(meta.get("sample_index", self.size() - 1))
            fname = f"frame_{idx:06d}.npz"
            fpath = os.path.join(self.per_frame_root, fname)

            np.savez_compressed(
                fpath,
                X=x_392.astype(np.int8),
                Y=y_56.astype(np.int8),
                frame_index=int(meta.get("frame_index", -1)),
                piece_frame_index=int(meta.get("piece_frame_index", -1)),
                kind=str(meta.get("kind", "")),
                cur_rot4=int(meta.get("cur_rot4", -1)),
                cur_col14=int(meta.get("cur_col14", -1)),
                tgt_rot4=int(meta.get("tgt_rot4", -1)),
                tgt_col14=int(meta.get("tgt_col14", -1)),
            )

    def size(self) -> int:
        return len(self.x_list)

    def save(self) -> str:
        tag = time.strftime("%Y%m%d_%H%M%S")
        npz_path = os.path.join(self.out_dir, f"task5_frames_{tag}.npz")
        csv_path = os.path.join(self.out_dir, f"task5_frames_{tag}_meta.csv")

        x = np.stack(self.x_list, axis=0) if self.x_list else np.zeros((0, 392), dtype=np.int8)
        y = np.stack(self.y_list, axis=0) if self.y_list else np.zeros((0, 56), dtype=np.int8)

        np.savez_compressed(npz_path, X=x, Y=y)

        if self.meta_rows:
            fieldnames = list(self.meta_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(self.meta_rows)
        else:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                f.write("")

        return npz_path


# =========================
# Built-in Task3 (Game + Pierre Dellacherie)
# =========================
class BuiltInTetrisGame:
    def __init__(
        self,
        recorder: Task5Recorder,
        move_step_ms: int,
        gravity_step_ms: int,
    ) -> None:
        self.recorder = recorder
        self.move_step_ms = move_step_ms
        self.gravity_step_ms = gravity_step_ms

        self.board: List[List[int]] = [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]

        self.score = 0
        self.lines_cleared_total = 0
        self.pieces_placed = 0
        self.game_over = False

        # PD target (rotation, col in 10-col board coords)
        self.target_rotation: Optional[int] = None
        self.target_col: Optional[int] = None
        self.new_piece_spawned = True

        self.bag: List[str] = []
        self.current_piece: Optional[Piece] = None
        self.next_piece_kind = self._draw_from_bag()
        self.spawn_piece()

        self._move_timer_ms = 0
        self._gravity_timer_ms = 0

        self.frame_index = 0
        self.piece_frame_index = 0

    # ---- bag/spawn
    def _refill_bag(self) -> None:
        self.bag = list(SHAPES.keys())
        random.shuffle(self.bag)

    def _draw_from_bag(self) -> str:
        if not self.bag:
            self._refill_bag()
        return self.bag.pop()

    def spawn_piece(self) -> None:
        kind = self.next_piece_kind
        self.next_piece_kind = self._draw_from_bag()

        spawn_col = 3
        candidate = Piece(kind=kind, rotation=0, row=0, col=spawn_col)

        for offset in [0, -1, 1, -2, 2]:
            p = Piece(kind=kind, rotation=0, row=0, col=spawn_col + offset)
            if self.is_valid_position(p):
                self.current_piece = p
                self.target_rotation = None
                self.target_col = None
                self.new_piece_spawned = True
                self.piece_frame_index = 0
                return

        self.current_piece = candidate
        self.game_over = True

    # ---- valid/lock/clear
    def is_valid_position(self, piece: Piece) -> bool:
        for r, c in piece.cells():
            if c < 0 or c >= BOARD_COLS:
                return False
            if r < 0 or r >= BOARD_ROWS:
                return False
            if self.board[r][c] != 0:
                return False
        return True

    def lock_piece(self, piece: Piece) -> None:
        for r, c in piece.cells():
            if 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS:
                self.board[r][c] = SHAPE_ID[piece.kind]

        cleared = self.clear_lines()
        self.lines_cleared_total += cleared
        self.pieces_placed += 1

        score_table = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        self.score += score_table.get(cleared, 0)

        self.spawn_piece()

    def clear_lines(self) -> int:
        new_board = [row for row in self.board if any(cell == 0 for cell in row)]
        cleared = BOARD_ROWS - len(new_board)
        while len(new_board) < BOARD_ROWS:
            new_board.insert(0, [0 for _ in range(BOARD_COLS)])
        self.board = new_board
        return cleared

    # ---- task5 encoding getters
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
        return self.current_piece.col + SIDE_PADDING_COLS

    def get_target_rot4(self) -> Optional[int]:
        if self.target_rotation is None:
            return None
        return self.target_rotation % 4

    def get_target_col14(self) -> Optional[int]:
        if self.target_col is None:
            return None
        return self.target_col + SIDE_PADDING_COLS

    def encode_current_piece_7x4x14(self) -> Optional[List[List[List[int]]]]:
        k = self.get_current_piece_kind_index()
        rot4 = self.get_current_piece_rot4()
        col14 = self.get_current_piece_col14()
        if k is None or rot4 is None or col14 is None:
            return None
        if not (0 <= k < 7 and 0 <= rot4 < 4 and 0 <= col14 < EXPORT_COLS):
            return None

        t = [[[0 for _ in range(EXPORT_COLS)] for _ in range(4)] for _ in range(7)]
        t[k][rot4][col14] = 1
        return t

    def encode_target_14x4(self) -> Optional[List[List[int]]]:
        col14 = self.get_target_col14()
        rot4 = self.get_target_rot4()
        if col14 is None or rot4 is None:
            return None
        if not (0 <= col14 < EXPORT_COLS and 0 <= rot4 < 4):
            return None

        t = [[0 for _ in range(4)] for _ in range(EXPORT_COLS)]
        t[col14][rot4] = 1
        return t

    def sample_task5_pair_every_frame(self) -> None:
        if self.current_piece is None or self.target_rotation is None or self.target_col is None:
            return

        x_t = self.encode_current_piece_7x4x14()
        y_t = self.encode_target_14x4()
        if x_t is None or y_t is None:
            raise RuntimeError("Task5 encoding failed: invalid X/Y.")

        x = flatten_7x4x14(x_t)
        y = flatten_14x4(y_t)

        meta = {
            "sample_index": self.recorder.size(),
            "frame_index": self.frame_index,
            "piece_frame_index": self.piece_frame_index,
            "kind": self.current_piece.kind,
            "cur_rot": int(self.current_piece.rotation),
            "cur_rot4": int(self.get_current_piece_rot4()),
            "cur_row": int(self.current_piece.row),
            "cur_col10": int(self.current_piece.col),
            "cur_col14": int(self.get_current_piece_col14()),
            "tgt_rot": int(self.target_rotation),
            "tgt_rot4": int(self.get_target_rot4()),
            "tgt_col10": int(self.target_col),
            "tgt_col14": int(self.get_target_col14()),
        }
        self.recorder.add(x, y, meta)

    # ---- PD heuristic
    def choose_best_move(self) -> Optional[Tuple[int, int]]:
        if self.current_piece is None:
            return None

        kind = self.current_piece.kind
        best_score = -float("inf")
        best_move = None

        rotations = len(SHAPES[kind])
        checked = set()

        for rotation in range(rotations):
            offsets = SHAPES[kind][rotation]
            min_col = min(c for _, c in offsets)
            max_col = max(c for _, c in offsets)

            for col in range(-min_col, BOARD_COLS - max_col):
                key = (rotation, col, tuple(sorted(offsets)))
                if key in checked:
                    continue
                checked.add(key)

                final_row = self.hard_drop_row(kind, rotation, col)
                if final_row is None:
                    continue

                p = Piece(kind=kind, rotation=rotation, row=final_row, col=col)
                after, cleared = self.simulate_lock(self.board, p)
                feats = self.compute_pd_features(after, p, cleared)
                score = self.pd_score(feats)

                if score > best_score:
                    best_score = score
                    best_move = (rotation, col)

        return best_move

    def hard_drop_row(self, kind: str, rotation: int, col: int) -> Optional[int]:
        row = 0
        p = Piece(kind=kind, rotation=rotation, row=row, col=col)
        if not self.is_valid_position(p):
            return None
        while True:
            nxt = Piece(kind=kind, rotation=rotation, row=row + 1, col=col)
            if self.is_valid_position(nxt):
                row += 1
            else:
                return row

    @staticmethod
    def simulate_lock(board: List[List[int]], piece: Piece) -> Tuple[List[List[int]], int]:
        test = [row[:] for row in board]
        for r, c in piece.cells():
            if 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS:
                test[r][c] = SHAPE_ID[piece.kind]

        kept = [r for r in test if any(cell == 0 for cell in r)]
        cleared = BOARD_ROWS - len(kept)
        while len(kept) < BOARD_ROWS:
            kept.insert(0, [0 for _ in range(BOARD_COLS)])
        return kept, cleared

    @staticmethod
    def count_holes(board: List[List[int]]) -> int:
        holes = 0
        for c in range(BOARD_COLS):
            filled = False
            for r in range(BOARD_ROWS):
                if board[r][c] != 0:
                    filled = True
                elif filled:
                    holes += 1
        return holes

    @staticmethod
    def row_transitions(board: List[List[int]]) -> int:
        t = 0
        for r in range(BOARD_ROWS):
            prev = 1
            for c in range(BOARD_COLS):
                cur = 1 if board[r][c] != 0 else 0
                if cur != prev:
                    t += 1
                prev = cur
            if prev != 1:
                t += 1
        return t

    @staticmethod
    def column_transitions(board: List[List[int]]) -> int:
        t = 0
        for c in range(BOARD_COLS):
            prev = 1
            for r in range(BOARD_ROWS):
                cur = 1 if board[r][c] != 0 else 0
                if cur != prev:
                    t += 1
                prev = cur
            if prev != 1:
                t += 1
        return t

    @staticmethod
    def well_sums(board: List[List[int]]) -> int:
        total = 0
        for c in range(BOARD_COLS):
            depth = 0
            for r in range(BOARD_ROWS):
                if board[r][c] != 0:
                    depth = 0
                    continue
                left = (c == 0 or board[r][c - 1] != 0)
                right = (c == BOARD_COLS - 1 or board[r][c + 1] != 0)
                if left and right:
                    depth += 1
                    total += depth
                else:
                    depth = 0
        return total

    @staticmethod
    def landing_height(piece: Piece) -> float:
        rows = [r for r, _ in piece.cells()]
        avg_row = sum(rows) / len(rows)
        return BOARD_ROWS - avg_row

    def compute_pd_features(
        self,
        board_after: List[List[int]],
        piece: Piece,
        lines: int,
    ) -> Dict[str, float]:
        return {
            "landing_height": self.landing_height(piece),
            "rows_eliminated": float(lines),
            "row_transitions": float(self.row_transitions(board_after)),
            "col_transitions": float(self.column_transitions(board_after)),
            "holes": float(self.count_holes(board_after)),
            "well_sums": float(self.well_sums(board_after)),
        }

    @staticmethod
    def pd_score(f: Dict[str, float]) -> float:
        return (
            -4.500158825082766 * f["landing_height"]
            + 3.4181268101392694 * f["rows_eliminated"]
            - 3.2178882868487753 * f["row_transitions"]
            - 9.348695305445199 * f["col_transitions"]
            - 7.899265427351652 * f["holes"]
            - 3.3855972247263626 * f["well_sums"]
        )

    # ---- frame step
    def step_frame(self, dt_ms: int) -> None:
        if self.game_over or self.current_piece is None:
            return

        self.frame_index += 1
        self.piece_frame_index += 1

        if self.new_piece_spawned:
            move = self.choose_best_move()
            if move is None:
                self.game_over = True
                return
            self.target_rotation, self.target_col = move
            self.new_piece_spawned = False

        self.sample_task5_pair_every_frame()

        self._move_timer_ms += dt_ms
        self._gravity_timer_ms += dt_ms

        if self._move_timer_ms >= self.move_step_ms:
            self._move_timer_ms = 0
            self._apply_one_control_step_toward_target()

        if self._gravity_timer_ms >= self.gravity_step_ms:
            self._gravity_timer_ms = 0
            self._apply_gravity_one_row()

    def _apply_one_control_step_toward_target(self) -> None:
        if self.current_piece is None or self.target_rotation is None or self.target_col is None:
            return

        p = self.current_piece
        kind = p.kind
        rot_count = len(SHAPES[kind])

        cur_rot = p.rotation % rot_count
        tgt_rot = self.target_rotation % rot_count

        if cur_rot != tgt_rot:
            cand = Piece(
                kind=kind,
                rotation=(cur_rot + 1) % rot_count,
                row=p.row,
                col=p.col,
            )
            if self.is_valid_position(cand):
                self.current_piece = cand
                return

            cand2 = Piece(
                kind=kind,
                rotation=(cur_rot - 1) % rot_count,
                row=p.row,
                col=p.col,
            )
            if self.is_valid_position(cand2):
                self.current_piece = cand2
            return

        if p.col < self.target_col:
            cand = Piece(kind=kind, rotation=p.rotation, row=p.row, col=p.col + 1)
            if self.is_valid_position(cand):
                self.current_piece = cand
            return

        if p.col > self.target_col:
            cand = Piece(kind=kind, rotation=p.rotation, row=p.row, col=p.col - 1)
            if self.is_valid_position(cand):
                self.current_piece = cand

    def _apply_gravity_one_row(self) -> None:
        if self.current_piece is None:
            return

        p = self.current_piece
        down = Piece(kind=p.kind, rotation=p.rotation, row=p.row + 1, col=p.col)
        if self.is_valid_position(down):
            self.current_piece = down
            return

        self.lock_piece(p)

    # ---- console proof
    def print_current_xy_matrices(self) -> None:
        if self.recorder.size() == 0:
            print("[1] No samples yet.")
            return

        i = self.recorder.size() - 1
        x = self.recorder.x_list[i]
        y = self.recorder.y_list[i]

        x_idx = int(np.argmax(x))
        y_idx = int(np.argmax(y))

        kind_idx, rot4, col14 = decode_x_index(x_idx)
        tgt_col14, tgt_rot4 = decode_y_index(y_idx)

        x_tensor = x.reshape(7, 4, 14)
        y_tensor = y.reshape(14, 4)

        print("\n" + "=" * 70)
        print(f"[1] Sample index = {i}")
        print("X: 7x4x14 (flatten=392)")
        print(f"  kind={PIECE_ORDER[kind_idx]}(idx={kind_idx}), rot4={rot4}, col14={col14}")
        print("X slice (this kind) -> 4x14:")
        print(x_tensor[kind_idx])
        print("\nY: 14x4 (flatten=56)")
        print(f"  target col14={tgt_col14}, target rot4={tgt_rot4}")
        print("Y full 14x4:")
        print(y_tensor)
        print("=" * 70 + "\n")


# =========================
# Renderer (with overlay)
# =========================
class TetrisRenderer:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        pygame.font.init()
        self.font_small = pygame.font.SysFont("microsoftyahei", 18)
        self.font_medium = pygame.font.SysFont("microsoftyahei", 22, bold=True)
        self.font_title = pygame.font.SysFont("microsoftyahei", 24, bold=True)

        self.board_x = LEFT_MARGIN
        self.board_y = TOP_MARGIN

        self.panel_x = self.board_x + BOARD_COLS * CELL_SIZE + 30
        self.panel_y = TOP_MARGIN
        self.panel_w = RIGHT_PANEL_WIDTH
        self.panel_h = BOARD_ROWS * CELL_SIZE

    def draw(self, game: BuiltInTetrisGame, paused: bool) -> None:
        self.screen.fill(WINDOW_BG)
        self._draw_title(paused)
        self._draw_board(game)
        self._draw_panel(game)
        pygame.display.flip()

    def _draw_title(self, paused: bool) -> None:
        title = "Task 5 - Detection in Simulation (X=7x4x14, Y=14x4)"
        t = self.font_title.render(title, True, TEXT_COLOR)
        self.screen.blit(t, (LEFT_MARGIN, 10))

        sub1 = "1:输出矩阵   2:保存数据   3:暂停/继续"
        sub2 = "4:重置   ESC/关闭:自动保存退出"
        s1 = self.font_small.render(sub1, True, (80, 88, 98))
        s2 = self.font_small.render(sub2, True, (80, 88, 98))
        self.screen.blit(s1, (LEFT_MARGIN, 40))
        self.screen.blit(s2, (LEFT_MARGIN, 60))

        if paused:
            badge = self.font_medium.render("PAUSED", True, (200, 50, 50))
            self.screen.blit(badge, (WINDOW_WIDTH - 170, 18))

    def _draw_board(self, game: BuiltInTetrisGame) -> None:
        rect = pygame.Rect(
            self.board_x,
            self.board_y,
            BOARD_COLS * CELL_SIZE,
            BOARD_ROWS * CELL_SIZE,
        )
        pygame.draw.rect(self.screen, (255, 255, 255), rect, border_radius=6)
        pygame.draw.rect(self.screen, BORDER_COLOR, rect, width=2, border_radius=6)

        board = game.board
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                x = self.board_x + c * CELL_SIZE
                y = self.board_y + r * CELL_SIZE
                cell = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                v = board[r][c]
                if v != 0:
                    kind = self._id_to_kind(v)
                    color = SHAPE_COLORS.get(kind, (130, 130, 130))
                    inner = cell.inflate(-4, -4)
                    pygame.draw.rect(self.screen, color, inner, border_radius=4)
                pygame.draw.rect(self.screen, GRID_LINE_COLOR, cell, width=1)

        # draw current piece (solid)
        cur = game.current_piece
        if cur is not None and not game.game_over:
            for r, c in cur.cells():
                if 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS:
                    x = self.board_x + c * CELL_SIZE
                    y = self.board_y + r * CELL_SIZE
                    cell = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                    inner = cell.inflate(-4, -4)
                    color = SHAPE_COLORS[cur.kind]
                    pygame.draw.rect(self.screen, color, inner, border_radius=4)
                    pygame.draw.rect(self.screen, (255, 255, 255), inner, width=2, border_radius=4)

        self._draw_overlays(game)

    def _draw_overlays(self, game: BuiltInTetrisGame) -> None:
        cur = game.current_piece
        if cur is None or game.game_over:
            return

        # 1) Current ghost landing (green)
        cur_rot = cur.rotation
        cur_col = cur.col
        ghost_row = game.hard_drop_row(cur.kind, cur_rot, cur_col)
        if ghost_row is not None:
            ghost_piece = Piece(kind=cur.kind, rotation=cur_rot, row=ghost_row, col=cur_col)
            self._draw_piece_outline(ghost_piece, GHOST_OVERLAY_COLOR)

        # 2) Target landing (red)
        if game.target_rotation is not None and game.target_col is not None:
            tgt_rot = game.target_rotation
            tgt_col = game.target_col
            tgt_row = game.hard_drop_row(cur.kind, tgt_rot, tgt_col)
            if tgt_row is not None:
                tgt_piece = Piece(kind=cur.kind, rotation=tgt_rot, row=tgt_row, col=tgt_col)
                self._draw_piece_outline(tgt_piece, TARGET_OVERLAY_COLOR)

                # label INSIDE board top to avoid header overlap
                label = f"TARGET col10={tgt_col}"
                surf = self.font_small.render(label, True, TARGET_OVERLAY_COLOR)
                px = self.board_x + tgt_col * CELL_SIZE
                py = self.board_y + 6
                self.screen.blit(surf, (px, py))

    def _draw_piece_outline(self, piece: Piece, color: Tuple[int, int, int]) -> None:
        for r, c in piece.cells():
            if 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS:
                x = self.board_x + c * CELL_SIZE
                y = self.board_y + r * CELL_SIZE
                cell = pygame.Rect(x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4)
                pygame.draw.rect(self.screen, color, cell, width=OVERLAY_STROKE, border_radius=4)

    def _draw_panel(self, game: BuiltInTetrisGame) -> None:
        rect = pygame.Rect(self.panel_x, self.panel_y, self.panel_w, self.panel_h)
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(self.screen, BORDER_COLOR, rect, width=2, border_radius=8)

        y = self.panel_y + 14
        title = self.font_medium.render("Task5 采样状态 / Overlay", True, TEXT_COLOR)
        self.screen.blit(title, (self.panel_x + 14, y))
        y += 38

        samples = game.recorder.size()

        cur = game.current_piece
        if cur is not None:
            kind_idx = game.get_current_piece_kind_index()
            rot4 = game.get_current_piece_rot4()
            col14 = game.get_current_piece_col14()
        else:
            kind_idx, rot4, col14 = None, None, None

        x_idx_str = "-"
        if kind_idx is not None and rot4 is not None and col14 is not None:
            x_idx_str = str(encode_x_index(kind_idx, rot4, col14))

        tgt_col14 = game.get_target_col14()
        tgt_rot4 = game.get_target_rot4()
        y_idx_str = "-"
        if tgt_col14 is not None and tgt_rot4 is not None:
            y_idx_str = str(encode_y_index(tgt_col14, tgt_rot4))

        stats = [
            f"Samples: {samples}",
            f"Frame: {game.frame_index}",
            f"PieceFrame: {game.piece_frame_index}",
            f"Pieces locked: {game.pieces_placed}",
            f"Score: {game.score}",
            f"Lines: {game.lines_cleared_total}",
            "",
            "=== X (Current Piece) ===",
            f"kind: {cur.kind if cur else '-'}   kind_idx: {kind_idx if kind_idx is not None else '-'}",
            f"rot4: {rot4 if rot4 is not None else '-'}   col14: {col14 if col14 is not None else '-'}",
            f"X_idx(0..391): {x_idx_str}",
            "",
            "=== Y (PD Target) ===",
            f"tgt_rot4: {tgt_rot4 if tgt_rot4 is not None else '-'}",
            f"tgt_col14: {tgt_col14 if tgt_col14 is not None else '-'}",
            f"Y_idx(0..55): {y_idx_str}",
            "",
            "Overlay:",
            "Green = current landing (ghost)",
            "Red   = target landing",
            "",
            "Keys:",
            "1: Print matrices",
            "2: Save dataset",
            "3: Pause/Resume",
            "4: Reset",
            "ESC: Auto-save & Exit",
        ]

        for line in stats:
            surf = self.font_small.render(line, True, TEXT_COLOR)
            self.screen.blit(surf, (self.panel_x + 14, y))
            y += 22

    @staticmethod
    def _id_to_kind(v: int) -> str:
        for k, sid in SHAPE_ID.items():
            if v == sid:
                return k
        return "O"


# =========================
# CLI / main
# =========================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="out_task5")
    p.add_argument("--max_samples", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--move_step_ms", type=int, default=MOVE_STEP_MS_DEFAULT)
    p.add_argument("--gravity_step_ms", type=int, default=GRAVITY_STEP_MS_DEFAULT)

    # 逐帧保存：默认开启（你在 PyCharm 直接运行也会生效）
    # 如果你将来真要关闭：可以把 default 改回 False
    p.add_argument("--per_frame_save", action="store_true", default=True)
    p.add_argument("--per_frame_dir", type=str, default="frames_x_y")
    return p.parse_args()


def safe_save_and_exit(recorder: Task5Recorder) -> None:
    if recorder.size() > 0:
        path = recorder.save()
        print(f"[AUTO-SAVE] Saved dataset: {path}")
    else:
        print("[AUTO-SAVE] No samples collected, nothing to save.")
    pygame.quit()
    sys.exit()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Task 5 - Detection in Simulation (X/Y Sampling + Overlay)")
    clock = pygame.time.Clock()

    def build_game(rec: Task5Recorder) -> BuiltInTetrisGame:
        return BuiltInTetrisGame(rec, args.move_step_ms, args.gravity_step_ms)

    recorder = Task5Recorder(
        args.out_dir,
        per_frame_save=args.per_frame_save,
        per_frame_dir=args.per_frame_dir,
    )

    # 强制打印确认（你以后再也不会“以为开了但其实没开”）
    per_dir = os.path.join(args.out_dir, args.per_frame_dir)
    print("[PER-FRAME] enabled =", args.per_frame_save)
    print("[PER-FRAME] dir =", os.path.abspath(per_dir))
    if args.per_frame_save:
        os.makedirs(per_dir, exist_ok=True)

    game = build_game(recorder)
    renderer = TetrisRenderer(screen)

    paused = False

    print("CWD:", os.getcwd())
    print("OUT:", os.path.abspath(args.out_dir))
    print("Hotkeys: 1(print), 2(save), 3(pause), 4(reset), ESC(exit autosave)")

    while True:
        dt = clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                safe_save_and_exit(recorder)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    safe_save_and_exit(recorder)

                if event.key == pygame.K_1:
                    game.print_current_xy_matrices()

                if event.key == pygame.K_2:
                    if recorder.size() > 0:
                        path = recorder.save()
                        print(f"[MANUAL SAVE] Saved dataset: {path}")
                    else:
                        print("[MANUAL SAVE] No samples yet.")

                if event.key == pygame.K_3:
                    paused = not paused
                    print(f"[PAUSE] {paused}")

                if event.key == pygame.K_4:
                    recorder = Task5Recorder(
                        args.out_dir,
                        per_frame_save=args.per_frame_save,
                        per_frame_dir=args.per_frame_dir,
                    )
                    # 重置后也再打印一次，确保没丢
                    per_dir2 = os.path.join(args.out_dir, args.per_frame_dir)
                    print("[RESET] Game and recorder reset.")
                    print("[PER-FRAME] enabled =", args.per_frame_save)
                    print("[PER-FRAME] dir =", os.path.abspath(per_dir2))
                    if args.per_frame_save:
                        os.makedirs(per_dir2, exist_ok=True)

                    game = build_game(recorder)
                    paused = False

        if not paused and not game.game_over:
            game.step_frame(dt)

            if recorder.size() >= args.max_samples:
                path = recorder.save()
                print(f"[OK] Reached max_samples={args.max_samples}. Saved: {path}")
                pygame.quit()
                return

        if game.game_over:
            print("[GAME OVER] autosave then exit.")
            safe_save_and_exit(recorder)

        renderer.draw(game, paused)


if __name__ == "__main__":
    main()