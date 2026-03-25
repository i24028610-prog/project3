from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pygame


# ============================================================
# Config
# ============================================================
CELL_SIZE = 26
BOARD_ROWS = 18
BOARD_COLS = 14
ACTION_CLASSES = 56  # 4 * 14

FPS = 60
MOVE_STEP_MS_DEFAULT = 45
GRAVITY_STEP_MS_DEFAULT = 80

WINDOW_BG = (245, 247, 250)
GRID_LINE_COLOR = (210, 214, 220)
TEXT_COLOR = (35, 42, 52)
PANEL_BG = (232, 236, 242)
BORDER_COLOR = (60, 70, 85)

TARGET_OVERLAY_COLOR = (235, 80, 80)
GHOST_OVERLAY_COLOR = (70, 180, 90)
OVERLAY_STROKE = 3

TOP_MARGIN = 92
BOTTOM_MARGIN = 110
LEFT_MARGIN = 40
RIGHT_PANEL_WIDTH = 420

WINDOW_WIDTH = (
    LEFT_MARGIN + BOARD_COLS * CELL_SIZE + 40 + RIGHT_PANEL_WIDTH + 80
)
WINDOW_HEIGHT = TOP_MARGIN + BOARD_ROWS * CELL_SIZE + BOTTOM_MARGIN


# ============================================================
# Tetromino definitions
# 注意：I 型方块固定为 3 格，不是标准 4 格
# ============================================================
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
KIND_TO_INDEX: Dict[str, int] = {
    "I": 0,
    "O": 1,
    "T": 2,
    "S": 3,
    "Z": 4,
    "J": 5,
    "L": 6,
}
INDEX_TO_KIND: Dict[int, str] = {v: k for k, v in KIND_TO_INDEX.items()}


# ============================================================
# Data structures
# ============================================================
@dataclass
class Piece:
    """表示一个当前方块或某个候选落点方块。"""

    kind: str
    rotation: int
    row: int
    col: int

    def cells(self) -> List[Tuple[int, int]]:
        """返回这个方块在棋盘上的所有格子坐标。"""
        rot_count = len(SHAPES[self.kind])
        r_idx = self.rotation % rot_count
        offsets = SHAPES[self.kind][r_idx]
        return [(self.row + r, self.col + c) for r, c in offsets]


@dataclass
class MoveEval:
    """保存一个候选动作落地后的评估结果。"""

    rotation: int
    anchor_col: int
    left_col14: int
    action_idx: int
    final_piece: Piece
    board_after_lock: List[List[int]]
    board_after_clear: List[List[int]]
    lines_cleared: int
    holes_after: int
    blockades_after: int
    aggregate_height_after: int
    bumpiness_after: int
    max_height_after: int
    near_full_rows_before: int
    near_full_rows_after: int
    row_transitions_after: int
    col_transitions_after: int
    well_sums_after: int
    landing_height: float
    teacher_score: float


@dataclass
class DecisionSample:
    """保存一个样本，既包含任务5要求的编码，也保留给 task6 用的原有字段。"""

    board_before: List[List[int]]
    active_mask: List[List[int]]
    board_after_lock: List[List[int]]
    board_after_clear: List[List[int]]

    legal_mask_4x14: List[List[int]]
    action_score_4x14: List[List[float]]
    action_prob_4x14: List[List[float]]

    current_state_7x4x14_rows: List[List[int]]
    target_14x4: List[List[int]]
    legal_target_14x4: List[List[int]]
    score_target_14x4: List[List[float]]
    prob_target_14x4: List[List[float]]

    cur_kind: str
    kind_id: int
    cur_rot4: int
    cur_col14: int
    current_state_id: int
    tgt_rot4: int
    tgt_col14: int
    target_action: int

    lines_cleared: int
    holes_after: int
    blockades_after: int
    aggregate_height_after: int
    bumpiness_after: int
    max_height_after: int
    near_full_rows_before: int
    near_full_rows_after: int
    row_transitions_after: int
    col_transitions_after: int
    well_sums_after: int
    landing_height: float
    teacher_score: float
    priority_class: str


# ============================================================
# Utils
# ============================================================
def leftmost_col(piece: Piece) -> int:
    """返回这个方块当前实际占用格子的最左列。"""
    return min(c for _, c in piece.cells())


def empty_mask(
    rows: int = BOARD_ROWS,
    cols: int = BOARD_COLS,
) -> List[List[int]]:
    """创建一个全 0 矩阵。"""
    return [[0 for _ in range(cols)] for _ in range(rows)]


def build_active_mask(piece: Piece) -> List[List[int]]:
    """把当前活动块变成 18x14 的掩码矩阵。"""
    mask = empty_mask()
    for r, c in piece.cells():
        if 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS:
            mask[r][c] = 1
    return mask


def softmax_from_scores(
    scores: List[float],
    temperature: float,
) -> List[float]:
    """把合法动作分数转成 softmax 概率，用作软标签。"""
    if not scores:
        return []

    temp = max(1e-6, temperature)
    max_score = max(scores)
    exps = [math.exp((score - max_score) / temp) for score in scores]
    total = sum(exps)

    if total <= 0:
        return [1.0 / len(scores) for _ in scores]

    return [value / total for value in exps]


def encode_current_state_7x4x14(
    kind_id: int,
    cur_rot4: int,
    cur_col14: int,
) -> List[List[int]]:
    """
    把当前块状态编码成 7x4x14。

    为了方便写 CSV，这里展平成 28 行 x 14 列：
    第 0~3 行表示 I 块的 4 个旋转，
    第 4~7 行表示 O 块的 4 个旋转，
    ...
    """
    out = [
        [[0 for _ in range(BOARD_COLS)] for _ in range(4)]
        for _ in range(7)
    ]
    if (
        0 <= kind_id < 7
        and 0 <= cur_rot4 < 4
        and 0 <= cur_col14 < BOARD_COLS
    ):
        out[kind_id][cur_rot4][cur_col14] = 1

    rows_28x14: List[List[int]] = []
    for kind_idx in range(7):
        for rot_idx in range(4):
            rows_28x14.append(out[kind_idx][rot_idx][:])
    return rows_28x14


def encode_target_14x4(
    tgt_col14: int,
    tgt_rot4: int,
) -> List[List[int]]:
    """把老师目标编码成 14x4，行是列，列是旋转。"""
    out = [[0 for _ in range(4)] for _ in range(BOARD_COLS)]
    if 0 <= tgt_col14 < BOARD_COLS and 0 <= tgt_rot4 < 4:
        out[tgt_col14][tgt_rot4] = 1
    return out


def transpose_4x14_to_14x4_int(
    matrix_4x14: List[List[int]],
) -> List[List[int]]:
    """把 4x14 的整型矩阵转成 14x4。"""
    out = [[0 for _ in range(4)] for _ in range(BOARD_COLS)]
    for rot_idx in range(4):
        for col_idx in range(BOARD_COLS):
            out[col_idx][rot_idx] = int(matrix_4x14[rot_idx][col_idx])
    return out


def transpose_4x14_to_14x4_float(
    matrix_4x14: List[List[float]],
) -> List[List[float]]:
    """把 4x14 的浮点矩阵转成 14x4。"""
    out = [[0.0 for _ in range(4)] for _ in range(BOARD_COLS)]
    for rot_idx in range(4):
        for col_idx in range(BOARD_COLS):
            out[col_idx][rot_idx] = float(matrix_4x14[rot_idx][col_idx])
    return out


# ============================================================
# Recorder
# ============================================================
class BlockCSVRecorder:
    """
    把每个新方块的一次决策写入一个大 CSV。

    这版程序同时保存：
    1) 任务5要求的 CURRENT_STATE_7x4x14 / TARGET_14x4
    2) 后续 task6 还能继续使用的原有 section
    """

    def __init__(self, out_dir: str) -> None:
        """初始化输出目录与 CSV 文件。"""
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.run_tag = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(
            self.out_dir,
            f"task5_blocks_{self.run_tag}.csv",
        )

        self._fp = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fp)
        self.count = 0

    def size(self) -> int:
        """返回当前已写入的样本数。"""
        return self.count

    @staticmethod
    def _pad_cols(row: List[object], cols: int) -> List[object]:
        """把一行补到固定列数，便于写入规则的 CSV。"""
        row = list(row)
        if len(row) < cols:
            row.extend([""] * (cols - len(row)))
        return row[:cols]

    def _write_matrix_section(
        self,
        title: str,
        matrix: List[List[object]],
        cols: int,
    ) -> None:
        """写一个矩阵 section。"""
        self._writer.writerow(self._pad_cols([title], cols))
        for row in matrix:
            self._writer.writerow(self._pad_cols(row, cols))
        self._writer.writerow([""] * cols)

    def add_block(self, sample: DecisionSample) -> None:
        """写入一个完整样本。每个方块只写一次。"""
        self._write_matrix_section(
            "CURRENT_STATE_7x4x14",
            sample.current_state_7x4x14_rows,
            BOARD_COLS,
        )
        self._write_matrix_section("TARGET_14x4", sample.target_14x4, 4)
        self._write_matrix_section(
            "LEGAL_TARGET_14x4",
            sample.legal_target_14x4,
            4,
        )
        self._write_matrix_section(
            "SCORE_TARGET_14x4",
            sample.score_target_14x4,
            4,
        )
        self._write_matrix_section(
            "PROB_TARGET_14x4",
            sample.prob_target_14x4,
            4,
        )

        self._write_matrix_section(
            "BOARD_BEFORE",
            sample.board_before,
            BOARD_COLS,
        )
        self._write_matrix_section(
            "ACTIVE_MASK",
            sample.active_mask,
            BOARD_COLS,
        )
        self._write_matrix_section(
            "BOARD_AFTER_LOCK",
            sample.board_after_lock,
            BOARD_COLS,
        )
        self._write_matrix_section(
            "BOARD_AFTER_CLEAR",
            sample.board_after_clear,
            BOARD_COLS,
        )
        self._write_matrix_section(
            "LEGAL_MASK",
            sample.legal_mask_4x14,
            BOARD_COLS,
        )
        self._write_matrix_section(
            "ACTION_SCORE",
            sample.action_score_4x14,
            BOARD_COLS,
        )
        self._write_matrix_section(
            "ACTION_PROB",
            sample.action_prob_4x14,
            BOARD_COLS,
        )

        meta_header = [
            "META",
            "kind",
            "kind_id",
            "cur_rot4",
            "cur_col14",
            "current_state_id",
            "tgt_rot4",
            "tgt_col14",
            "target_action",
            "lines_cleared",
            "holes_after",
            "blockades_after",
            "aggregate_height_after",
            "bumpiness_after",
            "max_height_after",
            "near_full_rows_before",
            "near_full_rows_after",
            "row_transitions_after",
            "col_transitions_after",
            "well_sums_after",
            "landing_height",
            "teacher_score",
            "priority_class",
            "sample_no",
        ]
        self._writer.writerow(meta_header)

        meta_value = [
            "META",
            sample.cur_kind,
            sample.kind_id,
            sample.cur_rot4,
            sample.cur_col14,
            sample.current_state_id,
            sample.tgt_rot4,
            sample.tgt_col14,
            sample.target_action,
            sample.lines_cleared,
            sample.holes_after,
            sample.blockades_after,
            sample.aggregate_height_after,
            sample.bumpiness_after,
            sample.max_height_after,
            sample.near_full_rows_before,
            sample.near_full_rows_after,
            sample.row_transitions_after,
            sample.col_transitions_after,
            sample.well_sums_after,
            f"{sample.landing_height:.4f}",
            f"{sample.teacher_score:.6f}",
            sample.priority_class,
            self.count,
        ]
        self._writer.writerow(meta_value)
        self._writer.writerow([])

        self.count += 1
        if self.count % 200 == 0:
            self._fp.flush()

    def close(self) -> None:
        """关闭文件并保存。"""
        try:
            self._fp.flush()
            self._fp.close()
        except Exception:
            pass


# ============================================================
# Game + PD teacher
# ============================================================
class BuiltInTetrisGame:
    """
    仿真俄罗斯方块环境。

    当前版本里，任务5的职责是：
    1) 在新方块生成时提取当前状态
    2) 提取当前 PD 规则给出的目标
    3) 把它们整理成 7x4x14 和 14x4
    """

    def __init__(
        self,
        recorder: BlockCSVRecorder,
        move_step_ms: int,
        gravity_step_ms: int,
        fast_collect: bool,
        softmax_temp: float,
    ) -> None:
        """初始化棋盘、计数器和当前局状态。"""
        self.recorder = recorder
        self.move_step_ms = move_step_ms
        self.gravity_step_ms = gravity_step_ms
        self.fast_collect = fast_collect
        self.softmax_temp = softmax_temp

        self.board: List[List[int]] = [
            [0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)
        ]

        self.score = 0
        self.lines_cleared_total = 0
        self.pieces_placed = 0
        self.game_over = False

        self.target_rotation: Optional[int] = None
        self.target_anchor_col: Optional[int] = None
        self.last_best_eval: Optional[MoveEval] = None
        self.last_priority_class: str = "normal"
        self.last_action_score_4x14: Optional[List[List[float]]] = None
        self.last_action_prob_4x14: Optional[List[List[float]]] = None
        self.last_legal_mask_4x14: Optional[List[List[int]]] = None

        self.new_piece_spawned = True
        self.sample_saved_for_current_piece = False

        self.bag: List[str] = []
        self.current_piece: Optional[Piece] = None
        self.next_piece_kind = self._draw_from_bag()
        self.spawn_piece()

        self._move_timer_ms = 0
        self._gravity_timer_ms = 0

    # ---- bag/spawn
    def _refill_bag(self) -> None:
        """使用 7-bag 重新生成方块袋。"""
        self.bag = list(SHAPES.keys())
        random.shuffle(self.bag)

    def _draw_from_bag(self) -> str:
        """从方块袋中抽取下一个方块类型。"""
        if not self.bag:
            self._refill_bag()
        return self.bag.pop()

    def spawn_piece(self) -> None:
        """
        生成新方块。

        注意：任务5采样是在“新方块出生瞬间”做一次。
        """
        kind = self.next_piece_kind
        self.next_piece_kind = self._draw_from_bag()

        spawn_col = BOARD_COLS // 2 - 2
        for offset in [0, -1, 1, -2, 2, -3, 3]:
            piece = Piece(
                kind=kind,
                rotation=0,
                row=0,
                col=spawn_col + offset,
            )
            if self.is_valid_position(piece):
                self.current_piece = piece
                self.target_rotation = None
                self.target_anchor_col = None
                self.last_best_eval = None
                self.last_priority_class = "normal"
                self.last_action_score_4x14 = None
                self.last_action_prob_4x14 = None
                self.last_legal_mask_4x14 = None
                self.new_piece_spawned = True
                self.sample_saved_for_current_piece = False
                return

        self.current_piece = Piece(
            kind=kind,
            rotation=0,
            row=0,
            col=spawn_col,
        )
        self.game_over = True

    # ---- collision
    def is_valid_position(self, piece: Piece) -> bool:
        """判断一个方块在当前棋盘是否合法。"""
        for row_idx, col_idx in piece.cells():
            if col_idx < 0 or col_idx >= BOARD_COLS:
                return False
            if row_idx < 0 or row_idx >= BOARD_ROWS:
                return False
            if self.board[row_idx][col_idx] != 0:
                return False
        return True

    @staticmethod
    def is_valid_position_on_board(
        board: List[List[int]],
        piece: Piece,
    ) -> bool:
        """判断一个方块在指定棋盘上是否合法。"""
        for row_idx, col_idx in piece.cells():
            if col_idx < 0 or col_idx >= BOARD_COLS:
                return False
            if row_idx < 0 or row_idx >= BOARD_ROWS:
                return False
            if board[row_idx][col_idx] != 0:
                return False
        return True

    # ---- lock/clear
    @staticmethod
    def clear_lines_from_board(
        board: List[List[int]],
    ) -> Tuple[List[List[int]], int]:
        """对某个棋盘执行消行，返回新棋盘和消掉的行数。"""
        new_board = [row[:] for row in board if any(cell == 0 for cell in row)]
        cleared = BOARD_ROWS - len(new_board)
        while len(new_board) < BOARD_ROWS:
            new_board.insert(0, [0 for _ in range(BOARD_COLS)])
        return new_board, cleared

    def clear_lines(self) -> int:
        """对当前真实棋盘执行消行。"""
        self.board, cleared = self.clear_lines_from_board(self.board)
        return cleared

    def lock_piece(self, piece: Piece) -> None:
        """把当前方块锁到棋盘，然后消行并生成下一个方块。"""
        for row_idx, col_idx in piece.cells():
            if 0 <= row_idx < BOARD_ROWS and 0 <= col_idx < BOARD_COLS:
                self.board[row_idx][col_idx] = SHAPE_ID[piece.kind]

        cleared = self.clear_lines()
        self.lines_cleared_total += cleared
        self.pieces_placed += 1

        score_table = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        self.score += score_table.get(cleared, 0)

        self.spawn_piece()

    # ---- board features
    @staticmethod
    def count_holes(board: List[List[int]]) -> int:
        """统计洞的数量。"""
        holes = 0
        for col_idx in range(BOARD_COLS):
            seen_filled = False
            for row_idx in range(BOARD_ROWS):
                if board[row_idx][col_idx] != 0:
                    seen_filled = True
                elif seen_filled:
                    holes += 1
        return holes

    @staticmethod
    def count_blockades(board: List[List[int]]) -> int:
        """统计洞上方的阻挡块数量。"""
        blockades = 0
        for col_idx in range(BOARD_COLS):
            holes_below = 0
            for row_idx in range(BOARD_ROWS - 1, -1, -1):
                if board[row_idx][col_idx] == 0:
                    holes_below += 1
                elif holes_below > 0:
                    blockades += 1
        return blockades

    @staticmethod
    def column_heights(board: List[List[int]]) -> List[int]:
        """返回每一列的高度。"""
        heights: List[int] = []
        for col_idx in range(BOARD_COLS):
            height = 0
            for row_idx in range(BOARD_ROWS):
                if board[row_idx][col_idx] != 0:
                    height = BOARD_ROWS - row_idx
                    break
            heights.append(height)
        return heights

    @classmethod
    def aggregate_height(cls, board: List[List[int]]) -> int:
        """返回所有列高度总和。"""
        return sum(cls.column_heights(board))

    @classmethod
    def bumpiness(cls, board: List[List[int]]) -> int:
        """返回相邻列高度差总和。"""
        heights = cls.column_heights(board)
        return sum(
            abs(heights[idx] - heights[idx + 1])
            for idx in range(BOARD_COLS - 1)
        )

    @classmethod
    def max_height(cls, board: List[List[int]]) -> int:
        """返回当前棋盘最大列高。"""
        heights = cls.column_heights(board)
        return max(heights) if heights else 0

    @staticmethod
    def near_full_rows(
        board: List[List[int]],
        min_filled: int = BOARD_COLS - 2,
    ) -> int:
        """统计接近满行但还没满的行数。"""
        count = 0
        for row_idx in range(BOARD_ROWS):
            filled = sum(1 for value in board[row_idx] if value != 0)
            if min_filled <= filled < BOARD_COLS:
                count += 1
        return count

    @staticmethod
    def row_transitions(board: List[List[int]]) -> int:
        """统计行方向 transitions。"""
        transitions = 0
        for row_idx in range(BOARD_ROWS):
            prev = 1
            for col_idx in range(BOARD_COLS):
                cur = 1 if board[row_idx][col_idx] != 0 else 0
                if cur != prev:
                    transitions += 1
                prev = cur
            if prev != 1:
                transitions += 1
        return transitions

    @staticmethod
    def column_transitions(board: List[List[int]]) -> int:
        """统计列方向 transitions。"""
        transitions = 0
        for col_idx in range(BOARD_COLS):
            prev = 1
            for row_idx in range(BOARD_ROWS):
                cur = 1 if board[row_idx][col_idx] != 0 else 0
                if cur != prev:
                    transitions += 1
                prev = cur
            if prev != 1:
                transitions += 1
        return transitions

    @staticmethod
    def well_sums(board: List[List[int]]) -> int:
        """统计井深总和。"""
        total = 0
        for col_idx in range(BOARD_COLS):
            depth = 0
            for row_idx in range(BOARD_ROWS):
                if board[row_idx][col_idx] != 0:
                    depth = 0
                    continue

                left_wall = (
                    col_idx == 0 or board[row_idx][col_idx - 1] != 0
                )
                right_wall = (
                    col_idx == BOARD_COLS - 1
                    or board[row_idx][col_idx + 1] != 0
                )
                if left_wall and right_wall:
                    depth += 1
                    total += depth
                else:
                    depth = 0
        return total

    @staticmethod
    def landing_height(piece: Piece) -> float:
        """计算当前落点方块的平均落地高度。"""
        rows = [row_idx for row_idx, _ in piece.cells()]
        avg_row = sum(rows) / len(rows)
        return BOARD_ROWS - avg_row

    # ---- simulation / PD planning
    @classmethod
    def hard_drop_row_on_board(
        cls,
        board: List[List[int]],
        kind: str,
        rotation: int,
        col: int,
    ) -> Optional[int]:
        """在指定棋盘上模拟 hard drop，返回最终落地行。"""
        row = 0
        piece = Piece(kind=kind, rotation=rotation, row=row, col=col)
        if not cls.is_valid_position_on_board(board, piece):
            return None

        while True:
            next_piece = Piece(
                kind=kind,
                rotation=rotation,
                row=row + 1,
                col=col,
            )
            if cls.is_valid_position_on_board(board, next_piece):
                row += 1
            else:
                return row

    @classmethod
    def simulate_lock(
        cls,
        board: List[List[int]],
        piece: Piece,
    ) -> Tuple[List[List[int]], List[List[int]], int]:
        """把一个候选方块锁到某个棋盘上，得到锁定后棋盘、消行后棋盘与消行数。"""
        board_after_lock = [row[:] for row in board]
        for row_idx, col_idx in piece.cells():
            if 0 <= row_idx < BOARD_ROWS and 0 <= col_idx < BOARD_COLS:
                board_after_lock[row_idx][col_idx] = SHAPE_ID[piece.kind]
        board_after_clear, cleared = cls.clear_lines_from_board(board_after_lock)
        return board_after_lock, board_after_clear, cleared

    @classmethod
    def pd_total_score(
        cls,
        lines_cleared: int,
        holes_after: int,
        blockades_after: int,
        aggregate_height_after: int,
        bumpiness_after: int,
        max_height_after: int,
        row_transitions_after: int,
        col_transitions_after: int,
        well_sums_after: int,
        landing_height: float,
    ) -> float:
        """
        使用一套 PD 风格评分选老师目标。

        这里仍然是当前仿真中的 PD 目标，不是神经网络目标。
        """
        return (
            220.0 * float(lines_cleared)
            - 11.0 * float(holes_after)
            - 4.0 * float(blockades_after)
            - 0.70 * float(aggregate_height_after)
            - 1.00 * float(bumpiness_after)
            - 2.20 * float(max_height_after)
            - 1.60 * float(row_transitions_after)
            - 1.90 * float(col_transitions_after)
            - 0.90 * float(well_sums_after)
            - 3.50 * float(landing_height)
        )

    def evaluate_all_moves(
        self,
    ) -> Optional[
        Tuple[
            MoveEval,
            List[List[int]],
            List[List[float]],
            List[List[float]],
        ]
    ]:
        """
        枚举当前方块所有合法落点，找到当前 PD 规则的目标动作。

        这里输出的目标会被 task5 提取成 14x4。
        """
        if self.current_piece is None:
            return None

        kind = self.current_piece.kind
        board_before = [row[:] for row in self.board]
        near_before = self.near_full_rows(board_before)

        legal_mask_4x14 = [
            [0 for _ in range(BOARD_COLS)] for _ in range(4)
        ]
        action_score_4x14 = [
            [0.0 for _ in range(BOARD_COLS)] for _ in range(4)
        ]
        action_prob_4x14 = [
            [0.0 for _ in range(BOARD_COLS)] for _ in range(4)
        ]

        best_eval: Optional[MoveEval] = None
        legal_actions: List[int] = []
        legal_scores: List[float] = []
        checked = set()

        rotations = len(SHAPES[kind])
        for rotation in range(rotations):
            offsets = SHAPES[kind][rotation]
            min_col = min(col_idx for _, col_idx in offsets)
            max_col = max(col_idx for _, col_idx in offsets)

            for anchor_col in range(-min_col, BOARD_COLS - max_col):
                key = (rotation, anchor_col, tuple(sorted(offsets)))
                if key in checked:
                    continue
                checked.add(key)

                final_row = self.hard_drop_row_on_board(
                    board_before,
                    kind,
                    rotation,
                    anchor_col,
                )
                if final_row is None:
                    continue

                final_piece = Piece(
                    kind=kind,
                    rotation=rotation,
                    row=final_row,
                    col=anchor_col,
                )
                left_col14 = leftmost_col(final_piece)
                if left_col14 < 0 or left_col14 >= BOARD_COLS:
                    continue

                action_idx = (rotation % 4) * BOARD_COLS + left_col14
                board_after_lock, board_after_clear, lines_cleared = (
                    self.simulate_lock(board_before, final_piece)
                )

                holes_after = self.count_holes(board_after_clear)
                blockades_after = self.count_blockades(board_after_clear)
                aggregate_height_after = self.aggregate_height(
                    board_after_clear
                )
                bumpiness_after = self.bumpiness(board_after_clear)
                max_height_after = self.max_height(board_after_clear)
                near_full_rows_after = self.near_full_rows(board_after_clear)
                row_transitions_after = self.row_transitions(
                    board_after_clear
                )
                col_transitions_after = self.column_transitions(
                    board_after_clear
                )
                well_sums_after = self.well_sums(board_after_clear)
                landing_h = self.landing_height(final_piece)

                score = self.pd_total_score(
                    lines_cleared=lines_cleared,
                    holes_after=holes_after,
                    blockades_after=blockades_after,
                    aggregate_height_after=aggregate_height_after,
                    bumpiness_after=bumpiness_after,
                    max_height_after=max_height_after,
                    row_transitions_after=row_transitions_after,
                    col_transitions_after=col_transitions_after,
                    well_sums_after=well_sums_after,
                    landing_height=landing_h,
                )

                legal_mask_4x14[rotation % 4][left_col14] = 1
                action_score_4x14[rotation % 4][left_col14] = score
                legal_actions.append(action_idx)
                legal_scores.append(score)

                cur_eval = MoveEval(
                    rotation=rotation,
                    anchor_col=anchor_col,
                    left_col14=left_col14,
                    action_idx=action_idx,
                    final_piece=final_piece,
                    board_after_lock=board_after_lock,
                    board_after_clear=board_after_clear,
                    lines_cleared=lines_cleared,
                    holes_after=holes_after,
                    blockades_after=blockades_after,
                    aggregate_height_after=aggregate_height_after,
                    bumpiness_after=bumpiness_after,
                    max_height_after=max_height_after,
                    near_full_rows_before=near_before,
                    near_full_rows_after=near_full_rows_after,
                    row_transitions_after=row_transitions_after,
                    col_transitions_after=col_transitions_after,
                    well_sums_after=well_sums_after,
                    landing_height=landing_h,
                    teacher_score=score,
                )

                if (
                    best_eval is None
                    or cur_eval.teacher_score > best_eval.teacher_score
                ):
                    best_eval = cur_eval

        if best_eval is None:
            return None

        probs = softmax_from_scores(legal_scores, self.softmax_temp)
        for action_idx, prob in zip(legal_actions, probs):
            rot_idx = action_idx // BOARD_COLS
            col_idx = action_idx % BOARD_COLS
            action_prob_4x14[rot_idx][col_idx] = prob

        return (
            best_eval,
            legal_mask_4x14,
            action_score_4x14,
            action_prob_4x14,
        )

    # ---- sampling
    @staticmethod
    def classify_priority(best_eval: MoveEval) -> str:
        """给样本分一个类型标签，后面可用于分析或训练加权。"""
        if best_eval.lines_cleared > 0:
            return "clear"
        if (
            best_eval.near_full_rows_before > 0
            or best_eval.near_full_rows_after > 0
        ):
            return "near_clear"
        return "normal"

    def build_decision_sample(self) -> Optional[DecisionSample]:
        """
        构造一个完整样本。

        这里同时整理：
        - 任务5要求的 7x4x14 当前状态
        - 任务5要求的 14x4 目标
        - task6 后续还要用的原始 section
        """
        if self.current_piece is None:
            return None
        if self.last_best_eval is None:
            return None
        if (
            self.last_action_score_4x14 is None
            or self.last_action_prob_4x14 is None
            or self.last_legal_mask_4x14 is None
        ):
            return None

        cur_piece = self.current_piece
        board_before = [row[:] for row in self.board]
        active_mask = build_active_mask(cur_piece)

        cur_kind = cur_piece.kind
        kind_id = KIND_TO_INDEX[cur_kind]
        cur_rot4 = cur_piece.rotation % 4
        cur_col14 = leftmost_col(cur_piece)
        current_state_id = (
            kind_id * ACTION_CLASSES
            + cur_rot4 * BOARD_COLS
            + cur_col14
        )

        best_eval = self.last_best_eval
        tgt_rot4 = best_eval.rotation % 4
        tgt_col14 = best_eval.left_col14
        target_action = best_eval.action_idx

        current_state_7x4x14_rows = encode_current_state_7x4x14(
            kind_id,
            cur_rot4,
            cur_col14,
        )
        target_14x4 = encode_target_14x4(tgt_col14, tgt_rot4)
        legal_target_14x4 = transpose_4x14_to_14x4_int(
            self.last_legal_mask_4x14
        )
        score_target_14x4 = transpose_4x14_to_14x4_float(
            self.last_action_score_4x14
        )
        prob_target_14x4 = transpose_4x14_to_14x4_float(
            self.last_action_prob_4x14
        )

        return DecisionSample(
            board_before=board_before,
            active_mask=active_mask,
            board_after_lock=[row[:] for row in best_eval.board_after_lock],
            board_after_clear=[row[:] for row in best_eval.board_after_clear],
            legal_mask_4x14=[row[:] for row in self.last_legal_mask_4x14],
            action_score_4x14=[
                [float(value) for value in row]
                for row in self.last_action_score_4x14
            ],
            action_prob_4x14=[
                [float(value) for value in row]
                for row in self.last_action_prob_4x14
            ],
            current_state_7x4x14_rows=current_state_7x4x14_rows,
            target_14x4=target_14x4,
            legal_target_14x4=legal_target_14x4,
            score_target_14x4=score_target_14x4,
            prob_target_14x4=prob_target_14x4,
            cur_kind=cur_kind,
            kind_id=kind_id,
            cur_rot4=cur_rot4,
            cur_col14=cur_col14,
            current_state_id=current_state_id,
            tgt_rot4=tgt_rot4,
            tgt_col14=tgt_col14,
            target_action=target_action,
            lines_cleared=best_eval.lines_cleared,
            holes_after=best_eval.holes_after,
            blockades_after=best_eval.blockades_after,
            aggregate_height_after=best_eval.aggregate_height_after,
            bumpiness_after=best_eval.bumpiness_after,
            max_height_after=best_eval.max_height_after,
            near_full_rows_before=best_eval.near_full_rows_before,
            near_full_rows_after=best_eval.near_full_rows_after,
            row_transitions_after=best_eval.row_transitions_after,
            col_transitions_after=best_eval.col_transitions_after,
            well_sums_after=best_eval.well_sums_after,
            landing_height=best_eval.landing_height,
            teacher_score=best_eval.teacher_score,
            priority_class=self.last_priority_class,
        )

    def sample_once_for_new_piece(self) -> None:
        """只在新方块刚生成时采一次样本。"""
        if self.sample_saved_for_current_piece:
            return

        sample = self.build_decision_sample()
        if sample is None:
            return

        self.recorder.add_block(sample)
        self.sample_saved_for_current_piece = True

    # ---- step
    def step_frame(self, dt_ms: int) -> None:
        """
        每帧推进一次仿真。

        关键逻辑：
        1) 新方块出现时，先求当前 PD 目标
        2) 立即提取当前状态与目标，保存成 task5 样本
        3) 后面下降过程中不重复采样
        """
        if self.game_over or self.current_piece is None:
            return

        if self.new_piece_spawned:
            evaluated = self.evaluate_all_moves()
            if evaluated is None:
                self.game_over = True
                return

            (
                best_eval,
                legal_mask_4x14,
                action_score_4x14,
                action_prob_4x14,
            ) = evaluated
            self.target_rotation = best_eval.rotation
            self.target_anchor_col = best_eval.anchor_col
            self.last_best_eval = best_eval
            self.last_legal_mask_4x14 = legal_mask_4x14
            self.last_action_score_4x14 = action_score_4x14
            self.last_action_prob_4x14 = action_prob_4x14
            self.last_priority_class = self.classify_priority(best_eval)

            self.sample_once_for_new_piece()
            self.new_piece_spawned = False

            if self.fast_collect:
                self.lock_piece(best_eval.final_piece)
                return

        self._move_timer_ms += dt_ms
        self._gravity_timer_ms += dt_ms

        if self._move_timer_ms >= self.move_step_ms:
            self._move_timer_ms = 0
            self._apply_one_control_step_toward_target()

        if self._gravity_timer_ms >= self.gravity_step_ms:
            self._gravity_timer_ms = 0
            self._apply_gravity_one_row()

    def _apply_one_control_step_toward_target(self) -> None:
        """让当前方块朝当前 PD 目标靠近一步。"""
        if (
            self.current_piece is None
            or self.target_rotation is None
            or self.target_anchor_col is None
        ):
            return

        piece = self.current_piece
        kind = piece.kind
        rot_count = len(SHAPES[kind])

        cur_rot = piece.rotation % rot_count
        tgt_rot = self.target_rotation % rot_count

        if cur_rot != tgt_rot:
            cw_piece = Piece(
                kind=kind,
                rotation=(cur_rot + 1) % rot_count,
                row=piece.row,
                col=piece.col,
            )
            if self.is_valid_position(cw_piece):
                self.current_piece = cw_piece
                return

            ccw_piece = Piece(
                kind=kind,
                rotation=(cur_rot - 1) % rot_count,
                row=piece.row,
                col=piece.col,
            )
            if self.is_valid_position(ccw_piece):
                self.current_piece = ccw_piece
            return

        if piece.col < self.target_anchor_col:
            right_piece = Piece(
                kind=kind,
                rotation=piece.rotation,
                row=piece.row,
                col=piece.col + 1,
            )
            if self.is_valid_position(right_piece):
                self.current_piece = right_piece
            return

        if piece.col > self.target_anchor_col:
            left_piece = Piece(
                kind=kind,
                rotation=piece.rotation,
                row=piece.row,
                col=piece.col - 1,
            )
            if self.is_valid_position(left_piece):
                self.current_piece = left_piece

    def _apply_gravity_one_row(self) -> None:
        """让当前方块自然下落一格，不能下落时就锁定。"""
        if self.current_piece is None:
            return

        piece = self.current_piece
        down_piece = Piece(
            kind=piece.kind,
            rotation=piece.rotation,
            row=piece.row + 1,
            col=piece.col,
        )
        if self.is_valid_position(down_piece):
            self.current_piece = down_piece
            return

        self.lock_piece(piece)


# ============================================================
# Renderer
# ============================================================
class TetrisRenderer:
    """负责绘制棋盘和右侧信息面板。"""

    def __init__(self, screen: pygame.Surface) -> None:
        """初始化字体和面板布局。"""
        self.screen = screen
        pygame.font.init()
        self.font_small = pygame.font.SysFont("microsoftyahei", 18)
        self.font_medium = pygame.font.SysFont(
            "microsoftyahei",
            22,
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

    def draw(self, game: BuiltInTetrisGame, paused: bool) -> None:
        """绘制整张界面。"""
        self.screen.fill(WINDOW_BG)
        self._draw_title(paused, game.fast_collect)
        self._draw_board(game)
        self._draw_panel(game)
        pygame.display.flip()

    def _draw_title(self, paused: bool, fast_collect: bool) -> None:
        """绘制标题栏和提示文字。"""
        title = "Task5 - Extract 7x4x14 Current State and 14x4 PD Target"
        self.screen.blit(
            self.font_title.render(title, True, TEXT_COLOR),
            (LEFT_MARGIN, 10),
        )
        sub1 = "1:打印当前  2:显示CSV路径  3:暂停/继续"
        sub2 = "4:重置  ESC/关闭:自动保存退出"
        mode_text = "FAST_COLLECT=ON" if fast_collect else "FAST_COLLECT=OFF"
        self.screen.blit(
            self.font_small.render(sub1, True, (80, 88, 98)),
            (LEFT_MARGIN, 40),
        )
        self.screen.blit(
            self.font_small.render(sub2, True, (80, 88, 98)),
            (LEFT_MARGIN, 60),
        )
        self.screen.blit(
            self.font_small.render(mode_text, True, (200, 70, 60)),
            (LEFT_MARGIN + 470, 60),
        )
        if paused:
            badge = self.font_medium.render("PAUSED", True, (200, 50, 50))
            self.screen.blit(badge, (WINDOW_WIDTH - 170, 18))

    def _draw_board(self, game: BuiltInTetrisGame) -> None:
        """绘制棋盘、已锁定块、活动块和目标描边。"""
        rect = pygame.Rect(
            self.board_x,
            self.board_y,
            BOARD_COLS * CELL_SIZE,
            BOARD_ROWS * CELL_SIZE,
        )
        pygame.draw.rect(self.screen, (255, 255, 255), rect, border_radius=6)
        pygame.draw.rect(
            self.screen,
            BORDER_COLOR,
            rect,
            width=2,
            border_radius=6,
        )

        for row_idx in range(BOARD_ROWS):
            for col_idx in range(BOARD_COLS):
                x = self.board_x + col_idx * CELL_SIZE
                y = self.board_y + row_idx * CELL_SIZE
                cell = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                value = game.board[row_idx][col_idx]
                if value != 0:
                    kind = ID_TO_KIND.get(value, "O")
                    color = SHAPE_COLORS.get(kind, (130, 130, 130))
                    inner = cell.inflate(-4, -4)
                    pygame.draw.rect(
                        self.screen,
                        color,
                        inner,
                        border_radius=4,
                    )
                pygame.draw.rect(
                    self.screen,
                    GRID_LINE_COLOR,
                    cell,
                    width=1,
                )

        cur_piece = game.current_piece
        if cur_piece is not None and not game.game_over:
            for row_idx, col_idx in cur_piece.cells():
                if 0 <= row_idx < BOARD_ROWS and 0 <= col_idx < BOARD_COLS:
                    x = self.board_x + col_idx * CELL_SIZE
                    y = self.board_y + row_idx * CELL_SIZE
                    cell = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                    inner = cell.inflate(-4, -4)
                    pygame.draw.rect(
                        self.screen,
                        SHAPE_COLORS[cur_piece.kind],
                        inner,
                        border_radius=4,
                    )
                    pygame.draw.rect(
                        self.screen,
                        (255, 255, 255),
                        inner,
                        width=2,
                        border_radius=4,
                    )

        self._draw_overlays(game)

    def _draw_overlays(self, game: BuiltInTetrisGame) -> None:
        """绘制绿色幽灵落点和红色 PD 目标落点。"""
        cur_piece = game.current_piece
        if cur_piece is None or game.game_over:
            return

        ghost_row = game.hard_drop_row_on_board(
            game.board,
            cur_piece.kind,
            cur_piece.rotation,
            cur_piece.col,
        )
        if ghost_row is not None:
            ghost_piece = Piece(
                kind=cur_piece.kind,
                rotation=cur_piece.rotation,
                row=ghost_row,
                col=cur_piece.col,
            )
            self._draw_piece_outline(ghost_piece, GHOST_OVERLAY_COLOR)

        best_eval = game.last_best_eval
        if best_eval is not None:
            self._draw_piece_outline(best_eval.final_piece, TARGET_OVERLAY_COLOR)

    def _draw_piece_outline(
        self,
        piece: Piece,
        color: Tuple[int, int, int],
    ) -> None:
        """用描边形式画出一个方块。"""
        for row_idx, col_idx in piece.cells():
            if 0 <= row_idx < BOARD_ROWS and 0 <= col_idx < BOARD_COLS:
                x = self.board_x + col_idx * CELL_SIZE
                y = self.board_y + row_idx * CELL_SIZE
                cell = pygame.Rect(
                    x + 2,
                    y + 2,
                    CELL_SIZE - 4,
                    CELL_SIZE - 4,
                )
                pygame.draw.rect(
                    self.screen,
                    color,
                    cell,
                    width=OVERLAY_STROKE,
                    border_radius=4,
                )

    def _draw_panel(self, game: BuiltInTetrisGame) -> None:
        """绘制右侧状态信息。"""
        rect = pygame.Rect(
            self.panel_x,
            self.panel_y,
            self.panel_w,
            self.panel_h,
        )
        pygame.draw.rect(self.screen, PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(
            self.screen,
            BORDER_COLOR,
            rect,
            width=2,
            border_radius=8,
        )

        y = self.panel_y + 14
        self.screen.blit(
            self.font_medium.render(
                "Task5 Extraction Panel",
                True,
                TEXT_COLOR,
            ),
            (self.panel_x + 14, y),
        )
        y += 38

        cur_piece = game.current_piece
        if cur_piece is not None:
            cur_rot4 = cur_piece.rotation % 4
            cur_col14 = leftmost_col(cur_piece)
            kind = cur_piece.kind
            kind_id = KIND_TO_INDEX[kind]
            current_state_id = (
                kind_id * ACTION_CLASSES
                + cur_rot4 * BOARD_COLS
                + cur_col14
            )
        else:
            cur_rot4 = -1
            cur_col14 = -1
            kind = "-"
            kind_id = -1
            current_state_id = -1

        best_eval = game.last_best_eval
        if best_eval is not None:
            tgt_rot4 = best_eval.rotation % 4
            tgt_col14 = best_eval.left_col14
            target_action = best_eval.action_idx
            lines_cleared = best_eval.lines_cleared
            teacher_score = best_eval.teacher_score
        else:
            tgt_rot4 = -1
            tgt_col14 = -1
            target_action = -1
            lines_cleared = -1
            teacher_score = 0.0

        lines = [
            f"Blocks saved: {game.recorder.size()}",
            f"Pieces placed: {game.pieces_placed}",
            f"Score: {game.score}",
            f"Lines total: {game.lines_cleared_total}",
            "",
            f"current kind={kind} kind_id={kind_id}",
            f"cur_rot4={cur_rot4} cur_col14={cur_col14}",
            f"current_state_id={current_state_id}",
            "",
            "Task5 explicit encodings:",
            "CURRENT_STATE_7x4x14 -> 28x14 rows",
            "TARGET_14x4 -> 14x4 rows",
            "",
            f"tgt_rot4={tgt_rot4} tgt_col14={tgt_col14}",
            f"target_action={target_action}",
            f"lines_cleared={lines_cleared}",
            f"teacher_score={teacher_score:.4f}",
        ]
        for line in lines:
            self.screen.blit(
                self.font_small.render(line, True, TEXT_COLOR),
                (self.panel_x + 14, y),
            )
            y += 20


# ============================================================
# main
# ============================================================
def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="out_task5")
    parser.add_argument("--max_blocks", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--move_step_ms",
        type=int,
        default=MOVE_STEP_MS_DEFAULT,
    )
    parser.add_argument(
        "--gravity_step_ms",
        type=int,
        default=GRAVITY_STEP_MS_DEFAULT,
    )
    parser.add_argument(
        "--fast_collect",
        type=int,
        default=1,
        choices=[0, 1],
    )
    parser.add_argument("--softmax_temp", type=float, default=0.35)
    return parser.parse_args()


def safe_save_and_exit(recorder: BlockCSVRecorder) -> None:
    """安全保存并退出程序。"""
    recorder.close()
    print(f"[AUTO-SAVE] Saved CSV: {os.path.abspath(recorder.csv_path)}")
    pygame.quit()
    sys.exit()


def main() -> None:
    """程序入口。"""
    args = parse_args()
    random.seed(args.seed)

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Task5 - Extract 7x4x14 and 14x4")
    clock = pygame.time.Clock()

    recorder = BlockCSVRecorder(args.out_dir)
    print("[CSV OUT] =", os.path.abspath(recorder.csv_path))
    print("Hotkeys: 1(print), 2(show csv path), 3(pause), 4(reset), ESC(exit autosave)")
    print("[FAST_COLLECT] =", bool(args.fast_collect))
    print("[SOFTMAX_TEMP] =", args.softmax_temp)
    print("[INFO] This version explicitly writes CURRENT_STATE_7x4x14 and TARGET_14x4.")
    print("[INFO] It also keeps BOARD_BEFORE / ACTIVE_MASK / LEGAL_MASK / ACTION_SCORE / ACTION_PROB for task6.")

    game = BuiltInTetrisGame(
        recorder=recorder,
        move_step_ms=args.move_step_ms,
        gravity_step_ms=args.gravity_step_ms,
        fast_collect=bool(args.fast_collect),
        softmax_temp=args.softmax_temp,
    )
    renderer = TetrisRenderer(screen)
    paused = False

    while True:
        dt = clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                safe_save_and_exit(recorder)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    safe_save_and_exit(recorder)

                if event.key == pygame.K_1:
                    cur_piece = game.current_piece
                    print("---- CURRENT ----")
                    if cur_piece is None:
                        print("piece=None")
                    else:
                        cur_rot4 = cur_piece.rotation % 4
                        cur_col14 = leftmost_col(cur_piece)
                        kind_id = KIND_TO_INDEX[cur_piece.kind]
                        current_state_id = (
                            kind_id * ACTION_CLASSES
                            + cur_rot4 * BOARD_COLS
                            + cur_col14
                        )
                        print(
                            f"kind={cur_piece.kind} kind_id={kind_id} "
                            f"rot4={cur_rot4} col14={cur_col14} "
                            f"current_state_id={current_state_id}"
                        )
                        print(
                            "CURRENT_STATE_7x4x14 row index =",
                            kind_id * 4 + cur_rot4,
                        )

                    if game.last_best_eval is not None:
                        best_eval = game.last_best_eval
                        print(
                            f"target_rot4={best_eval.rotation % 4} "
                            f"target_col14={best_eval.left_col14} "
                            f"target_action={best_eval.action_idx} "
                            f"lines_cleared={best_eval.lines_cleared} "
                            f"holes_after={best_eval.holes_after} "
                            f"teacher_score={best_eval.teacher_score:.6f}"
                        )
                        print(
                            "TARGET_14x4 hot index = "
                            f"row(col) {best_eval.left_col14}, "
                            f"col(rot) {best_eval.rotation % 4}"
                        )
                    else:
                        print("target=None")

                if event.key == pygame.K_2:
                    print("[CSV OUT] =", os.path.abspath(recorder.csv_path))

                if event.key == pygame.K_3:
                    paused = not paused

                if event.key == pygame.K_4:
                    recorder.close()
                    recorder = BlockCSVRecorder(args.out_dir)
                    game = BuiltInTetrisGame(
                        recorder=recorder,
                        move_step_ms=args.move_step_ms,
                        gravity_step_ms=args.gravity_step_ms,
                        fast_collect=bool(args.fast_collect),
                        softmax_temp=args.softmax_temp,
                    )
                    paused = False
                    print("[RESET] New CSV:", os.path.abspath(recorder.csv_path))

        if not paused and not game.game_over:
            game.step_frame(dt)

            if recorder.size() >= args.max_blocks:
                recorder.close()
                print(
                    f"[OK] Reached max_blocks={args.max_blocks}. "
                    f"Saved CSV: {os.path.abspath(recorder.csv_path)}"
                )
                pygame.quit()
                return

        if game.game_over:
            safe_save_and_exit(recorder)

        renderer.draw(game, paused)


if __name__ == "__main__":
    main()