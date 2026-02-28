"""
Task 2 - Tetris Simulation (pygame, 18x14 matrix export)
--------------------------------------------------------
要求满足：
1) 使用 pygame 构建俄罗斯方块仿真系统
2) 棋盘固定为 18x14
3) 可导出当前棋盘状态矩阵（18x14）
4) 不引用后续任务文件（任务2独立）
5) 尽量符合 PEP 8 风格
6) UI 边框与中英文显示优化（右侧面板不截断）

运行前安装：
    pip install pygame numpy
"""

from __future__ import annotations

import csv
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame


# =========================
# Constants (PEP 8 style)
# =========================

BOARD_ROWS = 18
BOARD_COLS = 14

CELL_SIZE = 28
GRID_LINE_WIDTH = 1

TOP_MARGIN = 20
LEFT_MARGIN = 20
SIDE_PANEL_WIDTH = 340  # UI优化：加宽侧边栏避免文字截断
EXTRA_VERTICAL_PADDING = 80  # 上下额外增加的总高度（像素）

WINDOW_WIDTH = LEFT_MARGIN * 2 + BOARD_COLS * CELL_SIZE + SIDE_PANEL_WIDTH
WINDOW_HEIGHT = TOP_MARGIN * 2 + BOARD_ROWS * CELL_SIZE + EXTRA_VERTICAL_PADDING
FPS = 60
DROP_INTERVAL_MS = 450

SAVE_DIR = Path("out_task2_sim")
AUTO_RECORD_EVERY_FRAME = False  # True 时每次更新都会记录状态到内存

# Colors
COLOR_BG = (20, 20, 24)
COLOR_GRID = (55, 55, 62)
COLOR_TEXT = (235, 235, 235)
COLOR_PANEL = (32, 32, 38)
COLOR_BORDER = (120, 120, 130)
COLOR_SUBTITLE = (210, 220, 240)
COLOR_SPLIT = (90, 90, 105)
COLOR_HINT = (255, 255, 180)

# Piece colors (IDs 1~7)
PIECE_COLORS: Dict[int, Tuple[int, int, int]] = {
    1: (0, 240, 240),    # I
    2: (240, 240, 0),    # O
    3: (160, 0, 240),    # T
    4: (0, 240, 0),      # S
    5: (240, 0, 0),      # Z
    6: (0, 0, 240),      # J
    7: (240, 160, 0),    # L
}

# Action names
ACTION_NONE = "NONE"
ACTION_LEFT = "LEFT"
ACTION_RIGHT = "RIGHT"
ACTION_ROTATE = "ROTATE"
ACTION_SOFT_DROP = "SOFT_DROP"
ACTION_HARD_DROP = "HARD_DROP"

VALID_ACTIONS = {
    ACTION_NONE,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_ROTATE,
    ACTION_SOFT_DROP,
    ACTION_HARD_DROP,
}

# Tetromino shapes (rotation states)
PIECE_SHAPES: Dict[str, List[List[List[int]]]] = {
    "I": [
        [[1, 1, 1, 1]],
        [[1], [1], [1], [1]],
    ],
    "O": [
        [[1, 1],
         [1, 1]],
    ],
    "T": [
        [[0, 1, 0],
         [1, 1, 1]],
        [[1, 0],
         [1, 1],
         [1, 0]],
        [[1, 1, 1],
         [0, 1, 0]],
        [[0, 1],
         [1, 1],
         [0, 1]],
    ],
    "S": [
        [[0, 1, 1],
         [1, 1, 0]],
        [[1, 0],
         [1, 1],
         [0, 1]],
    ],
    "Z": [
        [[1, 1, 0],
         [0, 1, 1]],
        [[0, 1],
         [1, 1],
         [1, 0]],
    ],
    "J": [
        [[1, 0, 0],
         [1, 1, 1]],
        [[1, 1],
         [1, 0],
         [1, 0]],
        [[1, 1, 1],
         [0, 0, 1]],
        [[0, 1],
         [0, 1],
         [1, 1]],
    ],
    "L": [
        [[0, 0, 1],
         [1, 1, 1]],
        [[1, 0],
         [1, 0],
         [1, 1]],
        [[1, 1, 1],
         [1, 0, 0]],
        [[1, 1],
         [0, 1],
         [0, 1]],
    ],
}

PIECE_TYPE_TO_ID = {
    "I": 1,
    "O": 2,
    "T": 3,
    "S": 4,
    "Z": 5,
    "J": 6,
    "L": 7,
}


@dataclass
class Piece:
    """当前下落方块。"""
    piece_type: str
    row: int
    col: int
    rotation_index: int = 0

    @property
    def piece_id(self) -> int:
        return PIECE_TYPE_TO_ID[self.piece_type]

    @property
    def rotations(self) -> List[List[List[int]]]:
        return PIECE_SHAPES[self.piece_type]

    @property
    def shape(self) -> List[List[int]]:
        return self.rotations[self.rotation_index]

    @property
    def height(self) -> int:
        return len(self.shape)

    @property
    def width(self) -> int:
        return len(self.shape[0]) if self.shape else 0

    def rotate_clockwise(self) -> None:
        """顺时针旋转（切换到下一个状态）。"""
        self.rotation_index = (self.rotation_index + 1) % len(self.rotations)

    def rotate_counterclockwise(self) -> None:
        """逆向回退旋转（当旋转后非法时恢复）。"""
        self.rotation_index = (self.rotation_index - 1) % len(self.rotations)


class TetrisGame:
    """
    任务二核心仿真系统（独立，不依赖后续任务文件）。

    - board 只保存已锁定方块（18x14）
    - get_state_matrix() 返回锁定方块 + 当前下落方块
    """

    def __init__(self, rows: int = BOARD_ROWS, cols: int = BOARD_COLS) -> None:
        self.rows = rows
        self.cols = cols
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.current_piece: Optional[Piece] = None
        self.game_over = False
        self.score = 0
        self.lines_cleared_total = 0
        self.last_action = ACTION_NONE

        # 录制缓存（任务二基础版）
        self.frame_states: List[np.ndarray] = []
        self.action_log: List[str] = []
        self.recording_enabled = AUTO_RECORD_EVERY_FRAME

        self.reset()

    # -------------------------
    # Public API (Task 3+ ready)
    # -------------------------
    def reset(self) -> None:
        """重置游戏。"""
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.game_over = False
        self.score = 0
        self.lines_cleared_total = 0
        self.last_action = ACTION_NONE
        self.frame_states.clear()
        self.action_log.clear()

        self.current_piece = self._spawn_piece()
        if self.current_piece is None:
            self.game_over = True

    def is_game_over(self) -> bool:
        """游戏是否结束。"""
        return self.game_over

    def get_board_matrix(self) -> np.ndarray:
        """返回仅包含已锁定方块的棋盘矩阵 (18, 14)。"""
        return self.board.copy()

    def get_state_matrix(self) -> np.ndarray:
        """
        返回完整状态矩阵 (18, 14):
        已锁定方块 + 当前下落方块
        """
        state = self.board.copy()

        if self.current_piece is None:
            return state

        for r_idx, row_data in enumerate(self.current_piece.shape):
            for c_idx, cell in enumerate(row_data):
                if cell == 0:
                    continue

                board_r = self.current_piece.row + r_idx
                board_c = self.current_piece.col + c_idx

                if 0 <= board_r < self.rows and 0 <= board_c < self.cols:
                    state[board_r, board_c] = self.current_piece.piece_id

        return state

    def step(self, action: str = ACTION_NONE) -> Dict[str, object]:
        """
        执行一步逻辑，预留给后续任务3/5使用。

        Returns:
            dict: 包含 moved / locked / cleared_lines / game_over 等信息
        """
        if action not in VALID_ACTIONS:
            action = ACTION_NONE

        if self.game_over:
            return {
                "game_over": True,
                "moved": False,
                "locked": False,
                "cleared_lines": 0,
            }

        self.last_action = action
        moved = False
        locked = False
        cleared_lines = 0

        if action == ACTION_LEFT:
            moved = self.move_piece(-1, 0)
        elif action == ACTION_RIGHT:
            moved = self.move_piece(1, 0)
        elif action == ACTION_ROTATE:
            moved = self.rotate_piece()
        elif action == ACTION_SOFT_DROP:
            moved = self.move_piece(0, 1)
            if not moved:
                locked = True
                cleared_lines = self.lock_current_piece_and_continue()
        elif action == ACTION_HARD_DROP:
            moved = self.hard_drop()
            locked = True
            cleared_lines = self.lock_current_piece_and_continue()
        else:
            moved = False

        if self.recording_enabled and not self.game_over:
            self.record_current_state(action)

        return {
            "game_over": self.game_over,
            "moved": moved,
            "locked": locked,
            "cleared_lines": cleared_lines,
        }

    # -------------------------
    # Core logic
    # -------------------------
    def _spawn_piece(self) -> Optional[Piece]:
        """生成新方块；若出生即碰撞则返回 None。"""
        piece_type = random.choice(list(PIECE_SHAPES.keys()))
        piece = Piece(piece_type=piece_type, row=0, col=0, rotation_index=0)

        # 居中生成
        piece.col = (self.cols - piece.width) // 2
        piece.row = 0

        if not self.is_valid_position(piece, piece.row, piece.col):
            return None

        return piece

    def is_valid_position(self, piece: Piece, new_row: int, new_col: int) -> bool:
        """
        检查方块放在指定位置是否合法（边界 + 与锁定方块碰撞）。
        """
        for r_idx, row_data in enumerate(piece.shape):
            for c_idx, cell in enumerate(row_data):
                if cell == 0:
                    continue

                board_r = new_row + r_idx
                board_c = new_col + c_idx

                if board_r < 0 or board_r >= self.rows:
                    return False
                if board_c < 0 or board_c >= self.cols:
                    return False
                if self.board[board_r, board_c] != 0:
                    return False

        return True

    def move_piece(self, dx: int, dy: int) -> bool:
        """
        移动当前方块。
        dx: 列方向（左-1，右+1）
        dy: 行方向（上-1，下+1）
        """
        if self.current_piece is None or self.game_over:
            return False

        new_col = self.current_piece.col + dx
        new_row = self.current_piece.row + dy

        if self.is_valid_position(self.current_piece, new_row, new_col):
            self.current_piece.col = new_col
            self.current_piece.row = new_row
            return True

        return False

    def rotate_piece(self) -> bool:
        """
        旋转当前方块；若旋转后非法则回退。
        带简单 wall-kick（左右微调尝试）。
        """
        if self.current_piece is None or self.game_over:
            return False

        self.current_piece.rotate_clockwise()

        if self.is_valid_position(
            self.current_piece,
            self.current_piece.row,
            self.current_piece.col,
        ):
            return True

        # 简单 wall-kick
        for offset_col in (-1, 1, -2, 2):
            test_col = self.current_piece.col + offset_col
            if self.is_valid_position(self.current_piece, self.current_piece.row, test_col):
                self.current_piece.col = test_col
                return True

        self.current_piece.rotate_counterclockwise()
        return False

    def hard_drop(self) -> bool:
        """
        硬降到最底部（只移动，不锁定）。
        返回是否至少移动过一次。
        """
        if self.current_piece is None or self.game_over:
            return False

        moved_any = False
        while self.move_piece(0, 1):
            moved_any = True

        return moved_any

    def auto_drop_once(self) -> None:
        """
        自动下落一步。若无法下落则锁定方块并生成新方块。
        """
        if self.current_piece is None or self.game_over:
            return

        moved = self.move_piece(0, 1)
        if not moved:
            self.lock_current_piece_and_continue()

    def lock_current_piece_and_continue(self) -> int:
        """
        锁定当前方块 -> 消行 -> 生成新方块。
        返回本次消除行数。
        """
        if self.current_piece is None:
            return 0

        self.lock_piece()
        cleared = self.clear_lines()

        next_piece = self._spawn_piece()
        if next_piece is None:
            self.current_piece = None
            self.game_over = True
        else:
            self.current_piece = next_piece

        return cleared

    def lock_piece(self) -> None:
        """将当前方块写入锁定棋盘。"""
        if self.current_piece is None:
            return

        piece = self.current_piece
        for r_idx, row_data in enumerate(piece.shape):
            for c_idx, cell in enumerate(row_data):
                if cell == 0:
                    continue

                board_r = piece.row + r_idx
                board_c = piece.col + c_idx

                if 0 <= board_r < self.rows and 0 <= board_c < self.cols:
                    self.board[board_r, board_c] = piece.piece_id

    def clear_lines(self) -> int:
        """清除满行并下移，返回清除行数。"""
        full_rows = []
        for r in range(self.rows):
            if np.all(self.board[r] != 0):
                full_rows.append(r)

        if not full_rows:
            return 0

        remaining_rows = [r for r in range(self.rows) if r not in full_rows]
        new_board = self.board[remaining_rows, :]

        num_cleared = len(full_rows)
        empty_rows = np.zeros((num_cleared, self.cols), dtype=np.int8)
        self.board = np.vstack((empty_rows, new_board)).astype(np.int8)

        self.lines_cleared_total += num_cleared
        self.score += self._score_for_lines(num_cleared)
        return num_cleared

    @staticmethod
    def _score_for_lines(num_cleared: int) -> int:
        """简单计分规则。"""
        score_map = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
        return score_map.get(num_cleared, num_cleared * 200)

    # -------------------------
    # Recording / Saving
    # -------------------------
    def record_current_state(self, action: str = ACTION_NONE) -> None:
        """记录当前完整状态矩阵与动作标签。"""
        state = self.get_state_matrix()
        self.frame_states.append(state.copy())
        self.action_log.append(action)

    def save_current_state_to_npy(self) -> Path:
        """保存当前状态矩阵（单帧）为 .npy 文件。"""
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_path = SAVE_DIR / f"task2_current_state_{timestamp}.npy"
        np.save(file_path, self.get_state_matrix().astype(np.int8))
        return file_path

    def save_recording(self) -> Optional[Tuple[Path, Path]]:
        """
        导出录制结果：
        - states: (N, 18, 14).npy
        - actions: csv
        """
        if not self.frame_states:
            return None

        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        states_path = SAVE_DIR / f"task2_states_{timestamp}.npy"
        actions_path = SAVE_DIR / f"task2_actions_{timestamp}.csv"

        states_array = np.stack(self.frame_states, axis=0).astype(np.int8)
        np.save(states_path, states_array)

        with actions_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["frame_idx", "action"])
            for idx, action in enumerate(self.action_log):
                writer.writerow([idx, action])

        return states_path, actions_path


# =========================
# Pygame Rendering Layer
# =========================

def board_to_screen(col: int, row: int) -> Tuple[int, int]:
    """棋盘坐标 -> 像素坐标（左上角）。"""
    x = LEFT_MARGIN + col * CELL_SIZE
    y = TOP_MARGIN + row * CELL_SIZE
    return x, y


def draw_cell(
    surface: pygame.Surface,
    row: int,
    col: int,
    color: Tuple[int, int, int],
    filled: bool = True,
) -> None:
    """绘制单个格子。"""
    x, y = board_to_screen(col, row)
    rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

    if filled:
        pygame.draw.rect(surface, color, rect)

    pygame.draw.rect(surface, COLOR_GRID, rect, width=GRID_LINE_WIDTH)


def draw_board(surface: pygame.Surface, state_matrix: np.ndarray) -> None:
    """根据状态矩阵绘制棋盘（含当前下落方块）。"""
    board_x, board_y = board_to_screen(0, 0)
    board_w = BOARD_COLS * CELL_SIZE
    board_h = BOARD_ROWS * CELL_SIZE

    # 棋盘背景 + 外框（轻微圆角更柔和）
    pygame.draw.rect(
        surface,
        (26, 26, 32),
        (board_x, board_y, board_w, board_h),
        border_radius=6,
    )
    pygame.draw.rect(
        surface,
        COLOR_BORDER,
        (board_x, board_y, board_w, board_h),
        width=2,
        border_radius=6,
    )

    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            cell_value = int(state_matrix[r, c])
            if cell_value == 0:
                draw_cell(surface, r, c, (0, 0, 0), filled=False)
            else:
                color = PIECE_COLORS.get(cell_value, (180, 180, 180))
                draw_cell(surface, r, c, color, filled=True)


def draw_side_panel(
    surface: pygame.Surface,
    font_small: pygame.font.Font,
    font_big: pygame.font.Font,
    font_title: pygame.font.Font,
    game: TetrisGame,
) -> None:
    """
    绘制右侧信息面板（优化版：边框、留白、中英文更美观）。
    """
    panel_x = LEFT_MARGIN * 2 + BOARD_COLS * CELL_SIZE
    panel_y = TOP_MARGIN
    panel_w = SIDE_PANEL_WIDTH - LEFT_MARGIN
    panel_h = BOARD_ROWS * CELL_SIZE

    panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

    pygame.draw.rect(surface, COLOR_PANEL, panel_rect, border_radius=12)
    pygame.draw.rect(surface, COLOR_BORDER, panel_rect, width=2, border_radius=12)

    # 内边距（关键）
    padding_x = 16
    padding_y = 14
    x0 = panel_x + padding_x
    y = panel_y + padding_y
    content_w = panel_w - padding_x * 2

    # 标题区（中英双语）
    title_cn = font_title.render("任务二：俄罗斯方块仿真", True, COLOR_TEXT)
    title_en = font_big.render("Task 2 - Tetris Simulation", True, COLOR_SUBTITLE)

    surface.blit(title_cn, (x0, y))
    y += 26
    surface.blit(title_en, (x0, y))
    y += 28

    pygame.draw.line(surface, COLOR_SPLIT, (x0, y), (x0 + content_w, y), 1)
    y += 10

    # 状态区
    status_lines = [
        f"分数 Score: {game.score}",
        f"消行 Lines: {game.lines_cleared_total}",
        f"结束 Game Over: {game.game_over}",
        f"录制 Record: {game.recording_enabled}",
        f"帧数 Frames: {len(game.frame_states)}",
    ]

    for line in status_lines:
        txt = font_small.render(line, True, COLOR_TEXT)
        surface.blit(txt, (x0, y))
        y += 20

    y += 2

    # 控制区
    controls_title = font_big.render("操作 Controls", True, COLOR_TEXT)
    surface.blit(controls_title, (x0, y))
    y += 15

    controls_lines = [
        "← / →  左右移动 Move",
        "↑      旋转 Rotate",
        "↓      软降 Soft Drop",
        "Space  硬降 Hard Drop",
        "P      暂停/继续 Pause",
        "R      录制开关 Record",
        "S      保存当前矩阵 Save",
        "E      导出录制数据 Export",
        "N      新游戏 New Game",
        "Esc    退出 Quit",
    ]

    for line in controls_lines:
        txt = font_small.render(line, True, COLOR_TEXT)
        surface.blit(txt, (x0, y))
        y += 20

    y += 6

    # 输出区
    output_title = font_big.render("矩阵输出 Matrix Output", True, COLOR_TEXT)
    surface.blit(output_title, (x0, y))
    y += 20

    output_lines = [
        "state_matrix shape = (18, 14)",
        "0 = 空格 / Empty",
        "1~7 = 方块类型 IDs",
    ]

    for line in output_lines:
        txt = font_small.render(line, True, (220, 220, 220))
        surface.blit(txt, (x0, y))
        y += 18


def draw_overlay_message(
    surface: pygame.Surface,
    font: pygame.font.Font,
    message: str,
    sub_message: Optional[str] = None,
) -> None:
    """棋盘中央覆盖提示（暂停/结束）。"""
    board_x, board_y = board_to_screen(0, 0)
    board_w = BOARD_COLS * CELL_SIZE
    board_h = BOARD_ROWS * CELL_SIZE

    overlay = pygame.Surface((board_w, board_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    surface.blit(overlay, (board_x, board_y))

    msg_surface = font.render(message, True, (255, 255, 255))
    msg_rect = msg_surface.get_rect(
        center=(board_x + board_w // 2, board_y + board_h // 2 - 18)
    )
    surface.blit(msg_surface, msg_rect)

    if sub_message:
        sub_font = pygame.font.SysFont(["consolas", "arial"], 20)
        sub_surface = sub_font.render(sub_message, True, (230, 230, 230))
        sub_rect = sub_surface.get_rect(
            center=(board_x + board_w // 2, board_y + board_h // 2 + 18)
        )
        surface.blit(sub_surface, sub_rect)


# =========================
# Input Handling
# =========================

def handle_keydown(event: pygame.event.Event, game: TetrisGame) -> Optional[str]:
    """
    处理键盘动作键，返回动作名称（仅游戏动作）。
    非动作键（如保存/导出）由 main() 处理。
    """
    if event.key == pygame.K_LEFT:
        game.step(ACTION_LEFT)
        return ACTION_LEFT

    if event.key == pygame.K_RIGHT:
        game.step(ACTION_RIGHT)
        return ACTION_RIGHT

    if event.key == pygame.K_UP:
        game.step(ACTION_ROTATE)
        return ACTION_ROTATE

    if event.key == pygame.K_DOWN:
        game.step(ACTION_SOFT_DROP)
        return ACTION_SOFT_DROP

    if event.key == pygame.K_SPACE:
        game.step(ACTION_HARD_DROP)
        return ACTION_HARD_DROP

    return None


# =========================
# Main Application
# =========================

def main() -> None:
    """程序入口。"""
    pygame.init()
    pygame.display.set_caption("Task 2 - Tetris Simulation (18x14 Matrix Export)")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    # 优先尝试 Windows 常见中文字体，其次英文等宽字体
    font_small = pygame.font.SysFont(
        ["microsoft yahei", "microsoftyahei", "consolas", "arial"],
        16,
    )
    font_big = pygame.font.SysFont(
        ["microsoft yahei", "microsoftyahei", "consolas", "arial"],
        18,
        bold=True,
    )
    font_title = pygame.font.SysFont(
        ["microsoft yahei", "microsoftyahei", "simhei", "arial"],
        20,
        bold=True,
    )
    font_overlay = pygame.font.SysFont(["consolas", "arial"], 28, bold=True)

    game = TetrisGame()
    paused = False
    last_drop_time = pygame.time.get_ticks()

    info_message = ""
    info_message_expire_ms = 0

    running = True
    while running:
        now_ms = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    continue

                if event.key == pygame.K_p:
                    paused = not paused
                    continue

                if event.key == pygame.K_n:
                    game.reset()
                    paused = False
                    last_drop_time = now_ms
                    info_message = "New game started."
                    info_message_expire_ms = now_ms + 1500
                    continue

                if event.key == pygame.K_r:
                    game.recording_enabled = not game.recording_enabled
                    info_message = f"Recording = {game.recording_enabled}"
                    info_message_expire_ms = now_ms + 1500
                    continue

                if event.key == pygame.K_s:
                    save_path = game.save_current_state_to_npy()
                    current_state = game.get_state_matrix()
                    print("Saved current state:", save_path)
                    print("Current state shape:", current_state.shape)
                    print(current_state)
                    info_message = "Saved current 18x14 state (.npy)"
                    info_message_expire_ms = now_ms + 1800
                    continue

                if event.key == pygame.K_e:
                    result = game.save_recording()
                    if result is None:
                        info_message = "No recorded frames to export."
                    else:
                        states_path, actions_path = result
                        print("Exported recording:")
                        print("  states :", states_path)
                        print("  actions:", actions_path)
                        info_message = "Exported recorded frames/actions."
                    info_message_expire_ms = now_ms + 1800
                    continue

                if not paused and not game.is_game_over():
                    handle_keydown(event, game)

        # 自动下落（按时间间隔触发）
        if not paused and not game.is_game_over():
            if now_ms - last_drop_time >= DROP_INTERVAL_MS:
                game.auto_drop_once()
                if game.recording_enabled and not game.is_game_over():
                    # 自动下落记录一帧，动作标签记 NONE
                    game.record_current_state(ACTION_NONE)
                last_drop_time = now_ms

        # 绘制
        screen.fill(COLOR_BG)

        state_matrix = game.get_state_matrix()
        draw_board(screen, state_matrix)
        draw_side_panel(screen, font_small, font_big, font_title, game)

        if paused and not game.is_game_over():
            draw_overlay_message(screen, font_overlay, "PAUSED", "Press P to continue")

        if game.is_game_over():
            draw_overlay_message(screen, font_overlay, "GAME OVER", "Press N for new game")

        if info_message and now_ms < info_message_expire_ms:
            info_surface = font_small.render(info_message, True, COLOR_HINT)
            screen.blit(info_surface, (LEFT_MARGIN, 4))

        pygame.display.flip()
        clock.tick(FPS)

    # 退出前：若录制开启且有缓存，自动导出，避免丢数据
    if game.recording_enabled and game.frame_states:
        result = game.save_recording()
        if result is not None:
            states_path, actions_path = result
            print("Auto-export on quit:")
            print("  states :", states_path)
            print("  actions:", actions_path)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()