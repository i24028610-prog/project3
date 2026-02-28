import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import pygame


# =========================
# Config
# =========================
CELL_SIZE = 28
BOARD_ROWS = 18
BOARD_COLS = 10

EXPORT_ROWS = 18
EXPORT_COLS = 14
SIDE_PADDING_COLS = 2

FPS = 60
AUTO_STEP_MS = 280

WINDOW_BG = (245, 247, 250)
GRID_LINE_COLOR = (210, 214, 220)
TEXT_COLOR = (35, 42, 52)
PANEL_BG = (232, 236, 242)
BORDER_COLOR = (60, 70, 85)
PATH_COLOR = (255, 140, 80)
TARGET_COLOR = (220, 70, 70)

TOP_MARGIN = 60
BOTTOM_MARGIN = 110
LEFT_MARGIN = 40
RIGHT_PANEL_WIDTH = 300

WINDOW_WIDTH = (
    LEFT_MARGIN + BOARD_COLS * CELL_SIZE + 40 + RIGHT_PANEL_WIDTH + 80
)
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


@dataclass
class Piece:
    """Current falling tetromino."""

    kind: str
    rotation: int
    row: int
    col: int

    def cells(self) -> List[Tuple[int, int]]:
        """Return absolute cell positions of current piece."""
        rotation_count = len(SHAPES[self.kind])
        rotation_index = self.rotation % rotation_count
        offsets = SHAPES[self.kind][rotation_index]
        return [(self.row + row, self.col + col) for row, col in offsets]


class TetrisGame:
    """Tetris simulation with PD heuristic auto-player."""

    def __init__(self) -> None:
        self.board: List[List[int]] = [
            [0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)
        ]
        self.score = 0
        self.lines_cleared_total = 0
        self.pieces_placed = 0
        self.game_over = False

        # Task 5 compatible variables
        self.target_rotation: Optional[int] = None
        self.target_col: Optional[int] = None
        self.target_final_row: Optional[int] = None
        self.new_piece_spawned = True

        # Motion path
        self.action_queue: Deque[str] = deque()
        self.last_action = "NONE"

        self.bag: List[str] = []
        self.current_piece: Optional[Piece] = None
        self.next_piece_kind = self._draw_from_bag()

        self.spawn_piece()

    # -------------------------
    # Piece generation
    # -------------------------
    def _refill_bag(self) -> None:
        """Refill piece bag using 7-bag shuffle."""
        self.bag = list(SHAPES.keys())
        random.shuffle(self.bag)

    def _draw_from_bag(self) -> str:
        """Draw one piece kind from bag."""
        if not self.bag:
            self._refill_bag()
        return self.bag.pop()

    def spawn_piece(self) -> None:
        """
        Spawn next piece.
        If spawn collides, game over.
        Also mark that a new piece has appeared.
        """
        kind = self.next_piece_kind
        self.next_piece_kind = self._draw_from_bag()

        spawn_col = 3
        piece = Piece(kind=kind, rotation=0, row=0, col=spawn_col)

        for offset in [0, -1, 1, -2, 2]:
            candidate = Piece(
                kind=kind,
                rotation=0,
                row=0,
                col=spawn_col + offset,
            )
            if self.is_valid_position(candidate):
                self.current_piece = candidate
                self.target_rotation = None
                self.target_col = None
                self.target_final_row = None
                self.new_piece_spawned = True
                self.action_queue.clear()
                self.last_action = "SPAWN"
                return

        self.current_piece = piece
        self.game_over = True

    # -------------------------
    # Collision / placement
    # -------------------------
    def is_valid_position(self, piece: Piece) -> bool:
        """Check whether piece is inside board and not colliding."""
        for row, col in piece.cells():
            if col < 0 or col >= BOARD_COLS:
                return False
            if row < 0 or row >= BOARD_ROWS:
                return False
            if self.board[row][col] != 0:
                return False
        return True

    def hard_drop_row(
        self,
        kind: str,
        rotation: int,
        col: int,
    ) -> Optional[int]:
        """Return final row after hard drop for given piece state."""
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

    def lock_piece(self, piece: Piece) -> None:
        """Merge piece into board, clear lines, update score, spawn next."""
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
        """Remove full rows and return number of cleared rows."""
        new_board = [
            row for row in self.board if any(cell == 0 for cell in row)
        ]
        cleared = BOARD_ROWS - len(new_board)

        while len(new_board) < BOARD_ROWS:
            new_board.insert(0, [0 for _ in range(BOARD_COLS)])

        self.board = new_board
        return cleared

    # -------------------------
    # Export board matrix 18x14
    # -------------------------
    def export_board_matrix_18x14(self) -> List[List[int]]:
        """
        Export locked board to 18x14 matrix.
        Left and right are padded with 2 columns of zeros.
        """
        matrix = []
        for row in range(EXPORT_ROWS):
            out_row = [0] * SIDE_PADDING_COLS
            out_row.extend(self.board[row][:BOARD_COLS])
            out_row.extend([0] * SIDE_PADDING_COLS)
            matrix.append(out_row)
        return matrix

    # -------------------------
    # Task 5 compatible helpers
    # -------------------------
    def get_current_piece_kind_index(self) -> Optional[int]:
        """Return current piece kind index in fixed 7-kind space."""
        if self.current_piece is None:
            return None
        return PIECE_TO_INDEX[self.current_piece.kind]

    def get_current_piece_rot4(self) -> Optional[int]:
        """Return current rotation encoded into fixed 4-way space."""
        if self.current_piece is None:
            return None
        return self.current_piece.rotation % 4

    def get_target_rot4(self) -> Optional[int]:
        """Return cached target rotation encoded into fixed 4-way space."""
        if self.target_rotation is None:
            return None
        return self.target_rotation % 4

    def get_current_piece_col14(self) -> Optional[int]:
        """Return current column mapped from 10-column board to 14-column space."""
        if self.current_piece is None:
            return None
        return self.current_piece.col + SIDE_PADDING_COLS

    def get_target_col14(self) -> Optional[int]:
        """Return target column mapped from 10-column board to 14-column space."""
        if self.target_col is None:
            return None
        return self.target_col + SIDE_PADDING_COLS

    def get_current_position_code(self) -> Optional[Tuple[int, int, int]]:
        """
        Return current state code:
        (kind_index, rotation_4, col_14)
        """
        kind_index = self.get_current_piece_kind_index()
        rot4 = self.get_current_piece_rot4()
        col14 = self.get_current_piece_col14()

        if kind_index is None or rot4 is None or col14 is None:
            return None
        return kind_index, rot4, col14

    def get_target_position_code(self) -> Optional[Tuple[int, int]]:
        """
        Return target state code:
        (col_14, rotation_4)
        """
        col14 = self.get_target_col14()
        rot4 = self.get_target_rot4()

        if col14 is None or rot4 is None:
            return None
        return col14, rot4

    def encode_current_piece_7x4x14(self) -> Optional[List[List[List[int]]]]:
        """
        Encode current piece position as one-hot tensor [7][4][14].
        """
        code = self.get_current_position_code()
        if code is None:
            return None

        kind_index, rot4, col14 = code
        tensor = [
            [[0 for _ in range(EXPORT_COLS)] for _ in range(4)]
            for _ in range(7)
        ]
        tensor[kind_index][rot4][col14] = 1
        return tensor

    def encode_target_14x4(self) -> Optional[List[List[int]]]:
        """
        Encode target position as one-hot tensor [14][4].
        """
        code = self.get_target_position_code()
        if code is None:
            return None

        col14, rot4 = code
        tensor = [[0 for _ in range(4)] for _ in range(EXPORT_COLS)]
        tensor[col14][rot4] = 1
        return tensor

    def get_task5_sample_fields(self) -> Optional[dict]:
        """
        Return structured fields for Task 5 dataset.
        """
        if self.current_piece is None:
            return None
        if self.target_rotation is None or self.target_col is None:
            return None

        return {
            "kind": self.current_piece.kind,
            "kind_index": self.get_current_piece_kind_index(),
            "current_row": self.current_piece.row,
            "current_rotation": self.current_piece.rotation,
            "current_rotation_4": self.get_current_piece_rot4(),
            "current_col_10": self.current_piece.col,
            "current_col_14": self.get_current_piece_col14(),
            "target_rotation": self.target_rotation,
            "target_rotation_4": self.get_target_rot4(),
            "target_col_10": self.target_col,
            "target_col_14": self.get_target_col14(),
            "target_final_row": self.target_final_row,
        }

    # -------------------------
    # PD heuristic auto-player
    # -------------------------
    def choose_best_move(self) -> Optional[Tuple[int, int]]:
        """
        Choose best move using Pierre Dellacherie heuristic.
        Return (rotation, col) or None.
        """
        if self.current_piece is None:
            return None

        kind = self.current_piece.kind
        best_score = -float("inf")
        best_move = None

        rotations = len(SHAPES[kind])
        checked = set()

        for rotation in range(rotations):
            shape_offsets = SHAPES[kind][rotation]
            min_col = min(col for _, col in shape_offsets)
            max_col = max(col for _, col in shape_offsets)

            for col in range(-min_col, BOARD_COLS - max_col):
                key = (rotation, col, tuple(sorted(shape_offsets)))
                if key in checked:
                    continue
                checked.add(key)

                final_row = self.hard_drop_row(kind, rotation, col)
                if final_row is None:
                    continue

                piece = Piece(
                    kind=kind,
                    rotation=rotation,
                    row=final_row,
                    col=col,
                )
                board_after, lines_cleared = self.simulate_lock(
                    self.board,
                    piece,
                )
                features = self.compute_pd_features(
                    board_after,
                    piece,
                    lines_cleared,
                )
                heuristic_score = self.pd_score(features)

                if heuristic_score > best_score:
                    best_score = heuristic_score
                    best_move = (rotation, col)

        return best_move

    @staticmethod
    def simulate_lock(
        board: List[List[int]],
        piece: Piece,
    ) -> Tuple[List[List[int]], int]:
        """Return simulated board after locking piece and clearing lines."""
        test_board = [row[:] for row in board]

        for row, col in piece.cells():
            if 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS:
                test_board[row][col] = SHAPE_ID[piece.kind]

        kept_rows = [
            row for row in test_board if any(cell == 0 for cell in row)
        ]
        lines_cleared = BOARD_ROWS - len(kept_rows)

        while len(kept_rows) < BOARD_ROWS:
            kept_rows.insert(0, [0 for _ in range(BOARD_COLS)])

        return kept_rows, lines_cleared

    @staticmethod
    def count_holes(board: List[List[int]]) -> int:
        """Count holes: empty cells with a filled cell above."""
        holes = 0
        for col in range(BOARD_COLS):
            filled_seen = False
            for row in range(BOARD_ROWS):
                if board[row][col] != 0:
                    filled_seen = True
                elif filled_seen:
                    holes += 1
        return holes

    @staticmethod
    def row_transitions(board: List[List[int]]) -> int:
        """Count row transitions. Boundaries are treated as filled."""
        transitions = 0
        for row in range(BOARD_ROWS):
            prev_filled = 1
            for col in range(BOARD_COLS):
                current_filled = 1 if board[row][col] != 0 else 0
                if current_filled != prev_filled:
                    transitions += 1
                prev_filled = current_filled
            if prev_filled != 1:
                transitions += 1
        return transitions

    @staticmethod
    def column_transitions(board: List[List[int]]) -> int:
        """Count column transitions. Boundaries are treated as filled."""
        transitions = 0
        for col in range(BOARD_COLS):
            prev_filled = 1
            for row in range(BOARD_ROWS):
                current_filled = 1 if board[row][col] != 0 else 0
                if current_filled != prev_filled:
                    transitions += 1
                prev_filled = current_filled
            if prev_filled != 1:
                transitions += 1
        return transitions

    @staticmethod
    def well_sums(board: List[List[int]]) -> int:
        """Return total well depth sum."""
        total = 0
        for col in range(BOARD_COLS):
            depth = 0
            for row in range(BOARD_ROWS):
                if board[row][col] != 0:
                    depth = 0
                    continue

                left_filled = (
                    col == 0 or board[row][col - 1] != 0
                )
                right_filled = (
                    col == BOARD_COLS - 1 or board[row][col + 1] != 0
                )

                if left_filled and right_filled:
                    depth += 1
                    total += depth
                else:
                    depth = 0
        return total

    @staticmethod
    def landing_height(piece: Piece) -> float:
        """Return approximate landing height from bottom."""
        rows = [row for row, _ in piece.cells()]
        avg_row = sum(rows) / len(rows)
        return BOARD_ROWS - avg_row

    def compute_pd_features(
        self,
        board_after: List[List[int]],
        piece: Piece,
        lines_cleared: int,
    ) -> Dict[str, float]:
        """Compute PD features for scoring."""
        return {
            "landing_height": self.landing_height(piece),
            "rows_eliminated": float(lines_cleared),
            "row_transitions": float(self.row_transitions(board_after)),
            "col_transitions": float(self.column_transitions(board_after)),
            "holes": float(self.count_holes(board_after)),
            "well_sums": float(self.well_sums(board_after)),
        }

    @staticmethod
    def pd_score(features: Dict[str, float]) -> float:
        """Classic Pierre Dellacherie heuristic weights."""
        return (
            -4.500158825082766 * features["landing_height"]
            + 3.4181268101392694 * features["rows_eliminated"]
            - 3.2178882868487753 * features["row_transitions"]
            - 9.348695305445199 * features["col_transitions"]
            - 7.899265427351652 * features["holes"]
            - 3.3855972247263626 * features["well_sums"]
        )

    # -------------------------
    # Motion trajectory planning
    # -------------------------
    def build_horizontal_rotation_path(
        self,
        piece: Piece,
        target_rotation: int,
        target_col: int,
    ) -> Optional[List[str]]:
        """
        Find a valid action path at current row:
        rotate / left / right
        """
        rotation_count = len(SHAPES[piece.kind])
        start_state = (piece.rotation % rotation_count, piece.col)
        target_state = (target_rotation % rotation_count, target_col)

        if start_state == target_state:
            return []

        queue: Deque[Tuple[int, int]] = deque([start_state])
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {
            start_state: None
        }
        action_used: Dict[Tuple[int, int], str] = {}

        while queue:
            rotation, col = queue.popleft()

            candidates = [
                ("rotate", ((rotation + 1) % rotation_count, col)),
                ("left", (rotation, col - 1)),
                ("right", (rotation, col + 1)),
            ]

            for action_name, next_state in candidates:
                next_rotation, next_col = next_state
                if next_state in parent:
                    continue

                candidate_piece = Piece(
                    kind=piece.kind,
                    rotation=next_rotation,
                    row=piece.row,
                    col=next_col,
                )
                if not self.is_valid_position(candidate_piece):
                    continue

                parent[next_state] = (rotation, col)
                action_used[next_state] = action_name

                if next_state == target_state:
                    path = []
                    cur = next_state
                    while parent[cur] is not None:
                        path.append(action_used[cur])
                        cur = parent[cur]
                    path.reverse()
                    return path

                queue.append(next_state)

        return None

    def plan_actions_for_current_piece(self) -> bool:
        """
        Plan motion trajectory for the current piece:
        rotate / left-right / down / lock
        """
        if self.current_piece is None:
            return False

        move = self.choose_best_move()
        if move is None:
            return False

        self.target_rotation, self.target_col = move
        self.target_final_row = self.hard_drop_row(
            self.current_piece.kind,
            self.target_rotation,
            self.target_col,
        )
        if self.target_final_row is None:
            return False

        path = self.build_horizontal_rotation_path(
            self.current_piece,
            self.target_rotation,
            self.target_col,
        )
        if path is None:
            return False

        down_steps = self.target_final_row - self.current_piece.row
        if down_steps < 0:
            return False

        self.action_queue.clear()
        for action in path:
            self.action_queue.append(action)

        for _ in range(down_steps):
            self.action_queue.append("down")

        self.action_queue.append("lock")
        return True

    def execute_one_action(self) -> None:
        """Execute one planned action."""
        if self.current_piece is None:
            return

        if not self.action_queue:
            self.last_action = "NONE"
            return

        action = self.action_queue.popleft()
        self.last_action = action.upper()

        if action == "rotate":
            candidate = Piece(
                kind=self.current_piece.kind,
                rotation=self.current_piece.rotation + 1,
                row=self.current_piece.row,
                col=self.current_piece.col,
            )
            if self.is_valid_position(candidate):
                self.current_piece = candidate
            return

        if action == "left":
            candidate = Piece(
                kind=self.current_piece.kind,
                rotation=self.current_piece.rotation,
                row=self.current_piece.row,
                col=self.current_piece.col - 1,
            )
            if self.is_valid_position(candidate):
                self.current_piece = candidate
            return

        if action == "right":
            candidate = Piece(
                kind=self.current_piece.kind,
                rotation=self.current_piece.rotation,
                row=self.current_piece.row,
                col=self.current_piece.col + 1,
            )
            if self.is_valid_position(candidate):
                self.current_piece = candidate
            return

        if action == "down":
            candidate = Piece(
                kind=self.current_piece.kind,
                rotation=self.current_piece.rotation,
                row=self.current_piece.row + 1,
                col=self.current_piece.col,
            )
            if self.is_valid_position(candidate):
                self.current_piece = candidate
            else:
                self.lock_piece(self.current_piece)
            return

        if action == "lock":
            self.lock_piece(self.current_piece)

    def auto_step(self) -> None:
        """
        One auto-play step with visible motion trajectory.

        New piece:
        - compute target once
        - build action sequence once

        Then:
        - execute one action per step
        """
        if self.game_over or self.current_piece is None:
            return

        if self.new_piece_spawned:
            ok = self.plan_actions_for_current_piece()
            if not ok:
                self.game_over = True
                return
            self.new_piece_spawned = False

        self.execute_one_action()


class TetrisRenderer:
    """Pygame renderer for the Tetris auto-player."""

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        pygame.font.init()
        self.font_small = pygame.font.SysFont("microsoftyahei", 19)
        self.font_medium = pygame.font.SysFont(
            "microsoftyahei",
            24,
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

    def draw(self, game: TetrisGame) -> None:
        """Draw full UI."""
        self.screen.fill(WINDOW_BG)
        self.draw_title()
        self.draw_board(game)
        self.draw_side_panel(game)
        self.draw_footer_hint(game)
        pygame.display.flip()

    def draw_title(self) -> None:
        """Draw title and subtitle."""
        title = "Task 3 - Tetris Auto Player (Pierre Dellacherie)"
        subtitle = "任务三：慢速 + 分步运动轨迹版本"

        title_surf = self.font_title.render(title, True, TEXT_COLOR)
        sub_surf = self.font_small.render(subtitle, True, (80, 88, 98))

        self.screen.blit(title_surf, (LEFT_MARGIN, 10))
        self.screen.blit(sub_surf, (LEFT_MARGIN, 36))

    def draw_board(self, game: TetrisGame) -> None:
        """Draw board, target preview and current piece."""
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
                    kind = self.id_to_kind(value)
                    color = SHAPE_COLORS.get(kind, (130, 130, 130))
                    inner_rect = cell_rect.inflate(-4, -4)
                    pygame.draw.rect(
                        self.screen,
                        color,
                        inner_rect,
                        border_radius=4,
                    )

                pygame.draw.rect(self.screen, GRID_LINE_COLOR, cell_rect, width=1)

        self.draw_motion_path(game)
        self.draw_target_preview(game)
        self.draw_current_piece(game)

    def draw_current_piece(self, game: TetrisGame) -> None:
        """Draw current moving piece."""
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

    def draw_target_preview(self, game: TetrisGame) -> None:
        """Draw target final position with red outline."""
        if game.current_piece is None:
            return
        if game.target_rotation is None or game.target_col is None:
            return
        if game.target_final_row is None:
            return

        target_piece = Piece(
            kind=game.current_piece.kind,
            rotation=game.target_rotation,
            row=game.target_final_row,
            col=game.target_col,
        )

        for row, col in target_piece.cells():
            if 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS:
                x = self.board_x + col * CELL_SIZE
                y = self.board_y + row * CELL_SIZE
                cell_rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                target_rect = cell_rect.inflate(-8, -8)
                pygame.draw.rect(
                    self.screen,
                    TARGET_COLOR,
                    target_rect,
                    width=2,
                    border_radius=3,
                )

    def draw_motion_path(self, game: TetrisGame) -> None:
        """Draw a motion path line from current piece center to target center."""
        if game.current_piece is None:
            return
        if game.target_rotation is None or game.target_col is None:
            return
        if game.target_final_row is None:
            return

        current_cells = game.current_piece.cells()
        target_piece = Piece(
            kind=game.current_piece.kind,
            rotation=game.target_rotation,
            row=game.target_final_row,
            col=game.target_col,
        )
        target_cells = target_piece.cells()

        current_center = self.cells_center(current_cells)
        target_center = self.cells_center(target_cells)

        pygame.draw.line(
            self.screen,
            PATH_COLOR,
            current_center,
            target_center,
            3,
        )

        pygame.draw.circle(self.screen, PATH_COLOR, current_center, 4)
        pygame.draw.circle(self.screen, PATH_COLOR, target_center, 4)

    def cells_center(self, cells: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Return screen center point of given piece cells."""
        xs = []
        ys = []

        for row, col in cells:
            x = self.board_x + col * CELL_SIZE + CELL_SIZE // 2
            y = self.board_y + row * CELL_SIZE + CELL_SIZE // 2
            xs.append(x)
            ys.append(y)

        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

    def draw_side_panel(self, game: TetrisGame) -> None:
        """Draw stats and next piece panel."""
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

        y = self.panel_y + 14

        stats_title = self.font_medium.render("自动运行状态", True, TEXT_COLOR)
        self.screen.blit(stats_title, (self.panel_x + 14, y))
        y += 40

        stats = [
            f"Score 分数: {game.score}",
            f"Lines 消行: {game.lines_cleared_total}",
            f"Pieces 方块数: {game.pieces_placed}",
            f"Board 矩阵: {EXPORT_ROWS}x{EXPORT_COLS}",
            f"Mode 模式: Auto (PD)",
            f"Step 间隔: {AUTO_STEP_MS} ms",
            f"Current rot4: {game.get_current_piece_rot4()}",
            f"Current col14: {game.get_current_piece_col14()}",
            f"Target rot4: {game.get_target_rot4()}",
            f"Target col14: {game.get_target_col14()}",
            f"Target row: {game.target_final_row}",
            f"Last action: {game.last_action}",
            f"Queue left: {len(game.action_queue)}",
        ]

        for line in stats:
            surf = self.font_small.render(line, True, TEXT_COLOR)
            self.screen.blit(surf, (self.panel_x + 14, y))
            y += 24

        y += 6
        next_title = self.font_medium.render("Next 下一个", True, TEXT_COLOR)
        self.screen.blit(next_title, (self.panel_x + 14, y))
        y += 42

        preview_rect = pygame.Rect(self.panel_x + 14, y, 150, 120)
        pygame.draw.rect(self.screen, (255, 255, 255), preview_rect, border_radius=6)
        pygame.draw.rect(
            self.screen,
            BORDER_COLOR,
            preview_rect,
            width=2,
            border_radius=6,
        )

        self.draw_next_piece(game.next_piece_kind, preview_rect)

        y += 138
        if game.game_over:
            over_text = self.font_medium.render("GAME OVER", True, TARGET_COLOR)
            tip_text = self.font_small.render(
                "Press R to restart / 按 R 重开",
                True,
                (90, 90, 90),
            )
            self.screen.blit(over_text, (self.panel_x + 14, y))
            self.screen.blit(tip_text, (self.panel_x + 14, y + 34))

    def draw_next_piece(self, kind: str, preview_rect: pygame.Rect) -> None:
        """Draw next tetromino preview centered in preview box."""
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

    def draw_footer_hint(self, game: TetrisGame) -> None:
        """Draw footer help and matrix preview."""
        y = TOP_MARGIN + BOARD_ROWS * CELL_SIZE + 16

        hints = [
            "R: 重开",
            "P: 暂停/继续",
            "ESC: 退出",
            "Task3 = 慢速 + 分步轨迹 + 自动决策",
        ]
        text = "    |    ".join(hints)

        surf = self.font_small.render(text, True, (75, 82, 90))
        self.screen.blit(surf, (LEFT_MARGIN, y))

        matrix = game.export_board_matrix_18x14()
        preview_1 = "M[0]: " + str(matrix[0])
        preview_2 = "M[1]: " + str(matrix[1])

        surf1 = self.font_small.render(preview_1[:120], True, (95, 100, 108))
        surf2 = self.font_small.render(preview_2[:120], True, (95, 100, 108))

        self.screen.blit(surf1, (LEFT_MARGIN, y + 24))
        self.screen.blit(surf2, (LEFT_MARGIN, y + 48))

    @staticmethod
    def id_to_kind(value: int) -> str:
        """Convert numeric block id back to tetromino kind."""
        for kind, shape_id in SHAPE_ID.items():
            if value == shape_id:
                return kind
        return "O"


def main() -> None:
    """Run Task 3 Tetris auto-player."""
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(
        "Task 3 - Tetris Auto Player (PD Heuristic)"
    )
    clock = pygame.time.Clock()

    game = TetrisGame()
    renderer = TetrisRenderer(screen)

    paused = False
    auto_timer = 0

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
                    game = TetrisGame()
                    paused = False
                    auto_timer = 0

                if event.key == pygame.K_p:
                    paused = not paused

        if not paused and not game.game_over and auto_timer >= AUTO_STEP_MS:
            auto_timer = 0
            game.auto_step()

        renderer.draw(game)


if __name__ == "__main__":
    main()