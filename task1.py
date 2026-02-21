"""
Task 1 - Real Image Capture for Tetris (.exe) [FULL VERSION - Edge Based]

What it does:
- Capture the real Tetris .exe board from screen (ROI).
- Convert each frame into an 18x14 grid and save as CSV (human-readable).
- Record one action per frame (CSV), aligned with frames.
- Show a GUI window and mark "landing position" (Mode A):
  per-column landing row + highlight a target column.

Why this version works for your game:
- Your blocks become GRAY after landing (low saturation / close to background).
- So we do NOT rely on color.
- We detect occupancy using EDGE ENERGY (Sobel magnitude), which still works
  for gray blocks due to borders/contrast.

Multi-monitor + Windows scaling fixes:
- DPI aware (reduces coordinate mismatch under scaling)
- ROI selection uses virtual desktop (all monitors) via mss.monitors[0]
- ROI coordinates corrected with virtual desktop left/top offset

Controls:
- Q: Quit
- R: Re-select ROI (overwrite roi_config.json)
- C: Calibrate edge threshold from current ROI (best at early game / empty board)

Dependencies:
- pip install opencv-python numpy mss pynput
"""

from __future__ import annotations

import csv
import ctypes
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from mss import mss
from pynput import keyboard

# -----------------------------
# Configuration
# -----------------------------

ROWS = 18
COLS = 14

FPS = 15
FRAME_INTERVAL = 1.0 / FPS

OUTPUT_DIR = Path("out_task1_real")
ROI_CONFIG_PATH = OUTPUT_DIR / "roi_config.json"

WINDOW_NAME = "Task1 Capture (Q quit | R reselect ROI | C calibrate)"

# Actions (edit if your project uses other names)
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

TOTAL_CELLS = ROWS * COLS


# -----------------------------
# DPI fix (Windows)
# -----------------------------


def make_process_dpi_aware() -> None:
    """Make process DPI-aware to reduce ROI mismatch under Windows scaling."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


# -----------------------------
# Data classes
# -----------------------------


@dataclass
class Roi:
    """Region of interest in absolute screen coordinates."""
    left: int
    top: int
    width: int
    height: int

    def to_mss(self) -> Dict[str, int]:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "width": int(self.width),
            "height": int(self.height),
        }

    def is_valid(self) -> bool:
        return self.width > 10 and self.height > 10


# -----------------------------
# Utilities
# -----------------------------


def ensure_out_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def load_roi() -> Optional[Roi]:
    if not ROI_CONFIG_PATH.exists():
        return None
    data = json.loads(ROI_CONFIG_PATH.read_text(encoding="utf-8"))
    roi = Roi(
        left=int(data["left"]),
        top=int(data["top"]),
        width=int(data["width"]),
        height=int(data["height"]),
    )
    return roi if roi.is_valid() else None


def save_roi(roi: Roi) -> None:
    ROI_CONFIG_PATH.write_text(
        json.dumps(roi.__dict__, indent=2),
        encoding="utf-8",
    )


# -----------------------------
# Screen capture
# -----------------------------


class ScreenCapture:
    """Capture frames from the screen using MSS."""

    def __init__(self) -> None:
        self._sct = mss()

    def get_virtual_monitor(self) -> Dict[str, int]:
        """Virtual desktop monitor dict (covers all screens)."""
        return self._sct.monitors[0]

    def grab_full_bgr(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Capture virtual desktop (all monitors).
        Return: (bgr_image, (origin_left, origin_top)).
        """
        monitor = self.get_virtual_monitor()
        shot = self._sct.grab(monitor)
        img = np.array(shot)  # BGRA
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        origin = (int(monitor["left"]), int(monitor["top"]))
        return bgr, origin

    def grab_roi_bgr(self, roi: Roi) -> np.ndarray:
        shot = self._sct.grab(roi.to_mss())
        img = np.array(shot)  # BGRA
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def select_roi_interactive(capture: ScreenCapture) -> Optional[Roi]:
    """
    Select ROI on the virtual desktop screenshot.

    IMPORTANT:
    cv2.selectROI returns coordinates relative to the screenshot image (0,0).
    We must add the virtual desktop origin (monitor[0] left/top) to convert
    into absolute coordinates for mss.grab.
    """
    full_bgr, (origin_left, origin_top) = capture.grab_full_bgr()
    if full_bgr.size == 0:
        return None

    title = "Select ROI (drag white board area, ENTER/SPACE confirm, C cancel)"
    rect = cv2.selectROI(
        title,
        full_bgr,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow(title)

    x, y, w, h = rect
    roi = Roi(
        left=int(x) + origin_left,
        top=int(y) + origin_top,
        width=int(w),
        height=int(h),
    )
    if not roi.is_valid():
        return None
    return roi


# -----------------------------
# Action listener
# -----------------------------


class ActionListener:
    """Record the latest action; consume one action per frame (then reset)."""

    def __init__(self) -> None:
        self.current_action: str = ACTION_NONE
        self._listener: Optional[keyboard.Listener] = None

    def start(self) -> None:
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()

    def consume_action(self) -> str:
        action = self.current_action
        self.current_action = ACTION_NONE
        return action if action in VALID_ACTIONS else ACTION_NONE

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        try:
            if key == keyboard.Key.left:
                self.current_action = ACTION_LEFT
            elif key == keyboard.Key.right:
                self.current_action = ACTION_RIGHT
            elif key == keyboard.Key.up:
                self.current_action = ACTION_ROTATE
            elif key == keyboard.Key.down:
                self.current_action = ACTION_SOFT_DROP
            elif key == keyboard.Key.space:
                self.current_action = ACTION_HARD_DROP
        except Exception:
            return


# -----------------------------
# Grid extraction (edge energy)
# -----------------------------


class GridExtractor:
    """
    Robust occupancy detection using EDGE ENERGY (Sobel magnitude).

    This works well when landed blocks become gray and color-based methods fail.

    Output grid values:
    - 0 = empty
    - 1 = filled
    """

    def __init__(self, energy_thresh: float = 18.0) -> None:
        self.energy_thresh = float(energy_thresh)

    @staticmethod
    def _edge_magnitude(gray: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    def bgr_to_grid(self, bgr: np.ndarray) -> np.ndarray:
        if bgr.size == 0:
            return np.zeros((ROWS, COLS), dtype=np.int8)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mag = self._edge_magnitude(gray)

        h, w = gray.shape[:2]
        cell_h = h / ROWS
        cell_w = w / COLS

        grid = np.zeros((ROWS, COLS), dtype=np.int8)

        for r in range(ROWS):
            for c in range(COLS):
                y0 = int(r * cell_h)
                y1 = int((r + 1) * cell_h)
                x0 = int(c * cell_w)
                x1 = int((c + 1) * cell_w)

                cell_mag = mag[y0:y1, x0:x1]
                if cell_mag.size == 0:
                    continue

                energy = float(np.mean(cell_mag))
                grid[r, c] = 1 if energy >= self.energy_thresh else 0

        return grid

    def calibrate_from_roi(self, roi_bgr: np.ndarray) -> float:
        """
        Calibrate threshold from current ROI.
        Best time to press 'C': early game when board is mostly empty.

        Method:
        - Compute edge magnitude of ROI
        - Use a robust percentile as background edge level
        - Set threshold = bg_level + margin
        """
        if roi_bgr.size == 0:
            return self.energy_thresh

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        mag = self._edge_magnitude(gray)

        bg_level = float(np.percentile(mag, 75))
        self.energy_thresh = float(np.clip(bg_level + 8.0, 5.0, 80.0))
        return self.energy_thresh


# -----------------------------
# Landing marker (Mode A)
# -----------------------------


class LandingMarker:
    """Mode A landing: per-column landing row from occupancy grid."""

    @staticmethod
    def compute_column_landings(grid: np.ndarray) -> List[int]:
        landings: List[int] = []
        for c in range(COLS):
            landing_row = ROWS - 1
            for r in range(ROWS - 1, -1, -1):
                if grid[r, c] != 0:
                    landing_row = max(0, r - 1)
                    break
            landings.append(landing_row)
        return landings

    @staticmethod
    def choose_target_column(landings: List[int]) -> int:
        if not landings:
            return 0
        return int(np.argmax(np.array(landings, dtype=np.int32)))


# -----------------------------
# CSV recorder
# -----------------------------


class CsvRecorder:
    """Save frames and actions to CSV (human-checkable)."""

    def __init__(self, out_dir: Path, tag: str) -> None:
        self.frames_path = out_dir / f"frames_{tag}.csv"
        self.actions_path = out_dir / f"actions_{tag}.csv"

        self._frames_file = open(self.frames_path, "w", newline="", encoding="utf-8")
        self._actions_file = open(self.actions_path, "w", newline="", encoding="utf-8")

        self.frames_writer = csv.writer(self._frames_file)
        self.actions_writer = csv.writer(self._actions_file)
        self.actions_writer.writerow(["frame_idx", "timestamp", "action"])

    def close(self) -> None:
        self._frames_file.close()
        self._actions_file.close()

    def write_frame(self, frame_idx: int, grid: np.ndarray) -> None:
        self.frames_writer.writerow([f"frame={frame_idx}"])
        for r in range(ROWS):
            self.frames_writer.writerow(grid[r, :].tolist())
        self.frames_writer.writerow([])

    def write_action(self, frame_idx: int, timestamp: float, action: str) -> None:
        if action not in VALID_ACTIONS:
            action = ACTION_NONE
        self.actions_writer.writerow([frame_idx, f"{timestamp:.6f}", action])


# -----------------------------
# Visualization
# -----------------------------


class Visualizer:
    """Draw grid lines, landing markers, target column, and HUD on ROI image."""

    def __init__(self) -> None:
        self.window_name = WINDOW_NAME

    def show(
        self,
        roi_bgr: np.ndarray,
        grid: np.ndarray,
        landings: List[int],
        target_col: int,
        energy_thresh: float,
    ) -> None:
        vis = roi_bgr.copy()
        h, w = vis.shape[:2]
        cell_h = h / ROWS
        cell_w = w / COLS

        # Grid lines
        for r in range(1, ROWS):
            y = int(r * cell_h)
            cv2.line(vis, (0, y), (w, y), (90, 90, 90), 1)
        for c in range(1, COLS):
            x = int(c * cell_w)
            cv2.line(vis, (x, 0), (x, h), (90, 90, 90), 1)

        # Landing markers
        for c, lr in enumerate(landings):
            cx = int((c + 0.5) * cell_w)
            cy = int((lr + 0.5) * cell_h)
            cv2.circle(vis, (cx, cy), 4, (0, 255, 255), -1)

        # Target column highlight
        x0 = int(target_col * cell_w)
        x1 = int((target_col + 1) * cell_w)
        cv2.rectangle(vis, (x0, 0), (x1, h), (0, 0, 255), 2)

        # HUD
        filled = int(np.sum(grid))
        text = f"edge_th={energy_thresh:.1f} filled={filled}/{TOTAL_CELLS}"
        cv2.putText(
            vis,
            text,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            text,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(self.window_name, vis)


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    make_process_dpi_aware()
    ensure_out_dir()

    capture = ScreenCapture()
    roi = load_roi()

    if roi is None:
        print("ROI not found. Please select ROI (white board area inside frame).")
        roi = select_roi_interactive(capture)
        if roi is None:
            print("ROI selection cancelled/invalid. Exiting.")
            return
        save_roi(roi)
        print(f"ROI saved to: {ROI_CONFIG_PATH}")

    extractor = GridExtractor(energy_thresh=18.0)
    marker = LandingMarker()
    visualizer = Visualizer()

    tag = now_tag()
    recorder = CsvRecorder(OUTPUT_DIR, tag)

    actions = ActionListener()
    actions.start()

    frame_idx = 0
    last_t = time.time()

    try:
        while True:
            now = time.time()
            if now - last_t < FRAME_INTERVAL:
                time.sleep(0.001)
                continue
            last_t = now

            roi_bgr = capture.grab_roi_bgr(roi)
            grid = extractor.bgr_to_grid(roi_bgr)

            landings = marker.compute_column_landings(grid)
            target_col = marker.choose_target_column(landings)

            action = actions.consume_action()
            recorder.write_frame(frame_idx, grid)
            recorder.write_action(frame_idx, now, action)

            visualizer.show(
                roi_bgr=roi_bgr,
                grid=grid,
                landings=landings,
                target_col=target_col,
                energy_thresh=extractor.energy_thresh,
            )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

            if key in (ord("r"), ord("R")):
                new_roi = select_roi_interactive(capture)
                if new_roi is not None:
                    roi = new_roi
                    save_roi(roi)
                    print(f"ROI updated and saved to: {ROI_CONFIG_PATH}")

            if key in (ord("c"), ord("C")):
                th = extractor.calibrate_from_roi(roi_bgr)
                print(f"Calibrated edge energy threshold -> {th:.2f}")

            frame_idx += 1

    finally:
        actions.stop()
        recorder.close()
        cv2.destroyAllWindows()
        print(f"Saved frames CSV: {recorder.frames_path}")
        print(f"Saved actions CSV: {recorder.actions_path}")


if __name__ == "__main__":
    main()