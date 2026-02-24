"""
Task 1 - Real Image Capture for Tetris (.exe) [FINAL - Per-frame CSV + Center Patch Overlay]

What it does:
- Capture the real Tetris .exe board from screen (ROI).
- Convert each frame into an 18x14 binary grid and save as CSV (one file per frame).
- Record one action per frame (CSV), aligned with frames.
- Show a GUI window:
  - ROI screenshot
  - 18x14 grid lines
  - Outline detected occupied cells (grid==1) in green (tight to real block size)
  - NO yellow dots, NO red target column (hidden)

Output:
- Frames: out_task1_real/frames_<tag>/frame_000000.csv, frame_000001.csv, ...
- Actions: out_task1_real/actions_<tag>.csv

Controls:
- Q: Quit
- R: Re-select ROI (overwrite roi_config.json)
- C: Calibrate background + thresholds from current ROI (best at early game / empty board)

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
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from mss import mss
from pynput import keyboard

# -----------------------------
# Configuration
# -----------------------------

ROWS = 18
COLS = 14
TOTAL_CELLS = ROWS * COLS

FPS = 15
FRAME_INTERVAL = 1.0 / FPS

OUTPUT_DIR = Path("out_task1_real")
ROI_CONFIG_PATH = OUTPUT_DIR / "roi_config.json"

WINDOW_NAME = "Task1 Capture (Q quit | R reselect ROI | C calibrate)"

# Actions
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
    cv2.selectROI returns coordinates relative to screenshot image (0,0).
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
# Grid extraction (center patch)
# -----------------------------


class GridExtractor:
    """
    Cell-center detection (tight to block size, no neighbor expansion).

    For each cell, analyze ONLY a center patch (avoid borders/edges).
    Filled if:
      - abs(mean_gray - bg_gray) >= diff_thresh
        OR
      - std_gray >= std_thresh

    Press 'C' to calibrate bg_gray and thresholds from current ROI.
    Best time: early game when the board is mostly empty.
    """

    def __init__(self) -> None:
        self.bg_gray = 245.0
        self.diff_thresh = 18.0
        self.std_thresh = 10.0
        self.center_ratio = 0.25  # use middle 50% area

    def bgr_to_grid(self, bgr: np.ndarray) -> np.ndarray:
        if bgr.size == 0:
            return np.zeros((ROWS, COLS), dtype=np.int8)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

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

                ph = int((y1 - y0) * self.center_ratio)
                pw = int((x1 - x0) * self.center_ratio)

                cy0 = y0 + ph
                cy1 = y1 - ph
                cx0 = x0 + pw
                cx1 = x1 - pw

                patch = gray[cy0:cy1, cx0:cx1]
                if patch.size == 0:
                    continue

                m = float(np.mean(patch))
                s = float(np.std(patch))

                if abs(m - self.bg_gray) >= self.diff_thresh or s >= self.std_thresh:
                    grid[r, c] = 1

        return grid

    def calibrate_from_roi(self, roi_bgr: np.ndarray) -> None:
        """Calibrate bg_gray and thresholds from current ROI (mostly empty board)."""
        if roi_bgr.size == 0:
            return

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape[:2]

        ph = max(12, h // 12)
        pw = max(12, w // 12)
        coords = [
            (0, 0),
            (0, w - pw),
            (h - ph, 0),
            (h - ph, w - pw),
            (0, w // 2 - pw // 2),
            (h - ph, w // 2 - pw // 2),
        ]

        samples = []
        for y, x in coords:
            y0 = int(np.clip(y, 0, h - 1))
            x0 = int(np.clip(x, 0, w - 1))
            y1 = int(np.clip(y0 + ph, 0, h))
            x1 = int(np.clip(x0 + pw, 0, w))
            p = gray[y0:y1, x0:x1]
            if p.size > 0:
                samples.append(p.reshape(-1))

        if not samples:
            return

        bg_pixels = np.concatenate(samples, axis=0)
        self.bg_gray = float(np.mean(bg_pixels))
        bg_std = float(np.std(bg_pixels))

        self.diff_thresh = float(np.clip(bg_std * 3.0 + 8.0, 10.0, 40.0))
        self.std_thresh = float(np.clip(bg_std * 2.0 + 6.0, 6.0, 25.0))


# -----------------------------
# CSV recorder (PER-FRAME)
# -----------------------------


class CsvRecorder:
    """
    Save frames as ONE CSV PER FRAME (18 rows x 14 cols),
    and actions as a single CSV aligned by frame_idx.
    """

    def __init__(self, out_dir: Path, tag: str) -> None:
        self.out_dir = out_dir
        self.tag = tag

        # Frames directory: out_task1_real/frames_<tag>/
        self.frames_dir = out_dir / f"frames_{tag}"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # Actions file: out_task1_real/actions_<tag>.csv
        self.actions_path = out_dir / f"actions_{tag}.csv"
        self._actions_file = open(self.actions_path, "w", newline="", encoding="utf-8")
        self.actions_writer = csv.writer(self._actions_file)
        self.actions_writer.writerow(["frame_idx", "timestamp", "action"])

    def close(self) -> None:
        self._actions_file.close()

    def write_frame(self, frame_idx: int, grid: np.ndarray) -> None:
        """
        Write one frame to:
          frames_<tag>/frame_000000.csv
        Content:
          18 rows, each with 14 values (0/1)
        """
        frame_path = self.frames_dir / f"frame_{frame_idx:06d}.csv"
        with open(frame_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for r in range(ROWS):
                w.writerow(grid[r, :].tolist())

    def write_action(self, frame_idx: int, timestamp: float, action: str) -> None:
        if action not in VALID_ACTIONS:
            action = ACTION_NONE
        self.actions_writer.writerow([frame_idx, f"{timestamp:.6f}", action])


# -----------------------------
# Visualization
# -----------------------------


class Visualizer:
    """Show ROI with grid lines and outline detected occupied cells (grid==1)."""

    def __init__(self) -> None:
        self.window_name = WINDOW_NAME

    def show(
        self,
        roi_bgr: np.ndarray,
        grid: np.ndarray,
        bg_gray: float,
        diff_th: float,
        std_th: float,
    ) -> None:
        vis = roi_bgr.copy()
        h, w = vis.shape[:2]
        cell_h = h / ROWS
        cell_w = w / COLS

        # Outline occupied cells (tight)
        for r in range(ROWS):
            for c in range(COLS):
                if grid[r, c] == 0:
                    continue
                x0 = int(c * cell_w)
                y0 = int(r * cell_h)
                x1 = int((c + 1) * cell_w)
                y1 = int((r + 1) * cell_h)
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)

        # Grid lines
        for r in range(1, ROWS):
            y = int(r * cell_h)
            cv2.line(vis, (0, y), (w, y), (90, 90, 90), 1)
        for c in range(1, COLS):
            x = int(c * cell_w)
            cv2.line(vis, (x, 0), (x, h), (90, 90, 90), 1)

        filled = int(np.sum(grid))
        text = (
            f"bg={bg_gray:.0f} diff={diff_th:.0f} std={std_th:.0f} "
            f"filled={filled}/{TOTAL_CELLS}"
        )
        cv2.putText(
            vis,
            text,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            text,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
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

    extractor = GridExtractor()
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

            action = actions.consume_action()
            recorder.write_frame(frame_idx, grid)
            recorder.write_action(frame_idx, now, action)

            visualizer.show(
                roi_bgr=roi_bgr,
                grid=grid,
                bg_gray=extractor.bg_gray,
                diff_th=extractor.diff_thresh,
                std_th=extractor.std_thresh,
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
                extractor.calibrate_from_roi(roi_bgr)
                print(
                    "Calibrated: "
                    f"bg_gray={extractor.bg_gray:.1f}, "
                    f"diff_th={extractor.diff_thresh:.1f}, "
                    f"std_th={extractor.std_thresh:.1f}"
                )

            frame_idx += 1

    finally:
        actions.stop()
        recorder.close()
        cv2.destroyAllWindows()
        print(f"Saved frames dir: {recorder.frames_dir}")
        print(f"Saved actions CSV: {recorder.actions_path}")


if __name__ == "__main__":
    main()
