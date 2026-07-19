"""Task8 trajectory selection program.

用途：
- 从自我对局或已有模拟数据中筛选更优轨迹。
- 按存活时间、消行能力、棋盘稳定性挑选训练样本。
- 生成 task8_selfimit_train_v6.py 使用的精英训练数据目录。

这个文件负责“准备训练集”，不直接输出最终可接入游戏的 .pt 模型。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
# Constants
# ============================================================
BOARD_ROWS = 18
BOARD_COLS = 14
ACTION_CLASSES = 56

STOP_REASON_TO_CODE = {
    "running": 0,
    "top_out": 1,
    "policy_fail": 2,
    "max_pieces_cap": 3,
}
CODE_TO_STOP_REASON = {v: k for k, v in STOP_REASON_TO_CODE.items()}

STAGE_EARLY = 0
STAGE_MID = 1
STAGE_LATE = 2


# ============================================================
# Helpers
# ============================================================
def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size > 0 else 0.0


def safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size > 0 else 0.0


def safe_quantile(x: np.ndarray, q: float) -> float:
    if x.size <= 0:
        return 0.0
    q = max(0.0, min(1.0, float(q)))
    return float(np.quantile(x, q))


def summarize_episode_metrics(lines: np.ndarray, pieces: np.ndarray) -> Dict[str, float]:
    def rate(threshold: int) -> float:
        return float(np.mean(lines >= threshold)) if lines.size > 0 else 0.0

    return {
        "episodes": float(lines.size),
        "mean_lines": safe_mean(lines),
        "median_lines": float(np.median(lines)) if lines.size > 0 else 0.0,
        "std_lines": safe_std(lines),
        "min_lines": float(np.min(lines)) if lines.size > 0 else 0.0,
        "max_lines": float(np.max(lines)) if lines.size > 0 else 0.0,
        "mean_pieces": safe_mean(pieces),
        "median_pieces": float(np.median(pieces)) if pieces.size > 0 else 0.0,
        "s10": rate(10),
        "s20": rate(20),
        "s30": rate(30),
        "s40": rate(40),
        "s50": rate(50),
        "early_death_lt5": float(np.mean(lines < 5)) if lines.size > 0 else 0.0,
    }


def build_episode_split(
    episode_ids: List[int],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[int], List[int], List[int]]:
    if len(episode_ids) < 3:
        raise ValueError(f"selected episodes too few: {len(episode_ids)}")

    ids = list(episode_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_val > 1:
            n_val -= 1
        else:
            n_train -= 1

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Board metrics
# ============================================================
def count_holes(board_occ: np.ndarray) -> int:
    holes = 0
    for col in range(BOARD_COLS):
        seen_filled = False
        for row in range(BOARD_ROWS):
            if int(board_occ[row, col]) != 0:
                seen_filled = True
            elif seen_filled:
                holes += 1
    return holes


def count_row_transitions(board_occ: np.ndarray) -> int:
    transitions = 0
    for row in range(BOARD_ROWS):
        prev_filled = 1
        for col in range(BOARD_COLS):
            cur_filled = 1 if int(board_occ[row, col]) != 0 else 0
            if cur_filled != prev_filled:
                transitions += 1
            prev_filled = cur_filled
        if prev_filled != 1:
            transitions += 1
    return transitions


def count_col_transitions(board_occ: np.ndarray) -> int:
    transitions = 0
    for col in range(BOARD_COLS):
        prev_filled = 1
        for row in range(BOARD_ROWS):
            cur_filled = 1 if int(board_occ[row, col]) != 0 else 0
            if cur_filled != prev_filled:
                transitions += 1
            prev_filled = cur_filled
        if prev_filled != 1:
            transitions += 1
    return transitions


def count_well_sums(board_occ: np.ndarray) -> int:
    total = 0
    for col in range(BOARD_COLS):
        depth = 0
        for row in range(BOARD_ROWS):
            if int(board_occ[row, col]) != 0:
                depth = 0
                continue

            left_filled = col == 0 or int(board_occ[row, col - 1]) != 0
            right_filled = col == BOARD_COLS - 1 or int(board_occ[row, col + 1]) != 0

            if left_filled and right_filled:
                depth += 1
                total += depth
            else:
                depth = 0
    return total


def column_heights(board_occ: np.ndarray) -> List[int]:
    heights: List[int] = []
    for col in range(BOARD_COLS):
        h = 0
        for row in range(BOARD_ROWS):
            if int(board_occ[row, col]) != 0:
                h = BOARD_ROWS - row
                break
        heights.append(h)
    return heights


def max_height(board_occ: np.ndarray) -> int:
    heights = column_heights(board_occ)
    return max(heights) if heights else 0


def roughness(board_occ: np.ndarray) -> int:
    heights = column_heights(board_occ)
    if len(heights) <= 1:
        return 0
    return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))


def board_metrics(board_occ: np.ndarray) -> Dict[str, int]:
    return {
        "holes": count_holes(board_occ),
        "row_transitions": count_row_transitions(board_occ),
        "col_transitions": count_col_transitions(board_occ),
        "well_sums": count_well_sums(board_occ),
        "max_height": max_height(board_occ),
        "roughness": roughness(board_occ),
    }


# ============================================================
# Data records
# ============================================================
@dataclass
class EpisodeInfo:
    episode_index: int
    seed: int
    final_lines: int
    final_pieces: int
    final_score: int
    stop_reason_code: int
    sample_count: int
    episode_score: float
    candidate_flag: int
    strong_flag: int
    selected_sample_count: int = 0
    selected_core_count: int = 0
    selected_context_count: int = 0
    selected_sustain_aux_count: int = 0
    selected_tail_aux_count: int = 0
    selected_fail_repair_aux_count: int = 0


# ============================================================
# Selector
# ============================================================
class TrajectorySelectorV6:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.npz_path = Path(args.selfplay_npz)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"selfplay npz not found: {self.npz_path}")

        self.episode_csv_path = Path(args.episode_csv) if args.episode_csv else None
        self.out_root = Path(args.out_dir)
        self.out_dir = self.out_root / args.tag
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.data = dict(np.load(self.npz_path, allow_pickle=False))
        self.n_samples = int(len(self.data["episode_index"]))

        required_keys = [
            "boards",
            "actives",
            "legals",
            "pieces",
            "targets",
            "episode_index",
            "seed",
            "piece_index",
            "lines_gain_immediate",
            "episode_final_lines",
            "episode_final_pieces",
            "episode_final_score",
            "stop_reason_code",
            "holes_after",
            "row_transitions_after",
            "col_transitions_after",
            "well_sums_after",
            "max_height_after",
            "roughness_after",
        ]
        for key in required_keys:
            if key not in self.data:
                raise KeyError(f"missing key in selfplay npz: {key}")

        self.episode_rows: List[EpisodeInfo] = []
        self.episode_map: Dict[int, EpisodeInfo] = {}

        self.selected_sample_mask = np.zeros(self.n_samples, dtype=bool)
        self.selected_core_mask = np.zeros(self.n_samples, dtype=bool)
        self.selected_context_mask = np.zeros(self.n_samples, dtype=bool)
        self.selected_sustain_aux_mask = np.zeros(self.n_samples, dtype=bool)
        self.selected_tail_aux_mask = np.zeros(self.n_samples, dtype=bool)
        self.selected_fail_repair_aux_mask = np.zeros(self.n_samples, dtype=bool)
        self.selected_fallback_mask = np.zeros(self.n_samples, dtype=bool)

        self.holes_before: np.ndarray
        self.row_transitions_before: np.ndarray
        self.col_transitions_before: np.ndarray
        self.well_sums_before: np.ndarray
        self.max_height_before: np.ndarray
        self.roughness_before: np.ndarray

        self.delta_holes: np.ndarray
        self.delta_row_transitions: np.ndarray
        self.delta_col_transitions: np.ndarray
        self.delta_well_sums: np.ndarray
        self.delta_max_height: np.ndarray
        self.delta_roughness: np.ndarray

        self.risk_before: np.ndarray
        self.risk_after: np.ndarray
        self.structure_gain: np.ndarray

        self.future_risk_mean: np.ndarray
        self.future_lines_sum: np.ndarray
        self.future_topout_soon: np.ndarray
        self.steps_to_end: np.ndarray

        self.stage_bucket: np.ndarray = np.zeros(self.n_samples, dtype=np.int8)

        self.is_repair_sample: np.ndarray
        self.is_rescue_sample: np.ndarray
        self.is_stable_sample: np.ndarray
        self.is_healthy_clear_sample: np.ndarray
        self.is_bad_structure_sample: np.ndarray
        self.is_sustain_sample: np.ndarray
        self.is_tail_candidate: np.ndarray
        self.is_fail_repair_candidate: np.ndarray = np.zeros(self.n_samples, dtype=np.int8)
        self.is_candidate_episode_sample: np.ndarray
        self.is_strong_episode_sample: np.ndarray

        self.sample_weight_v6: np.ndarray

    def run(self) -> None:
        self.build_before_metrics()
        self.build_episode_infos()
        self.build_future_metrics()
        self.build_structural_flags()
        self.select_samples()
        self.save_outputs()
        self.print_summary()

    def build_before_metrics(self) -> None:
        boards = self.data["boards"].astype(np.int8)

        n = boards.shape[0]
        self.holes_before = np.zeros(n, dtype=np.int16)
        self.row_transitions_before = np.zeros(n, dtype=np.int16)
        self.col_transitions_before = np.zeros(n, dtype=np.int16)
        self.well_sums_before = np.zeros(n, dtype=np.int16)
        self.max_height_before = np.zeros(n, dtype=np.int16)
        self.roughness_before = np.zeros(n, dtype=np.int16)

        for i in range(n):
            m = board_metrics(boards[i])
            self.holes_before[i] = int(m["holes"])
            self.row_transitions_before[i] = int(m["row_transitions"])
            self.col_transitions_before[i] = int(m["col_transitions"])
            self.well_sums_before[i] = int(m["well_sums"])
            self.max_height_before[i] = int(m["max_height"])
            self.roughness_before[i] = int(m["roughness"])

            if (i + 1) % 5000 == 0:
                print(f"[METRICS] computed before metrics: {i + 1}/{n}")

        self.delta_holes = self.data["holes_after"].astype(np.int16) - self.holes_before
        self.delta_row_transitions = self.data["row_transitions_after"].astype(np.int16) - self.row_transitions_before
        self.delta_col_transitions = self.data["col_transitions_after"].astype(np.int16) - self.col_transitions_before
        self.delta_well_sums = self.data["well_sums_after"].astype(np.int16) - self.well_sums_before
        self.delta_max_height = self.data["max_height_after"].astype(np.int16) - self.max_height_before
        self.delta_roughness = self.data["roughness_after"].astype(np.int16) - self.roughness_before

        self.risk_before = (
            self.args.risk_holes_coef * self.holes_before.astype(np.float32)
            + self.args.risk_height_coef * self.max_height_before.astype(np.float32)
            + self.args.risk_rough_coef * self.roughness_before.astype(np.float32)
            + self.args.risk_row_trans_coef * self.row_transitions_before.astype(np.float32)
            + self.args.risk_col_trans_coef * self.col_transitions_before.astype(np.float32)
        )

        self.risk_after = (
            self.args.risk_holes_coef * self.data["holes_after"].astype(np.float32)
            + self.args.risk_height_coef * self.data["max_height_after"].astype(np.float32)
            + self.args.risk_rough_coef * self.data["roughness_after"].astype(np.float32)
            + self.args.risk_row_trans_coef * self.data["row_transitions_after"].astype(np.float32)
            + self.args.risk_col_trans_coef * self.data["col_transitions_after"].astype(np.float32)
        )

        self.structure_gain = (
            -self.args.gain_holes_coef * self.delta_holes.astype(np.float32)
            -self.args.gain_height_coef * self.delta_max_height.astype(np.float32)
            -self.args.gain_rough_coef * self.delta_roughness.astype(np.float32)
            -self.args.gain_row_trans_coef * self.delta_row_transitions.astype(np.float32)
            -self.args.gain_col_trans_coef * self.delta_col_transitions.astype(np.float32)
            + self.args.gain_clear_coef * self.data["lines_gain_immediate"].astype(np.float32)
        )

    def build_episode_infos(self) -> None:
        ep_ids = self.data["episode_index"].astype(np.int32)
        seeds = self.data["seed"].astype(np.int32)
        final_lines = self.data["episode_final_lines"].astype(np.int32)
        final_pieces = self.data["episode_final_pieces"].astype(np.int32)
        final_scores = self.data["episode_final_score"].astype(np.int32)
        stop_codes = self.data["stop_reason_code"].astype(np.int32)

        unique_eps = sorted(np.unique(ep_ids).tolist())
        per_ep_lines = []
        per_ep_pieces = []
        per_ep_scores = []

        tmp_rows: List[EpisodeInfo] = []

        for ep in unique_eps:
            idx = np.where(ep_ids == ep)[0]
            if idx.size <= 0:
                continue

            seed = int(seeds[idx[0]])
            lines = int(final_lines[idx[0]])
            pieces = int(final_pieces[idx[0]])
            score = int(final_scores[idx[0]])
            stop_code = int(stop_codes[idx[0]])
            sample_count = int(idx.size)

            episode_score = (
                self.args.episode_line_weight * float(lines)
                + self.args.episode_piece_weight * float(pieces)
            )

            row = EpisodeInfo(
                episode_index=int(ep),
                seed=seed,
                final_lines=lines,
                final_pieces=pieces,
                final_score=score,
                stop_reason_code=stop_code,
                sample_count=sample_count,
                episode_score=episode_score,
                candidate_flag=0,
                strong_flag=0,
            )
            tmp_rows.append(row)

            per_ep_lines.append(lines)
            per_ep_pieces.append(pieces)
            per_ep_scores.append(episode_score)

        lines_arr = np.asarray(per_ep_lines, dtype=np.float32)
        pieces_arr = np.asarray(per_ep_pieces, dtype=np.float32)
        scores_arr = np.asarray(per_ep_scores, dtype=np.float32)

        self.base_summary = summarize_episode_metrics(lines_arr, pieces_arr)

        candidate_score_thresh = safe_quantile(scores_arr, 1.0 - self.args.candidate_top_pct)
        candidate_lines_thresh = max(
            float(self.args.candidate_lines_floor),
            safe_quantile(lines_arr, 1.0 - self.args.candidate_top_pct),
        )
        candidate_pieces_thresh = max(
            float(self.args.candidate_pieces_floor),
            safe_quantile(pieces_arr, 1.0 - self.args.candidate_top_pct),
        )

        strong_score_thresh = safe_quantile(scores_arr, 1.0 - self.args.strong_top_pct)
        strong_lines_thresh = max(
            float(self.args.strong_lines_floor),
            safe_quantile(lines_arr, 1.0 - self.args.strong_top_pct),
        )
        strong_pieces_thresh = max(
            float(self.args.strong_pieces_floor),
            safe_quantile(pieces_arr, 1.0 - self.args.strong_top_pct),
        )

        self.thresholds = {
            "candidate_score_thresh": float(candidate_score_thresh),
            "candidate_lines_thresh": float(candidate_lines_thresh),
            "candidate_pieces_thresh": float(candidate_pieces_thresh),
            "strong_score_thresh": float(strong_score_thresh),
            "strong_lines_thresh": float(strong_lines_thresh),
            "strong_pieces_thresh": float(strong_pieces_thresh),
        }

        for row in tmp_rows:
            candidate_flag = (
                (row.episode_score >= candidate_score_thresh)
                or (row.final_lines >= candidate_lines_thresh)
                or (row.final_pieces >= candidate_pieces_thresh)
            )
            strong_flag = (
                (row.episode_score >= strong_score_thresh)
                or (row.final_lines >= strong_lines_thresh)
                or (row.final_pieces >= strong_pieces_thresh)
            )

            row.candidate_flag = int(bool(candidate_flag))
            row.strong_flag = int(bool(strong_flag))

            self.episode_rows.append(row)
            self.episode_map[row.episode_index] = row

        ep_candidate = np.zeros(self.n_samples, dtype=np.int8)
        ep_strong = np.zeros(self.n_samples, dtype=np.int8)
        for row in self.episode_rows:
            idx = np.where(ep_ids == row.episode_index)[0]
            ep_candidate[idx] = row.candidate_flag
            ep_strong[idx] = row.strong_flag

        self.is_candidate_episode_sample = ep_candidate
        self.is_strong_episode_sample = ep_strong

    def build_future_metrics(self) -> None:
        ep_ids = self.data["episode_index"].astype(np.int32)
        piece_index = self.data["piece_index"].astype(np.int32)
        stop_codes = self.data["stop_reason_code"].astype(np.int32)
        final_pieces = self.data["episode_final_pieces"].astype(np.int32)
        lines_gain = self.data["lines_gain_immediate"].astype(np.int32)

        n = self.n_samples
        self.future_risk_mean = np.full(n, 999.0, dtype=np.float32)
        self.future_lines_sum = np.zeros(n, dtype=np.float32)
        self.future_topout_soon = np.zeros(n, dtype=np.int8)
        self.steps_to_end = np.zeros(n, dtype=np.int16)

        unique_eps = sorted(np.unique(ep_ids).tolist())
        horizon = max(1, int(self.args.future_horizon))

        for ep in unique_eps:
            idx = np.where(ep_ids == ep)[0]
            if idx.size <= 0:
                continue

            idx_sorted = idx[np.argsort(piece_index[idx])]
            ep_stop_code = int(stop_codes[idx_sorted[0]])
            ep_final_pieces = int(final_pieces[idx_sorted[0]])

            for pos, sample_i in enumerate(idx_sorted.tolist()):
                remaining_steps = max(0, ep_final_pieces - int(piece_index[sample_i]))
                self.steps_to_end[sample_i] = int(remaining_steps)

                future_idx = idx_sorted[pos + 1: pos + 1 + horizon]
                if future_idx.size > 0:
                    self.future_risk_mean[sample_i] = safe_mean(self.risk_before[future_idx])
                    self.future_lines_sum[sample_i] = float(np.sum(lines_gain[future_idx]))
                else:
                    self.future_risk_mean[sample_i] = float(self.risk_after[sample_i])
                    self.future_lines_sum[sample_i] = 0.0

                if ep_stop_code == STOP_REASON_TO_CODE["top_out"] and remaining_steps <= horizon:
                    self.future_topout_soon[sample_i] = 1
                else:
                    self.future_topout_soon[sample_i] = 0

    def build_structural_flags(self) -> None:
        candidate_mask = self.is_candidate_episode_sample > 0
        if candidate_mask.sum() <= 0:
            candidate_mask = np.ones(self.n_samples, dtype=bool)

        risk_pool = self.risk_before[candidate_mask]
        gain_pool = self.structure_gain[candidate_mask]
        future_risk_pool = self.future_risk_mean[candidate_mask]
        future_lines_pool = self.future_lines_sum[candidate_mask]

        future_risk_pool = future_risk_pool[np.isfinite(future_risk_pool)]
        future_risk_pool = future_risk_pool[future_risk_pool < 900.0]

        rescue_risk_thresh = max(
            float(self.args.rescue_risk_floor),
            safe_quantile(risk_pool, self.args.rescue_risk_quantile),
        )
        stable_risk_max = safe_quantile(risk_pool, self.args.stable_risk_quantile)
        repair_gain_thresh = max(
            float(self.args.repair_gain_floor),
            safe_quantile(gain_pool, self.args.repair_gain_quantile),
        )
        rescue_gain_thresh = max(
            float(self.args.rescue_gain_floor),
            safe_quantile(gain_pool, self.args.rescue_gain_quantile),
        )

        future_risk_thresh = max(
            float(self.args.future_risk_floor),
            safe_quantile(future_risk_pool, self.args.future_risk_quantile),
        )
        tail_future_lines_thresh = max(
            float(self.args.tail_future_lines_floor),
            safe_quantile(future_lines_pool, self.args.tail_future_lines_quantile),
        )

        self.thresholds.update(
            {
                "rescue_risk_thresh": float(rescue_risk_thresh),
                "stable_risk_max": float(stable_risk_max),
                "repair_gain_thresh": float(repair_gain_thresh),
                "rescue_gain_thresh": float(rescue_gain_thresh),
                "future_risk_thresh": float(future_risk_thresh),
                "tail_future_lines_thresh": float(tail_future_lines_thresh),
            }
        )

        self.is_repair_sample = (
            (self.delta_holes <= self.args.repair_delta_holes_max)
            | (self.structure_gain >= repair_gain_thresh)
        ).astype(np.int8)

        self.is_rescue_sample = (
            (self.risk_before >= rescue_risk_thresh)
            & (self.structure_gain >= rescue_gain_thresh)
            & (self.delta_holes <= self.args.rescue_delta_holes_max)
            & (self.delta_max_height <= self.args.rescue_delta_height_max)
        ).astype(np.int8)

        self.is_stable_sample = (
            (self.risk_before <= stable_risk_max)
            & (self.delta_holes <= self.args.stable_delta_holes_max)
            & (self.delta_max_height <= self.args.stable_delta_height_max)
            & (self.delta_roughness <= self.args.stable_delta_rough_max)
            & (self.delta_row_transitions <= self.args.stable_delta_row_trans_max)
            & (self.delta_col_transitions <= self.args.stable_delta_col_trans_max)
        ).astype(np.int8)

        self.is_healthy_clear_sample = (
            (self.data["lines_gain_immediate"].astype(np.int16) > 0)
            & (self.delta_holes <= self.args.healthy_clear_delta_holes_max)
            & (self.delta_max_height <= self.args.healthy_clear_delta_height_max)
            & (self.delta_roughness <= self.args.healthy_clear_delta_rough_max)
            & (self.delta_row_transitions <= self.args.healthy_clear_delta_row_trans_max)
            & (self.delta_col_transitions <= self.args.healthy_clear_delta_col_trans_max)
        ).astype(np.int8)

        self.is_bad_structure_sample = (
            (self.delta_holes >= self.args.bad_delta_holes_min)
            | (self.delta_max_height >= self.args.bad_delta_height_min)
            | (self.delta_roughness >= self.args.bad_delta_rough_min)
            | (
                (self.risk_before >= rescue_risk_thresh)
                & (self.structure_gain < self.args.bad_structure_gain_max)
            )
        ).astype(np.int8)

        sustain_risk_max = max(
            float(self.args.sustain_risk_floor),
            float(stable_risk_max + self.args.sustain_risk_margin),
        )

        self.is_sustain_sample = (
            (self.is_bad_structure_sample == 0)
            & (self.risk_after <= sustain_risk_max)
            & (self.future_risk_mean <= future_risk_thresh)
            & (self.steps_to_end >= self.args.sustain_min_steps_to_end)
            & (self.future_topout_soon == 0)
            & (
                (self.is_stable_sample > 0)
                | (self.is_healthy_clear_sample > 0)
                | (
                    (self.is_repair_sample > 0)
                    & (self.risk_after <= rescue_risk_thresh)
                )
            )
        ).astype(np.int8)

        self.is_tail_candidate = (
            (self.is_sustain_sample > 0)
            & (self.is_strong_episode_sample > 0)
            & (self.future_lines_sum >= tail_future_lines_thresh)
        ).astype(np.int8)

    def _build_stage_masks(
        self,
        local_piece_index: np.ndarray,
        final_pieces: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        denom = max(int(final_pieces), 1)
        frac = local_piece_index.astype(np.float32) / float(denom)

        early_mask = frac <= self.args.stage_early_end_ratio
        late_mask = frac >= self.args.stage_late_start_ratio
        mid_mask = (~early_mask) & (~late_mask)
        return early_mask, mid_mask, late_mask

    def _rank_positions(
        self,
        sample_idx: np.ndarray,
        positions: np.ndarray,
        prefer_high_risk: bool = False,
    ) -> np.ndarray:
        if positions.size <= 0:
            return positions

        risk_local = self.risk_before[sample_idx][positions]
        gain_local = self.structure_gain[sample_idx][positions]
        future_risk_local = self.future_risk_mean[sample_idx][positions]

        if prefer_high_risk:
            order = np.lexsort((
                future_risk_local,
                -gain_local,
                -risk_local,
            ))
        else:
            order = np.lexsort((
                future_risk_local,
                -gain_local,
                risk_local,
            ))
        return positions[order]

    def _trim_late_overflow(
        self,
        context_local: np.ndarray,
        sustain_aux_local: np.ndarray,
        tail_aux_local: np.ndarray,
        fail_repair_aux_local: np.ndarray,
        late_mask: np.ndarray,
        sample_idx: np.ndarray,
        late_overflow: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if late_overflow <= 0:
            return context_local, sustain_aux_local, tail_aux_local, fail_repair_aux_local

        def drop_from(mask: np.ndarray, need_drop: int) -> int:
            if need_drop <= 0:
                return 0
            cand = np.where(mask & late_mask)[0]
            if cand.size <= 0:
                return 0
            ranked = self._rank_positions(sample_idx, cand, prefer_high_risk=False)
            drop_n = min(need_drop, len(ranked))
            mask[ranked[:drop_n]] = False
            return drop_n

        need = late_overflow
        need -= drop_from(tail_aux_local, need)
        need -= drop_from(sustain_aux_local, need)
        need -= drop_from(context_local, need)
        need -= drop_from(fail_repair_aux_local, need)
        return context_local, sustain_aux_local, tail_aux_local, fail_repair_aux_local

    def _add_stage_support(
        self,
        selected_local: np.ndarray,
        target_need: int,
        stage_mask: np.ndarray,
        bad_mask: np.ndarray,
        sample_idx: np.ndarray,
        structural_pool: np.ndarray,
        context_local: np.ndarray,
    ) -> np.ndarray:
        if target_need <= 0:
            return context_local

        pool = (~selected_local) & (~bad_mask) & stage_mask & structural_pool
        cand = np.where(pool)[0]
        ranked = self._rank_positions(sample_idx, cand, prefer_high_risk=True)

        added = 0
        for pos in ranked.tolist():
            context_local[pos] = True
            added += 1
            if added >= target_need:
                break

        if added < target_need:
            loose_pool = (~selected_local) & (~bad_mask) & stage_mask & (self.structure_gain[sample_idx] >= self.args.stage_balance_gain_floor)
            cand2 = np.where(loose_pool & (~context_local))[0]
            ranked2 = self._rank_positions(sample_idx, cand2, prefer_high_risk=True)
            for pos in ranked2.tolist():
                context_local[pos] = True
                added += 1
                if added >= target_need:
                    break

        return context_local

    def _rebalance_stage_mix(
        self,
        sample_idx: np.ndarray,
        row: EpisodeInfo,
        core_local: np.ndarray,
        context_local: np.ndarray,
        sustain_aux_local: np.ndarray,
        tail_aux_local: np.ndarray,
        fail_repair_aux_local: np.ndarray,
        early_mask: np.ndarray,
        mid_mask: np.ndarray,
        late_mask: np.ndarray,
        bad_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        structural_pool = (
            (self.is_repair_sample[sample_idx] > 0)
            | (self.is_rescue_sample[sample_idx] > 0)
            | (self.is_stable_sample[sample_idx] > 0)
            | (self.is_healthy_clear_sample[sample_idx] > 0)
            | (self.is_fail_repair_candidate[sample_idx] > 0)
        )

        selected_local = core_local | context_local | sustain_aux_local | tail_aux_local | fail_repair_aux_local
        base_total = int(selected_local.sum())
        if base_total <= 0:
            return context_local, sustain_aux_local, tail_aux_local, fail_repair_aux_local

        early_min = int(math.ceil(base_total * self.args.stage_early_min_ratio))
        mid_min = int(math.ceil(base_total * self.args.stage_mid_min_ratio))
        late_max = int(math.floor(base_total * self.args.stage_late_max_ratio))

        late_count = int(np.sum(selected_local & late_mask))
        late_overflow = max(0, late_count - late_max)
        context_local, sustain_aux_local, tail_aux_local, fail_repair_aux_local = self._trim_late_overflow(
            context_local=context_local,
            sustain_aux_local=sustain_aux_local,
            tail_aux_local=tail_aux_local,
            fail_repair_aux_local=fail_repair_aux_local,
            late_mask=late_mask,
            sample_idx=sample_idx,
            late_overflow=late_overflow,
        )

        selected_local = core_local | context_local | sustain_aux_local | tail_aux_local | fail_repair_aux_local
        early_count = int(np.sum(selected_local & early_mask))
        mid_count = int(np.sum(selected_local & mid_mask))

        if early_count < early_min:
            context_local = self._add_stage_support(
                selected_local=selected_local,
                target_need=early_min - early_count,
                stage_mask=early_mask,
                bad_mask=bad_mask,
                sample_idx=sample_idx,
                structural_pool=structural_pool,
                context_local=context_local,
            )
            selected_local = core_local | context_local | sustain_aux_local | tail_aux_local | fail_repair_aux_local

        if mid_count < mid_min:
            context_local = self._add_stage_support(
                selected_local=selected_local,
                target_need=mid_min - mid_count,
                stage_mask=mid_mask,
                bad_mask=bad_mask,
                sample_idx=sample_idx,
                structural_pool=structural_pool,
                context_local=context_local,
            )

        return context_local, sustain_aux_local, tail_aux_local, fail_repair_aux_local

    def select_samples(self) -> None:
        ep_ids = self.data["episode_index"].astype(np.int32)
        piece_index = self.data["piece_index"].astype(np.int32)

        core_global = np.zeros(self.n_samples, dtype=bool)
        context_global = np.zeros(self.n_samples, dtype=bool)
        sustain_aux_global = np.zeros(self.n_samples, dtype=bool)
        tail_aux_global = np.zeros(self.n_samples, dtype=bool)
        fail_repair_aux_global = np.zeros(self.n_samples, dtype=bool)
        fallback_global = np.zeros(self.n_samples, dtype=bool)

        for row in self.episode_rows:
            ep = row.episode_index
            sample_idx = np.where(ep_ids == ep)[0]
            if sample_idx.size <= 0:
                continue

            is_fail_episode = (
                row.final_lines < self.args.fail_repair_episode_lines_max
                and row.stop_reason_code == STOP_REASON_TO_CODE["top_out"]
            )

            if row.candidate_flag != 1 and not is_fail_episode:
                continue

            local_piece_idx = piece_index[sample_idx]
            early_mask, mid_mask, late_mask = self._build_stage_masks(
                local_piece_index=local_piece_idx,
                final_pieces=row.final_pieces,
            )

            self.stage_bucket[sample_idx[early_mask]] = STAGE_EARLY
            self.stage_bucket[sample_idx[mid_mask]] = STAGE_MID
            self.stage_bucket[sample_idx[late_mask]] = STAGE_LATE

            repair = self.is_repair_sample[sample_idx] > 0
            rescue = self.is_rescue_sample[sample_idx] > 0
            stable = self.is_stable_sample[sample_idx] > 0
            healthy_clear = self.is_healthy_clear_sample[sample_idx] > 0
            sustain = self.is_sustain_sample[sample_idx] > 0
            tail = self.is_tail_candidate[sample_idx] > 0
            bad = self.is_bad_structure_sample[sample_idx] > 0

            core_local = (~bad) & (repair | rescue | stable | healthy_clear)

            if row.strong_flag == 1 and int(core_local.sum()) < self.args.strong_fallback_min_keep:
                gain_local = self.structure_gain[sample_idx]
                candidate_local = np.where(~bad)[0]
                if candidate_local.size > 0:
                    order = candidate_local[np.argsort(-gain_local[candidate_local])]
                    topk = order[: min(self.args.strong_fallback_topk, len(order))]
                    core_local[topk] = True

            context_local = np.zeros_like(core_local, dtype=bool)
            core_positions = np.where(core_local)[0]
            if core_positions.size > 0 and self.args.context_radius > 0:
                for cp in core_positions.tolist():
                    left = max(0, cp - self.args.context_radius)
                    right = min(len(core_local), cp + self.args.context_radius + 1)
                    for pos in range(left, right):
                        if not bad[pos]:
                            context_local[pos] = True

            sustain_aux_local = np.zeros_like(core_local, dtype=bool)
            if row.strong_flag == 1:
                sustain_pool = (
                    (~bad)
                    & (~core_local)
                    & (~context_local)
                    & sustain
                )
                sustain_candidates = np.where(sustain_pool)[0]
                if sustain_candidates.size > 0:
                    core_cnt = int(core_local.sum())
                    cap = min(
                        self.args.max_sustain_aux_per_episode,
                        max(1, int(math.ceil(max(core_cnt, 1) * self.args.sustain_aux_ratio))),
                    )
                    ranked = self._rank_positions(sample_idx, sustain_candidates, prefer_high_risk=False)
                    keep = ranked[: min(cap, len(ranked))]
                    sustain_aux_local[keep] = True

            tail_aux_local = np.zeros_like(core_local, dtype=bool)
            if row.strong_flag == 1:
                late_start_piece = max(1, int(math.floor(row.final_pieces * self.args.tail_late_ratio)))
                late_mask_for_tail = local_piece_idx >= late_start_piece
                tail_pool = (
                    (~bad)
                    & (~core_local)
                    & (~context_local)
                    & (~sustain_aux_local)
                    & tail
                    & late_mask_for_tail
                )
                tail_candidates = np.where(tail_pool)[0]
                if tail_candidates.size > 0:
                    cap = min(self.args.max_tail_aux_per_episode, len(tail_candidates))
                    ranked = self._rank_positions(sample_idx, tail_candidates, prefer_high_risk=False)
                    keep = ranked[:cap]
                    tail_aux_local[keep] = True

            fail_repair_aux_local = np.zeros_like(core_local, dtype=bool)
            if is_fail_episode:
                fail_pool = (
                    (~bad)
                    & late_mask
                    & (~core_local)
                    & (~context_local)
                    & (~sustain_aux_local)
                    & (~tail_aux_local)
                    & (self.risk_before[sample_idx] >= float(self.thresholds["rescue_risk_thresh"]))
                    & (
                        (self.structure_gain[sample_idx] >= self.args.fail_repair_gain_floor)
                        | (self.delta_holes[sample_idx] <= self.args.fail_repair_delta_holes_max)
                    )
                )
                fail_candidates = np.where(fail_pool)[0]
                self.is_fail_repair_candidate[sample_idx[fail_candidates]] = 1
                if fail_candidates.size > 0:
                    ranked = self._rank_positions(sample_idx, fail_candidates, prefer_high_risk=True)
                    keep = ranked[: min(self.args.max_fail_repair_aux_per_episode, len(ranked))]
                    fail_repair_aux_local[keep] = True

            context_local, sustain_aux_local, tail_aux_local, fail_repair_aux_local = self._rebalance_stage_mix(
                sample_idx=sample_idx,
                row=row,
                core_local=core_local,
                context_local=context_local,
                sustain_aux_local=sustain_aux_local,
                tail_aux_local=tail_aux_local,
                fail_repair_aux_local=fail_repair_aux_local,
                early_mask=early_mask,
                mid_mask=mid_mask,
                late_mask=late_mask,
                bad_mask=bad,
            )

            final_local = core_local | context_local | sustain_aux_local | tail_aux_local | fail_repair_aux_local

            core_global[sample_idx[core_local]] = True
            context_global[sample_idx[context_local & (~core_local)]] = True
            sustain_aux_global[sample_idx[sustain_aux_local]] = True
            tail_aux_global[sample_idx[tail_aux_local]] = True
            fail_repair_aux_global[sample_idx[fail_repair_aux_local]] = True
            fallback_global[sample_idx[(sustain_aux_local | tail_aux_local | fail_repair_aux_local) & (~core_local) & (~context_local)]] = True

            row.selected_core_count = int(core_local.sum())
            row.selected_context_count = int((context_local & (~core_local)).sum())
            row.selected_sustain_aux_count = int(sustain_aux_local.sum())
            row.selected_tail_aux_count = int(tail_aux_local.sum())
            row.selected_fail_repair_aux_count = int(fail_repair_aux_local.sum())
            row.selected_sample_count = int(final_local.sum())

        self.selected_core_mask = core_global
        self.selected_context_mask = context_global
        self.selected_sustain_aux_mask = sustain_aux_global
        self.selected_tail_aux_mask = tail_aux_global
        self.selected_fail_repair_aux_mask = fail_repair_aux_global
        self.selected_fallback_mask = fallback_global
        self.selected_sample_mask = (
            core_global
            | context_global
            | sustain_aux_global
            | tail_aux_global
            | fail_repair_aux_global
        )

        episode_score_arr = (
            self.data["episode_final_lines"].astype(np.float32) * self.args.episode_line_weight
            + self.data["episode_final_pieces"].astype(np.float32) * self.args.episode_piece_weight
        )
        episode_score_q90 = max(float(np.percentile(episode_score_arr, 90)), 1.0)
        episode_score_norm = np.clip(episode_score_arr / episode_score_q90, 0.0, 2.0)

        weights = (
            1.0
            + 1.6 * self.is_rescue_sample.astype(np.float32)
            + 1.2 * self.is_repair_sample.astype(np.float32)
            + 0.9 * self.is_healthy_clear_sample.astype(np.float32)
            + 0.5 * self.is_stable_sample.astype(np.float32)
            + 0.35 * self.selected_sustain_aux_mask.astype(np.float32)
            + 0.25 * self.selected_tail_aux_mask.astype(np.float32)
            + 0.45 * self.selected_fail_repair_aux_mask.astype(np.float32)
            + 0.15 * self.selected_context_mask.astype(np.float32)
            + 0.25 * self.is_strong_episode_sample.astype(np.float32)
            + 0.10 * self.is_candidate_episode_sample.astype(np.float32)
            + 0.25 * episode_score_norm.astype(np.float32)
        )

        weights = np.where(self.is_bad_structure_sample > 0, 0.5, weights)
        self.sample_weight_v6 = np.clip(weights.astype(np.float32), 1.0, 5.0)

    def save_outputs(self) -> None:
        selected_idx = np.where(self.selected_sample_mask)[0]
        if selected_idx.size <= 0:
            raise ValueError("no selected samples after V6 structural selection")

        selected_episode_ids = sorted(
            [row.episode_index for row in self.episode_rows if row.selected_sample_count > 0]
        )
        if len(selected_episode_ids) < 3:
            raise ValueError(f"selected episodes too few: {len(selected_episode_ids)}")

        train_eps, val_eps, test_eps = build_episode_split(
            episode_ids=selected_episode_ids,
            seed=self.args.seed,
            train_ratio=self.args.train_ratio,
            val_ratio=self.args.val_ratio,
        )
        train_eps_set = set(train_eps)
        val_eps_set = set(val_eps)
        test_eps_set = set(test_eps)

        old_to_new = {int(old_idx): new_idx for new_idx, old_idx in enumerate(selected_idx.tolist())}

        train_idx = np.array(
            [old_to_new[int(old)] for old in selected_idx.tolist() if int(self.data["episode_index"][old]) in train_eps_set],
            dtype=np.int32,
        )
        val_idx = np.array(
            [old_to_new[int(old)] for old in selected_idx.tolist() if int(self.data["episode_index"][old]) in val_eps_set],
            dtype=np.int32,
        )
        test_idx = np.array(
            [old_to_new[int(old)] for old in selected_idx.tolist() if int(self.data["episode_index"][old]) in test_eps_set],
            dtype=np.int32,
        )

        selected_npz_path = self.out_dir / f"selected_dataset_{self.args.tag}.npz"
        selected_payload: Dict[str, np.ndarray] = {}
        for key, value in self.data.items():
            if isinstance(value, np.ndarray) and value.shape[0] == self.n_samples:
                selected_payload[key] = value[selected_idx]

        selected_payload["holes_before"] = self.holes_before[selected_idx]
        selected_payload["row_transitions_before"] = self.row_transitions_before[selected_idx]
        selected_payload["col_transitions_before"] = self.col_transitions_before[selected_idx]
        selected_payload["well_sums_before"] = self.well_sums_before[selected_idx]
        selected_payload["max_height_before"] = self.max_height_before[selected_idx]
        selected_payload["roughness_before"] = self.roughness_before[selected_idx]

        selected_payload["delta_holes"] = self.delta_holes[selected_idx]
        selected_payload["delta_row_transitions"] = self.delta_row_transitions[selected_idx]
        selected_payload["delta_col_transitions"] = self.delta_col_transitions[selected_idx]
        selected_payload["delta_well_sums"] = self.delta_well_sums[selected_idx]
        selected_payload["delta_max_height"] = self.delta_max_height[selected_idx]
        selected_payload["delta_roughness"] = self.delta_roughness[selected_idx]

        selected_payload["risk_before"] = self.risk_before[selected_idx].astype(np.float32)
        selected_payload["risk_after"] = self.risk_after[selected_idx].astype(np.float32)
        selected_payload["structure_gain"] = self.structure_gain[selected_idx].astype(np.float32)

        selected_payload["future_risk_mean"] = self.future_risk_mean[selected_idx].astype(np.float32)
        selected_payload["future_lines_sum"] = self.future_lines_sum[selected_idx].astype(np.float32)
        selected_payload["future_topout_soon"] = self.future_topout_soon[selected_idx].astype(np.int8)
        selected_payload["steps_to_end"] = self.steps_to_end[selected_idx].astype(np.int16)
        selected_payload["stage_bucket"] = self.stage_bucket[selected_idx].astype(np.int8)

        selected_payload["is_repair_sample"] = self.is_repair_sample[selected_idx].astype(np.int8)
        selected_payload["is_rescue_sample"] = self.is_rescue_sample[selected_idx].astype(np.int8)
        selected_payload["is_stable_sample"] = self.is_stable_sample[selected_idx].astype(np.int8)
        selected_payload["is_healthy_clear_sample"] = self.is_healthy_clear_sample[selected_idx].astype(np.int8)
        selected_payload["is_bad_structure_sample"] = self.is_bad_structure_sample[selected_idx].astype(np.int8)
        selected_payload["is_sustain_sample"] = self.is_sustain_sample[selected_idx].astype(np.int8)
        selected_payload["is_tail_candidate"] = self.is_tail_candidate[selected_idx].astype(np.int8)
        selected_payload["is_fail_repair_candidate"] = self.is_fail_repair_candidate[selected_idx].astype(np.int8)

        selected_payload["selected_core_mask"] = self.selected_core_mask[selected_idx].astype(np.int8)
        selected_payload["selected_context_mask"] = self.selected_context_mask[selected_idx].astype(np.int8)
        selected_payload["selected_sustain_aux_mask"] = self.selected_sustain_aux_mask[selected_idx].astype(np.int8)
        selected_payload["selected_tail_aux_mask"] = self.selected_tail_aux_mask[selected_idx].astype(np.int8)
        selected_payload["selected_fail_repair_aux_mask"] = self.selected_fail_repair_aux_mask[selected_idx].astype(np.int8)
        selected_payload["selected_fallback_mask"] = self.selected_fallback_mask[selected_idx].astype(np.int8)

        selected_payload["sample_weight_v6"] = self.sample_weight_v6[selected_idx].astype(np.float32)

        selected_payload["train_idx"] = train_idx
        selected_payload["val_idx"] = val_idx
        selected_payload["test_idx"] = test_idx

        np.savez_compressed(selected_npz_path, **selected_payload)

        np.save(self.out_dir / "boards.npy", selected_payload["boards"])
        np.save(self.out_dir / "actives.npy", selected_payload["actives"])
        np.save(self.out_dir / "legals.npy", selected_payload["legals"])
        np.save(self.out_dir / "legal_masks.npy", selected_payload["legals"])
        np.save(self.out_dir / "pieces.npy", selected_payload["pieces"])
        np.save(self.out_dir / "state_vecs.npy", selected_payload["pieces"])
        np.save(self.out_dir / "targets.npy", selected_payload["targets"])
        np.save(self.out_dir / "target_actions.npy", selected_payload["targets"])

        np.save(self.out_dir / "sample_weight_v6.npy", selected_payload["sample_weight_v6"])
        np.save(self.out_dir / "train_idx.npy", train_idx)
        np.save(self.out_dir / "val_idx.npy", val_idx)
        np.save(self.out_dir / "test_idx.npy", test_idx)

        np.save(self.out_dir / "is_repair_sample.npy", selected_payload["is_repair_sample"])
        np.save(self.out_dir / "is_rescue_sample.npy", selected_payload["is_rescue_sample"])
        np.save(self.out_dir / "is_stable_sample.npy", selected_payload["is_stable_sample"])
        np.save(self.out_dir / "is_healthy_clear_sample.npy", selected_payload["is_healthy_clear_sample"])
        np.save(self.out_dir / "is_sustain_sample.npy", selected_payload["is_sustain_sample"])
        np.save(self.out_dir / "is_tail_candidate.npy", selected_payload["is_tail_candidate"])
        np.save(self.out_dir / "is_fail_repair_candidate.npy", selected_payload["is_fail_repair_candidate"])

        episode_rows_out: List[Dict[str, object]] = []
        for row in self.episode_rows:
            episode_rows_out.append(
                {
                    "episode_index": row.episode_index,
                    "seed": row.seed,
                    "final_lines": row.final_lines,
                    "final_pieces": row.final_pieces,
                    "final_score": row.final_score,
                    "stop_reason": CODE_TO_STOP_REASON.get(row.stop_reason_code, str(row.stop_reason_code)),
                    "sample_count": row.sample_count,
                    "episode_score": row.episode_score,
                    "candidate_flag": row.candidate_flag,
                    "strong_flag": row.strong_flag,
                    "selected_sample_count": row.selected_sample_count,
                    "selected_core_count": row.selected_core_count,
                    "selected_context_count": row.selected_context_count,
                    "selected_sustain_aux_count": row.selected_sustain_aux_count,
                    "selected_tail_aux_count": row.selected_tail_aux_count,
                    "selected_fail_repair_aux_count": row.selected_fail_repair_aux_count,
                }
            )
        write_csv(
            self.out_dir / f"selected_episodes_{self.args.tag}.csv",
            [
                "episode_index",
                "seed",
                "final_lines",
                "final_pieces",
                "final_score",
                "stop_reason",
                "sample_count",
                "episode_score",
                "candidate_flag",
                "strong_flag",
                "selected_sample_count",
                "selected_core_count",
                "selected_context_count",
                "selected_sustain_aux_count",
                "selected_tail_aux_count",
                "selected_fail_repair_aux_count",
            ],
            episode_rows_out,
        )

        sample_rows_out: List[Dict[str, object]] = []
        for new_idx, old_idx in enumerate(selected_idx.tolist()):
            stage_code = int(self.stage_bucket[old_idx])
            if stage_code == STAGE_EARLY:
                stage_name = "early"
            elif stage_code == STAGE_MID:
                stage_name = "mid"
            else:
                stage_name = "late"

            sample_rows_out.append(
                {
                    "new_index": new_idx,
                    "old_index": int(old_idx),
                    "episode_index": int(self.data["episode_index"][old_idx]),
                    "seed": int(self.data["seed"][old_idx]),
                    "piece_index": int(self.data["piece_index"][old_idx]),
                    "stage_bucket": stage_name,
                    "target_action": int(self.data["targets"][old_idx]),
                    "lines_gain_immediate": int(self.data["lines_gain_immediate"][old_idx]),
                    "episode_final_lines": int(self.data["episode_final_lines"][old_idx]),
                    "episode_final_pieces": int(self.data["episode_final_pieces"][old_idx]),
                    "holes_before": int(self.holes_before[old_idx]),
                    "holes_after": int(self.data["holes_after"][old_idx]),
                    "delta_holes": int(self.delta_holes[old_idx]),
                    "max_height_before": int(self.max_height_before[old_idx]),
                    "max_height_after": int(self.data["max_height_after"][old_idx]),
                    "delta_height": int(self.delta_max_height[old_idx]),
                    "roughness_before": int(self.roughness_before[old_idx]),
                    "roughness_after": int(self.data["roughness_after"][old_idx]),
                    "delta_roughness": int(self.delta_roughness[old_idx]),
                    "risk_before": float(self.risk_before[old_idx]),
                    "risk_after": float(self.risk_after[old_idx]),
                    "structure_gain": float(self.structure_gain[old_idx]),
                    "future_risk_mean": float(self.future_risk_mean[old_idx]),
                    "future_lines_sum": float(self.future_lines_sum[old_idx]),
                    "future_topout_soon": int(self.future_topout_soon[old_idx]),
                    "steps_to_end": int(self.steps_to_end[old_idx]),
                    "is_repair_sample": int(self.is_repair_sample[old_idx]),
                    "is_rescue_sample": int(self.is_rescue_sample[old_idx]),
                    "is_stable_sample": int(self.is_stable_sample[old_idx]),
                    "is_healthy_clear_sample": int(self.is_healthy_clear_sample[old_idx]),
                    "is_sustain_sample": int(self.is_sustain_sample[old_idx]),
                    "is_tail_candidate": int(self.is_tail_candidate[old_idx]),
                    "is_fail_repair_candidate": int(self.is_fail_repair_candidate[old_idx]),
                    "is_bad_structure_sample": int(self.is_bad_structure_sample[old_idx]),
                    "selected_core_mask": int(self.selected_core_mask[old_idx]),
                    "selected_context_mask": int(self.selected_context_mask[old_idx]),
                    "selected_sustain_aux_mask": int(self.selected_sustain_aux_mask[old_idx]),
                    "selected_tail_aux_mask": int(self.selected_tail_aux_mask[old_idx]),
                    "selected_fail_repair_aux_mask": int(self.selected_fail_repair_aux_mask[old_idx]),
                    "sample_weight_v6": float(self.sample_weight_v6[old_idx]),
                }
            )
        write_csv(
            self.out_dir / f"selected_sample_index_{self.args.tag}.csv",
            [
                "new_index",
                "old_index",
                "episode_index",
                "seed",
                "piece_index",
                "stage_bucket",
                "target_action",
                "lines_gain_immediate",
                "episode_final_lines",
                "episode_final_pieces",
                "holes_before",
                "holes_after",
                "delta_holes",
                "max_height_before",
                "max_height_after",
                "delta_height",
                "roughness_before",
                "roughness_after",
                "delta_roughness",
                "risk_before",
                "risk_after",
                "structure_gain",
                "future_risk_mean",
                "future_lines_sum",
                "future_topout_soon",
                "steps_to_end",
                "is_repair_sample",
                "is_rescue_sample",
                "is_stable_sample",
                "is_healthy_clear_sample",
                "is_sustain_sample",
                "is_tail_candidate",
                "is_fail_repair_candidate",
                "is_bad_structure_sample",
                "selected_core_mask",
                "selected_context_mask",
                "selected_sustain_aux_mask",
                "selected_tail_aux_mask",
                "selected_fail_repair_aux_mask",
                "sample_weight_v6",
            ],
            sample_rows_out,
        )

        selected_lines = selected_payload["episode_final_lines"].astype(np.int32)
        selected_pieces = selected_payload["episode_final_pieces"].astype(np.int32)

        selected_ep_lines = np.array(
            [row.final_lines for row in self.episode_rows if row.selected_sample_count > 0],
            dtype=np.float32,
        )
        selected_ep_pieces = np.array(
            [row.final_pieces for row in self.episode_rows if row.selected_sample_count > 0],
            dtype=np.float32,
        )

        selected_stage_bucket = selected_payload["stage_bucket"].astype(np.int8)

        summary = {
            "input_npz": str(self.npz_path),
            "input_episode_csv": str(self.episode_csv_path) if self.episode_csv_path else "",
            "tag": self.args.tag,
            "n_input_samples": int(self.n_samples),
            "n_selected_samples": int(selected_idx.size),
            "n_input_episodes": int(len(self.episode_rows)),
            "n_selected_episodes": int(len(selected_episode_ids)),
            "train_samples": int(train_idx.size),
            "val_samples": int(val_idx.size),
            "test_samples": int(test_idx.size),
            "train_episodes": int(len(train_eps)),
            "val_episodes": int(len(val_eps)),
            "test_episodes": int(len(test_eps)),
            "thresholds": self.thresholds,
            "base_summary": self.base_summary,
            "selected_episode_summary": summarize_episode_metrics(selected_ep_lines, selected_ep_pieces),
            "selected_sample_summary": summarize_episode_metrics(selected_lines, selected_pieces),
            "flag_counts_selected": {
                "repair": int(selected_payload["is_repair_sample"].sum()),
                "rescue": int(selected_payload["is_rescue_sample"].sum()),
                "stable": int(selected_payload["is_stable_sample"].sum()),
                "healthy_clear": int(selected_payload["is_healthy_clear_sample"].sum()),
                "sustain": int(selected_payload["is_sustain_sample"].sum()),
                "tail_candidate": int(selected_payload["is_tail_candidate"].sum()),
                "fail_repair_candidate": int(selected_payload["is_fail_repair_candidate"].sum()),
                "selected_sustain_aux": int(selected_payload["selected_sustain_aux_mask"].sum()),
                "selected_tail_aux": int(selected_payload["selected_tail_aux_mask"].sum()),
                "selected_fail_repair_aux": int(selected_payload["selected_fail_repair_aux_mask"].sum()),
                "bad_structure": int(selected_payload["is_bad_structure_sample"].sum()),
                "core": int(selected_payload["selected_core_mask"].sum()),
                "context": int(selected_payload["selected_context_mask"].sum()),
                "stage_early": int(np.sum(selected_stage_bucket == STAGE_EARLY)),
                "stage_mid": int(np.sum(selected_stage_bucket == STAGE_MID)),
                "stage_late": int(np.sum(selected_stage_bucket == STAGE_LATE)),
            },
            "args": vars(self.args),
        }

        manifest_path = self.out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        summary_txt_path = self.out_dir / f"selection_summary_{self.args.tag}.txt"
        lines = []
        lines.append(f"tag: {self.args.tag}")
        lines.append(f"input_npz: {self.npz_path}")
        lines.append(f"n_input_samples: {self.n_samples}")
        lines.append(f"n_selected_samples: {selected_idx.size}")
        lines.append(f"n_input_episodes: {len(self.episode_rows)}")
        lines.append(f"n_selected_episodes: {len(selected_episode_ids)}")
        lines.append("")
        lines.append("thresholds:")
        for k, v in self.thresholds.items():
            lines.append(f"  {k}: {v:.6f}")
        lines.append("")
        lines.append("base_summary:")
        for k, v in self.base_summary.items():
            if k == "episodes":
                lines.append(f"  {k}: {int(v)}")
            else:
                lines.append(f"  {k}: {v:.6f}")
        lines.append("")
        selected_ep_summary = summarize_episode_metrics(selected_ep_lines, selected_ep_pieces)
        lines.append("selected_episode_summary:")
        for k, v in selected_ep_summary.items():
            if k == "episodes":
                lines.append(f"  {k}: {int(v)}")
            else:
                lines.append(f"  {k}: {v:.6f}")
        lines.append("")
        lines.append("selected_flag_counts:")
        lines.append(f"  repair: {int(selected_payload['is_repair_sample'].sum())}")
        lines.append(f"  rescue: {int(selected_payload['is_rescue_sample'].sum())}")
        lines.append(f"  stable: {int(selected_payload['is_stable_sample'].sum())}")
        lines.append(f"  healthy_clear: {int(selected_payload['is_healthy_clear_sample'].sum())}")
        lines.append(f"  sustain: {int(selected_payload['is_sustain_sample'].sum())}")
        lines.append(f"  tail_candidate: {int(selected_payload['is_tail_candidate'].sum())}")
        lines.append(f"  fail_repair_candidate: {int(selected_payload['is_fail_repair_candidate'].sum())}")
        lines.append(f"  selected_sustain_aux: {int(selected_payload['selected_sustain_aux_mask'].sum())}")
        lines.append(f"  selected_tail_aux: {int(selected_payload['selected_tail_aux_mask'].sum())}")
        lines.append(f"  selected_fail_repair_aux: {int(selected_payload['selected_fail_repair_aux_mask'].sum())}")
        lines.append(f"  bad_structure: {int(selected_payload['is_bad_structure_sample'].sum())}")
        lines.append(f"  core: {int(selected_payload['selected_core_mask'].sum())}")
        lines.append(f"  context: {int(selected_payload['selected_context_mask'].sum())}")
        lines.append(f"  stage_early: {int(np.sum(selected_stage_bucket == STAGE_EARLY))}")
        lines.append(f"  stage_mid: {int(np.sum(selected_stage_bucket == STAGE_MID))}")
        lines.append(f"  stage_late: {int(np.sum(selected_stage_bucket == STAGE_LATE))}")

        summary_txt_path.write_text("\n".join(lines), encoding="utf-8")

        self.saved_paths = {
            "selected_npz": selected_npz_path,
            "boards_npy": self.out_dir / "boards.npy",
            "actives_npy": self.out_dir / "actives.npy",
            "legals_npy": self.out_dir / "legals.npy",
            "legal_masks_npy": self.out_dir / "legal_masks.npy",
            "pieces_npy": self.out_dir / "pieces.npy",
            "state_vecs_npy": self.out_dir / "state_vecs.npy",
            "targets_npy": self.out_dir / "targets.npy",
            "target_actions_npy": self.out_dir / "target_actions.npy",
            "sample_weight_v6_npy": self.out_dir / "sample_weight_v6.npy",
            "train_idx_npy": self.out_dir / "train_idx.npy",
            "val_idx_npy": self.out_dir / "val_idx.npy",
            "test_idx_npy": self.out_dir / "test_idx.npy",
            "episodes_csv": self.out_dir / f"selected_episodes_{self.args.tag}.csv",
            "sample_index_csv": self.out_dir / f"selected_sample_index_{self.args.tag}.csv",
            "manifest_json": manifest_path,
            "summary_txt": summary_txt_path,
        }

    def print_summary(self) -> None:
        print(f"[SELECT] input_samples={self.n_samples}")
        print(f"[SELECT] selected_samples={int(self.selected_sample_mask.sum())}")
        print(f"[SELECT] input_episodes={len(self.episode_rows)}")
        print(f"[SELECT] selected_episodes={sum(1 for x in self.episode_rows if x.selected_sample_count > 0)}")
        print(
            "[THRESH] "
            f"candidate_score={self.thresholds['candidate_score_thresh']:.4f} "
            f"candidate_lines={self.thresholds['candidate_lines_thresh']:.4f} "
            f"candidate_pieces={self.thresholds['candidate_pieces_thresh']:.4f} | "
            f"strong_score={self.thresholds['strong_score_thresh']:.4f} "
            f"strong_lines={self.thresholds['strong_lines_thresh']:.4f} "
            f"strong_pieces={self.thresholds['strong_pieces_thresh']:.4f}"
        )
        print(
            "[STRUCT] "
            f"rescue_risk={self.thresholds['rescue_risk_thresh']:.4f} "
            f"stable_risk_max={self.thresholds['stable_risk_max']:.4f} "
            f"repair_gain={self.thresholds['repair_gain_thresh']:.4f} "
            f"rescue_gain={self.thresholds['rescue_gain_thresh']:.4f}"
        )
        print(
            "[FUTURE] "
            f"future_risk_thresh={self.thresholds['future_risk_thresh']:.4f} "
            f"tail_future_lines_thresh={self.thresholds['tail_future_lines_thresh']:.4f}"
        )
        print(
            "[STAGE] "
            f"early_min={self.args.stage_early_min_ratio:.2f} "
            f"mid_min={self.args.stage_mid_min_ratio:.2f} "
            f"late_max={self.args.stage_late_max_ratio:.2f}"
        )
        for name, path in self.saved_paths.items():
            print(f"[SAVE] {name} -> {path}")


# ============================================================
# CLI
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--selfplay_npz",
        type=str,
        default=r"C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfplay\selfplay_dataset_v1.npz",
    )
    parser.add_argument(
        "--episode_csv",
        type=str,
        default=r"C:\Users\25361\PycharmProjects\pythonProject3\out_task8_selfplay\episode_metrics_v1.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out_task8_elite_v6",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="v6_struct_mainline_v2",
    )

    parser.add_argument("--episode_line_weight", type=float, default=10.0)
    parser.add_argument("--episode_piece_weight", type=float, default=0.05)

    parser.add_argument("--candidate_top_pct", type=float, default=0.40)
    parser.add_argument("--candidate_lines_floor", type=int, default=10)
    parser.add_argument("--candidate_pieces_floor", type=int, default=90)

    parser.add_argument("--strong_top_pct", type=float, default=0.12)
    parser.add_argument("--strong_lines_floor", type=int, default=22)
    parser.add_argument("--strong_pieces_floor", type=int, default=130)

    parser.add_argument("--risk_holes_coef", type=float, default=2.0)
    parser.add_argument("--risk_height_coef", type=float, default=1.2)
    parser.add_argument("--risk_rough_coef", type=float, default=0.5)
    parser.add_argument("--risk_row_trans_coef", type=float, default=0.15)
    parser.add_argument("--risk_col_trans_coef", type=float, default=0.15)

    parser.add_argument("--gain_holes_coef", type=float, default=2.5)
    parser.add_argument("--gain_height_coef", type=float, default=1.2)
    parser.add_argument("--gain_rough_coef", type=float, default=0.6)
    parser.add_argument("--gain_row_trans_coef", type=float, default=0.15)
    parser.add_argument("--gain_col_trans_coef", type=float, default=0.15)
    parser.add_argument("--gain_clear_coef", type=float, default=1.5)

    parser.add_argument("--repair_delta_holes_max", type=int, default=-1)
    parser.add_argument("--repair_gain_quantile", type=float, default=0.70)
    parser.add_argument("--repair_gain_floor", type=float, default=1.0)

    parser.add_argument("--rescue_risk_quantile", type=float, default=0.75)
    parser.add_argument("--rescue_risk_floor", type=float, default=18.0)
    parser.add_argument("--rescue_gain_quantile", type=float, default=0.50)
    parser.add_argument("--rescue_gain_floor", type=float, default=0.0)
    parser.add_argument("--rescue_delta_holes_max", type=int, default=0)
    parser.add_argument("--rescue_delta_height_max", type=int, default=1)

    parser.add_argument("--stable_risk_quantile", type=float, default=0.35)
    parser.add_argument("--stable_delta_holes_max", type=int, default=0)
    parser.add_argument("--stable_delta_height_max", type=int, default=0)
    parser.add_argument("--stable_delta_rough_max", type=int, default=0)
    parser.add_argument("--stable_delta_row_trans_max", type=int, default=1)
    parser.add_argument("--stable_delta_col_trans_max", type=int, default=1)

    parser.add_argument("--healthy_clear_delta_holes_max", type=int, default=0)
    parser.add_argument("--healthy_clear_delta_height_max", type=int, default=1)
    parser.add_argument("--healthy_clear_delta_rough_max", type=int, default=1)
    parser.add_argument("--healthy_clear_delta_row_trans_max", type=int, default=2)
    parser.add_argument("--healthy_clear_delta_col_trans_max", type=int, default=2)

    parser.add_argument("--bad_delta_holes_min", type=int, default=2)
    parser.add_argument("--bad_delta_height_min", type=int, default=3)
    parser.add_argument("--bad_delta_rough_min", type=int, default=4)
    parser.add_argument("--bad_structure_gain_max", type=float, default=0.0)

    parser.add_argument("--future_horizon", type=int, default=3)
    parser.add_argument("--future_risk_quantile", type=float, default=0.50)
    parser.add_argument("--future_risk_floor", type=float, default=24.0)
    parser.add_argument("--sustain_risk_margin", type=float, default=4.0)
    parser.add_argument("--sustain_risk_floor", type=float, default=28.0)
    parser.add_argument("--sustain_min_steps_to_end", type=int, default=3)
    parser.add_argument("--tail_future_lines_quantile", type=float, default=0.65)
    parser.add_argument("--tail_future_lines_floor", type=float, default=1.0)

    parser.add_argument("--context_radius", type=int, default=1)
    parser.add_argument("--max_sustain_aux_per_episode", type=int, default=3)
    parser.add_argument("--sustain_aux_ratio", type=float, default=0.25)
    parser.add_argument("--max_tail_aux_per_episode", type=int, default=2)
    parser.add_argument("--tail_late_ratio", type=float, default=0.70)

    parser.add_argument("--fail_repair_episode_lines_max", type=int, default=20)
    parser.add_argument("--fail_repair_gain_floor", type=float, default=0.0)
    parser.add_argument("--fail_repair_delta_holes_max", type=int, default=0)
    parser.add_argument("--max_fail_repair_aux_per_episode", type=int, default=2)

    parser.add_argument("--stage_early_end_ratio", type=float, default=0.30)
    parser.add_argument("--stage_late_start_ratio", type=float, default=0.70)
    parser.add_argument("--stage_early_min_ratio", type=float, default=0.25)
    parser.add_argument("--stage_mid_min_ratio", type=float, default=0.45)
    parser.add_argument("--stage_late_max_ratio", type=float, default=0.30)
    parser.add_argument("--stage_balance_gain_floor", type=float, default=0.0)

    parser.add_argument("--strong_fallback_min_keep", type=int, default=6)
    parser.add_argument("--strong_fallback_topk", type=int, default=8)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selector = TrajectorySelectorV6(args)
    selector.run()


if __name__ == "__main__":
    main()
