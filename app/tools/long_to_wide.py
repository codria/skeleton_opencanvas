"""
long_to_wide.py
extract_csv.py が --format long で出力した CSV を、Wide 形式に変換する CLI ツール。

Usage:
    python -m app.tools.long_to_wide INPUT.csv [OUTPUT.csv]

OUTPUT を省略すると INPUT を in-place で上書きする（一時ファイル経由で安全）。
Wide 形式は 1 行 = 1 (frame_idx, person)、列がランドマーク別に展開されており、
Excel などで人間が見やすい。列順は extract_csv --format wide と完全一致する。
"""

from __future__ import annotations
import argparse
import logging
import os
import sys

import pandas as pd

logger = logging.getLogger(__name__)


# extract_csv.py と同じ 33 点の順序
LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER",
    "LEFT_EAR", "RIGHT_EAR",
    "MOUTH_LEFT", "MOUTH_RIGHT",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_PINKY", "RIGHT_PINKY",
    "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_THUMB", "RIGHT_THUMB",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE",
    "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

# Long 側の列名 → Wide 側のサフィックス（extract_csv._write_wide_header と一致）
ATTRS: list[tuple[str, str]] = [
    ("x", "x"), ("y", "y"), ("z", "z"), ("visibility", "vis"),
    ("wx", "wx"), ("wy", "wy"), ("wz", "wz"), ("w_visibility", "wvis"),
]


def long_to_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Long -> Wide 変換。

    Long 側の列: frame_idx, t_sec, person, lm_idx, lm_name, x, y, z, visibility,
                  wx, wy, wz, w_visibility
    Wide 側の列: frame_idx, t_sec, person, <NAME>_x, <NAME>_y, ..., <NAME>_wvis, ...
    """
    key_cols = ["frame_idx", "t_sec", "person"]
    pieces: list[pd.DataFrame] = []
    for attr, suffix in ATTRS:
        # 各 attr について長列 → 列 (ランドマーク名) にピボット
        p = long_df.pivot_table(
            index=key_cols, columns="lm_name", values=attr, aggfunc="first"
        )
        p.columns = [f"{lm}_{suffix}" for lm in p.columns]
        pieces.append(p)
    wide = pd.concat(pieces, axis=1).reset_index()

    # 列順を extract_csv --format wide と完全に一致させる
    ordered: list[str] = list(key_cols)
    for name in LANDMARK_NAMES:
        for _long, suffix in ATTRS:
            col = f"{name}_{suffix}"
            if col in wide.columns:
                ordered.append(col)
    return wide[ordered]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.tools.long_to_wide",
        description="extract_csv.py --format long で作った CSV を Wide に変換する。",
    )
    p.add_argument("input", help="入力 CSV (Long format)")
    p.add_argument("output", nargs="?", default=None,
                   help="出力 CSV。省略で INPUT を in-place で上書き（一時ファイル経由で安全）。")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG ログを出す")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not os.path.exists(args.input):
        logger.error(f"入力ファイルが見つかりません: {args.input}")
        return 2

    logger.info(f"読込: {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Long: {len(df)} rows, {df.shape[1]} cols")

    wide = long_to_wide(df)
    logger.info(f"Wide: {len(wide)} rows, {wide.shape[1]} cols")

    if args.output is None:
        # in-place 上書きは、一時ファイル経由 → os.replace（原子的操作）
        tmp = args.input + ".tmp"
        wide.to_csv(tmp, index=False)
        os.replace(tmp, args.input)
        out_path = args.input
    else:
        wide.to_csv(args.output, index=False)
        out_path = args.output

    logger.info(f"完了: {len(df)} rows (long) → {len(wide)} rows (wide) → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
