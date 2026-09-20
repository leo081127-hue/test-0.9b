"""資料品質報告:在训练前驗證 matches.csv(特別是你的真實資料)。

檢查項目:
  ERROR:重複 match_id / 缺必要欄位 / margin 與比分不符 / home_win 與 margin 矛盾
  WARN :date 不可排序或大斷層 / 賠率 <=1.01 / vig 超出 [0.95, 1.25] /
        收盤讓分覆蓋率偏離 50%±10 / 主隊勝率異常 / 特徵缺漏率 >20% / 近10場 >10

用法:
  python data/qa_report.py --matches 你的資料.csv [--json output/qa.json]
常數 exit code 0(這是報告工具);問題數量見輸出。
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

REQUIRED = [
    "match_id", "date", "season", "home", "away",
    "home_score", "away_score", "home_win", "margin",
    "open_spread", "close_spread",
    "open_ml_home", "open_ml_away", "close_ml_home", "close_ml_away",
]
OPT_FEATURES = [
    "home_form_w", "home_form_l", "away_form_w", "away_form_l",
    "home_avg_pts", "away_avg_pts", "home_rest", "away_rest",
    "home_record", "away_record", "h2h_home_w", "h2h_away_w",
]


REQUIRED_SOCCER = [
    "match_id", "date", "season", "home", "away",
    "home_score", "away_score", "home_win", "margin",
    "open_ml_home", "open_ml_draw", "open_ml_away",
    "close_ml_home", "close_ml_draw", "close_ml_away",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--sport", choices=("basketball", "soccer", "auto"), default="auto")
    ap.add_argument("--json", default=None, help="輸出 JSON 報告路徑")
    args = ap.parse_args()

    df = pd.read_csv(args.matches)
    sport = args.sport
    if sport == "auto":
        sport = "soccer" if "open_ml_draw" in df.columns else "basketball"
    required = REQUIRED_SOCCER if sport == "soccer" else REQUIRED
    print(f"sport: {sport}")

    issues: list[dict] = []

    def error(check: str, msg: str):
        issues.append({"severity": "ERROR", "check": check, "message": msg})

    def warn(check: str, msg: str):
        issues.append({"severity": "WARN", "check": check, "message": msg})

    print(f"loaded {len(df)} rows, {len(df.columns)} columns")

    # ---- 必要欄位 ----
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        error("required_columns", f"缺少必要欄位: {missing_cols}")
    if missing_cols:
        df = df[[c for c in df.columns if c not in missing_cols]]

    # ---- 重複 id ----
    dup = df["match_id"].duplicated().sum() if "match_id" in df.columns else 0
    if dup:
        error("duplicate_id", f"{dup} 個重複 match_id")

    # ---- date ----
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        n_bad = int(d.isna().sum())
        if n_bad:
            error("date_parse", f"{n_bad} 筆 date 無法解析")
        else:
            if not d.is_monotonic_increasing:
                warn("date_order", "date 不是時間遞增(記得 sort_values)")
            gap = d.diff().dt.days
            gap = gap[gap > 0]
            if len(gap) and gap.max() > 60:
                warn("date_gap", f"最大日期斷層 {int(gap.max())} 天(跨賽季正常,否則查資料)")
        print(f"date range: {d.min().date()} .. {d.max().date()}")

    # ---- 比分/結果一致性 ----
    if all(c in df.columns for c in ("home_score", "away_score", "margin", "home_win")):
        m_diff = (df["home_score"] - df["away_score"]) - df["margin"]
        bad_m = int((m_diff.abs() > 0.5).sum())
        if bad_m:
            error("margin_consistency", f"{bad_m} 筆 margin != home_score - away_score")
        bad_w = int(((df["margin"] > 0).astype(int) != df["home_win"]).sum())
        if bad_w:
            error("win_consistency", f"{bad_w} 筆 home_win 與 margin 矛盾")
        if (df["home_score"] < 0).any() or (df["away_score"] < 0).any():
            error("negative_score", "有負數比分")
    if sport == "soccer" and "home_score" in df.columns:
        hs = pd.to_numeric(df["home_score"], errors="coerce")
        bad_i = int(((hs != hs.round()) | hs.isna()).sum())
        if bad_i:
            error("goals_not_int", f"{bad_i} 筆進球數不是整數(足球)")

    # ---- 賠率 ----
    ml_cols = ("open_ml_home", "open_ml_away", "close_ml_home", "close_ml_away")
    if sport == "soccer":
        ml_cols = ml_cols + ("open_ml_draw", "close_ml_draw")
    for col in ml_cols:
        if col in df.columns:
            n_low = int((df[col] <= 1.01).sum())
            if n_low:
                error(f"odds_low:{col}", f"{n_low} 筆賠率 <= 1.01")
    if sport == "soccer" and all(f"{t}_ml_draw" in df.columns for t in ("open", "close")):
        for tag in ("open", "close"):
            vig = (1 / df[f"{tag}_ml_home"] + 1 / df[f"{tag}_ml_draw"]
                   + 1 / df[f"{tag}_ml_away"])
            n_bad = int(((vig < 0.95) | (vig > 1.35)).sum())
            if n_bad:
                warn(f"vig_{tag}", f"{n_bad} 筆 1X2 vig 超出 [0.95, 1.35]( {vig.min():.3f}~{vig.max():.3f} )")
            else:
                print(f"vig {tag} (1X2): {vig.min():.3f} ~ {vig.max():.3f} (ok)")
    elif all(c in df.columns for c in ("open_ml_home", "open_ml_away",
                                       "close_ml_home", "close_ml_away")):
        for tag, cols in (("open", ("open_ml_home", "open_ml_away")),
                          ("close", ("close_ml_home", "close_ml_away"))):
            vig = 1 / df[cols[0]] + 1 / df[cols[1]]
            n_bad = int(((vig < 0.95) | (vig > 1.25)).sum())
            if n_bad:
                warn(f"vig_{tag}", f"{n_bad} 筆 vig 超出 [0.95, 1.25]( vig={vig.min():.3f}~{vig.max():.3f} )")
            else:
                print(f"vig {tag}: {vig.min():.3f} ~ {vig.max():.3f} (ok)")

    # ---- 讓分覆蓋率(足球 v1 無讓分 → 跳過)----
    if all(c in df.columns for c in ("margin", "close_spread")) and df["close_spread"].notna().any():
        cov = (df["margin"] > df["close_spread"]).mean()
        if not (0.40 <= cov <= 0.60):
            warn("cover_rate", f"收盤讓分覆蓋率 {cov:.3f} 偏離 0.50 太多(盤口慣例或標籤問題?)")
        else:
            print(f"home cover rate (close): {cov:.3f} (ok)")

    # ---- 主隊勝率 / 足球 1X2 基率 ----
    if "home_win" in df.columns:
        hw = df["home_win"].mean()
        lo, hi = (0.35, 0.70) if sport != "soccer" else (0.30, 0.55)
        if not (lo <= hw <= hi):
            warn("home_win_rate", f"主隊勝率 {hw:.3f} 異常({sport} 常見 {lo:.2f}~{hi:.2f})")
        else:
            print(f"home win rate: {hw:.3f} (ok)")
    if sport == "soccer" and "margin" in df.columns:
        dr = (df["margin"] == 0).mean()
        if not (0.15 <= dr <= 0.40):
            warn("draw_rate", f"和局率 {dr:.3f} 異常(足球常見 0.20~0.32)")
        else:
            print(f"draw rate: {dr:.3f} (ok)")

    # ---- 特徵缺漏 ----
    for col in OPT_FEATURES:
        if col in df.columns:
            na = df[col].isna().mean()
            if na > 0.20:
                warn(f"feature_na:{col}", f"缺漏率 {na:.1%}")
    if "home_form_w" in df.columns and "home_form_l" in df.columns:
        s_h = df["home_form_w"].fillna(0) + df["home_form_l"].fillna(0)
        s_a = df["away_form_w"].fillna(0) + df["away_form_l"].fillna(0)
        if sport == "soccer":
            s_h = s_h + df["home_form_d"].fillna(0) if "home_form_d" in df.columns else s_h
            s_a = s_a + df["away_form_d"].fillna(0) if "away_form_d" in df.columns else s_a
        bad = int(((s_h > 10) | (s_a > 10)).sum())
        if bad:
            warn("form_window", f"{bad} 筆近10場(勝+平+負) > 10")

    # ---- 輸出 ----
    n_err = sum(1 for i in issues if i["severity"] == "ERROR")
    n_warn = sum(1 for i in issues if i["severity"] == "WARN")
    print("\n" + "=" * 60)
    if not issues:
        print("QA PASS:沒有發現問題")
    for i in issues:
        print(f"  [{i['severity']}] {i['check']}: {i['message']}")
    print(f"total: {n_err} errors, {n_warn} warnings")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"n_rows": len(df), "n_errors": n_err, "n_warnings": n_warn,
                       "issues": issues}, f, ensure_ascii=False, indent=2)
        print(f"json -> {args.json}")


if __name__ == "__main__":
    main()
