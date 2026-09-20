"""回測:把模型的機率當「下注依據」,算真正在意的事——錢。

策略:|p - 0.5| >= threshold 才下注(否則跳過),flat 1 unit;
同時跑兩條線對照:
  - model      : 用模型機率(來自 evaluate.py --pred-out 的 csv)
  - market_open: 用「開盤賠率去抽水後的隱含機率」(賭場的開盤價,基準)

指標:
  - n_bets / hit_rate
  - ROI = 總獲利 / 下注場次(flat 1u)
  - CLV  = 平均(p_下注方 - 收盤市場隱含機率_同方)
           > 0 表示你在價格上贏過收盤市場(長期可贏的訊號)
  - max_consec_loss / max_drawdown(累計利潤曲線)
  - Kelly 平均份額(參考,flat 策略不用它下注)
  - 門檻掃描 0.50~0.70 的 n_bets / ROI

注意:真實下注用的是「開盤賠率」(開賽前你能拿到的價格),模型預測時間點 = 開盤前。

範例:
  python eval/backtest.py --preds output/preds.csv --matches-csv data/demo/matches.csv \
      --threshold 0.55 --report output/backtest.json --plot output/equity.png
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd


def implied_home(odds_home, odds_away) -> np.ndarray:
    qh = 1.0 / np.asarray(odds_home, dtype=float)
    qa = 1.0 / np.asarray(odds_away, dtype=float)
    return qh / (qh + qa)


def run_strategy(p_home: np.ndarray, odds_home: np.ndarray, odds_away: np.ndarray,
                 y_home: np.ndarray, close_imp_home: np.ndarray,
                 threshold: float) -> dict:
    """flat 1u 策略。回傳指標 dict。"""
    n = len(p_home)
    home_side = p_home >= threshold
    away_side = p_home <= (1.0 - threshold)
    take_home = home_side & ~away_side
    take_away = away_side & ~home_side
    bet = np.where(take_home, 1, np.where(take_away, -1, 0))  # 1=主 -1=客 0=skip

    idx = np.where(bet != 0)[0]
    out = {"n_bets": int(len(idx)), "skipped": int(n - len(idx))}
    # 全長 profit 向量(跳過的場次=0)→ 資產曲線對齊所有場次
    o_side = np.where(bet == 1, odds_home, odds_away)
    p_side = np.where(bet == 1, p_home, 1.0 - p_home)
    y_side = np.where(bet == 1, y_home, 1.0 - y_home)
    cs_side = np.where(bet == 1, close_imp_home, 1.0 - close_imp_home)
    profit_full = np.where(bet != 0, np.where(y_side == 1, o_side - 1.0, -1.0), 0.0)
    out["_profit"] = np.cumsum(profit_full)
    if len(idx) == 0:
        out.update(hit_rate=None, roi=None, clv=None, max_consec_loss=0,
                   max_drawdown=0.0, avg_kelly=None)
        return out

    o = o_side[idx]            # 買進價格(十進位)
    p_bet = p_side[idx]        # 模型對下注方的機率
    y = y_side[idx]            # 下注方是否贏(0/1)
    close_side = cs_side[idx]
    profit = profit_full[idx]

    # Kelly:f* = (p*o - 1)/(o - 1)
    kelly = np.clip((p_bet * o - 1.0) / (o - 1.0), 0.0, 0.25)

    # 最大連敗
    max_streak, cur = 0, 0
    for w in profit:
        cur = cur + 1 if w < 0 else 0
        max_streak = max(max_streak, cur)

    # drawdown 用全長曲線(包含跳過的場次)
    cum_full = out["_profit"]
    peak_full = np.maximum.accumulate(np.concatenate([[0.0], cum_full]))[1:]
    dd_full = peak_full - cum_full

    out.update(
        hit_rate=float(y.mean()),
        roi=float(profit.mean()),
        clv=float((p_bet - close_side).mean()),
        max_consec_loss=int(max_streak),
        max_drawdown=float(dd_full.max()),
        avg_kelly=float(kelly.mean()),
        net_profit=float(profit.sum()),
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True, help="evaluate.py --pred-out 的 csv")
    ap.add_argument("--matches-csv", required=True)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--report", default="output/backtest.json")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    preds = pd.read_csv(args.preds)
    df = pd.read_csv(args.matches_csv)
    df = df.set_index("match_id")

    rows = []
    for _, r in preds.iterrows():
        m = df.loc[r["match_id"]]
        rows.append({
            "match_id": r["match_id"],
            "date": r.get("date", m.get("date", "")),
            "p_home_model": float(r["p_home"]),
            "o_home": float(m["open_ml_home"]),
            "o_away": float(m["open_ml_away"]),
            "c_home": float(m["close_ml_home"]),
            "c_away": float(m["close_ml_away"]),
            "y_home": int(m["home_win"]),
        })
    d = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    p_m = d["p_home_model"].values
    o_h, o_a = d["o_home"].values, d["o_away"].values
    c_i = implied_home(d["c_home"].values, d["c_away"].values)
    y = d["y_home"].values

    # model 線
    model = run_strategy(p_m, o_h, o_a, y, c_i, args.threshold)
    # market_open 線(開盤去抽水隱含機率,同一門檻邏輯)
    m_open = implied_home(o_h, o_a)
    market = run_strategy(m_open, o_h, o_a, y, c_i, args.threshold)

    # 門檻掃描
    scan = []
    for t in np.arange(0.50, 0.701, 0.025):
        t = round(float(t), 3)
        mres = run_strategy(p_m, o_h, o_a, y, c_i, t)
        scan.append({"threshold": t, "n_bets": mres["n_bets"],
                     "roi": None if mres["roi"] is None else round(mres["roi"], 4),
                     "hit_rate": None if mres["hit_rate"] is None else round(mres["hit_rate"], 4)})

    report = {
        "n_games": int(len(d)),
        "threshold": args.threshold,
        "note": "flat 1u,下注價=開盤賠率;CLV>0 表示價格贏過收盤市場",
        "model": {k: v for k, v in model.items() if k != "_profit"},
        "market_open": {k: v for k, v in market.items() if k != "_profit"},
        "threshold_scan": scan,
    }
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    def f3(v, nd=4):
        return f"{v:.{nd}f}" if v is not None else "n/a"

    print(f"\ngames={len(d)}  threshold={args.threshold}")
    print(f"{'line':<14}{'bets':>6}{'hit':>8}{'ROI':>9}{'CLV':>9}{'maxDD':>8}{'連敗':>6}{'Kelly':>8}")
    for name in ("model", "market_open"):
        r = report[name]
        print(f"{name:<14}{r['n_bets']:>6}{f3(r['hit_rate'],3):>8}{f3(r['roi']):>9}"
              f"{f3(r['clv']):>9}{f3(r['max_drawdown'],1):>8}{r['max_consec_loss']:>6}"
              f"{f3(r['avg_kelly'],3):>8}")
    print("\nthreshold scan (model):")
    print(f"{'thr':>6}{'bets':>7}{'ROI':>9}{'hit':>8}")
    for s in scan:
        print(f"{s['threshold']:>6.3f}{s['n_bets']:>7}{str(f3(s['roi'])):>9}{str(f3(s['hit_rate'],3)):>8}")
    print(f"\nreport -> {args.report}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        if len(d) > 0:
            x = np.arange(len(d))
            ax1.plot(x, model["_profit"], label="model", lw=1.5)
            ax1.plot(x, market["_profit"], label="market_open", lw=1.2, alpha=0.8)
            ax1.axhline(0, color="k", lw=0.8)
            ax1.set_title("Cumulative profit (flat 1u, bet @ open odds)")
            ax1.set_xlabel("game index (chronological)")
            ax1.legend(); ax1.grid(alpha=0.3)
            thr = [s["threshold"] for s in scan]
            roi = [s["roi"] if s["roi"] is not None else np.nan for s in scan]
            ax2.plot(thr, roi, "o-")
            ax2.axhline(0, color="k", lw=0.8)
            ax2.set_title("ROI by threshold (model)")
            ax2.set_xlabel("confidence threshold")
            ax2.grid(alpha=0.3)
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"plot   -> {args.plot}")


if __name__ == "__main__":
    main()
