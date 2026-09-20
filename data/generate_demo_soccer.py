"""產生合成 demo 足球賽事資料(1X2 三結果):比分 + 開/收盤 1X2 賠率 + 自訂特徵。

統計核心(Dixon & Coles 1997 風格):
- 每隊潛在攻擊力 atk / 防守力 def(高=強)+ 全局主場優勢。
- 進球數 ~ Poisson:λh = BASE·exp(0.32·(atk_h − def_a) + γ + 休息效應),λa 對稱。
- **Dixon-Coles τ 低分修正**(κ>0):獨立 Poisson 會低估 0-0 與 1-1、
  高估 1-0/0-1。修正只動四格低分:
      P(0,0) *= 1 + λh·λa·κ    P(1,1) *= 1 + κ
      P(1,0) *= 1 − λa·κ       P(0,1) *= 1 − λh·κ
  再重新正規化。真實聯賽 κ 的拟合值約 0.05~0.15。
- 1X2 真實機率 = 修正後比分矩陣的 主勝/和/客勝 邊際;實際比分從同一分佈抽樣
  → 市場機率與實際結果自洽。
- 約 25% 賽事的**開盤**被「誤價」(三結果機率隨機位移 ±0.02~0.06),
  收盤修正約 75% → 模型有可學的訊號,market close baseline 依然很強。
- 休息天數真實影響進球率(λ 上 +1.5%/天差)但 1X2 賠率看不到 → 可學非效率。

輸出欄位見 data/SCHEMA.md(sport=soccer;v1 無讓分,spread/total 留空)。

範例:
  python data/generate_demo_soccer.py --seasons 3 --games-per-season 150 --out data/demo/soccer.csv
"""
from __future__ import annotations

import argparse
import math
import os
from collections import deque
from datetime import date, timedelta

import numpy as np
import pandas as pd

BASE_GOALS = 1.32   # 聯賽平均每隊進球
HFA_LOG = 0.22      # 主場優勢(log 尺度)
KAPPA = 0.10        # Dixon-Coles 低分修正 κ
MAX_GOALS = 10      # 比分矩陣上界(λ≤4 時尾巴 < 1e-5)
VIG = 1.05          # 抽水(overround):Σ(1/odds) = VIG ≈ 1.05(>1 才是 bookmaker edge)


def _poisson_row(lam: float, kmax: int = MAX_GOALS) -> np.ndarray:
    pmf = np.zeros(kmax + 1)
    pmf[0] = math.exp(-lam)
    for k in range(1, kmax + 1):
        pmf[k] = pmf[k - 1] * lam / k
    return pmf


def score_matrix(lam_h: float, lam_a: float, kappa: float = KAPPA) -> np.ndarray:
    """Dixon-Coles 修正後的比分聯合分佈(已正規化和為 1)。"""
    M = np.outer(_poisson_row(lam_h), _poisson_row(lam_a))
    M[0, 0] *= 1.0 + lam_h * lam_a * kappa
    M[1, 1] *= 1.0 + kappa
    M[1, 0] *= 1.0 - lam_a * kappa
    M[0, 1] *= 1.0 - lam_h * kappa
    s = M.sum()
    return M / s


def one_x_two(m: np.ndarray) -> np.ndarray:
    """比分矩陣 → (主勝, 和, 客勝) 機率。M[x, y]:x=主隊進球(行)、y=客隊進球(欄)。"""
    p_home = float(m[np.tril_indices(len(m), k=-1)].sum())   # x > y
    p_away = float(m[np.triu_indices(len(m), k=1)].sum())    # x < y
    p_draw = float(1.0 - p_home - p_away)
    return np.array([p_home, p_draw, p_away])


def _renorm3(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 0.03, None)
    return p / p.sum()


def _odds(p: np.ndarray, rng) -> np.ndarray:
    """機率 → 十進位賠率(含 VIG 抽水 + 微小雜訊)。

    o = (1/p)/VIG → 1/o = p·VIG → Σ(1/o) = VIG ≈ 1.05(overround,與籃球版一致)。
    """
    o = (1.0 / p) / VIG + rng.normal(0.0, 0.004, len(p))
    return np.round(np.maximum(1.01, o), 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, default=3)
    ap.add_argument("--teams", type=int, default=12)
    ap.add_argument("--games-per-season", type=int, default=150)
    ap.add_argument("--league", default="PremierDemo")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/demo/soccer.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.teams
    teams = [f"T{i:02d}" for i in range(n)]
    atk = rng.normal(0.0, 1.0, n)
    dfn = rng.normal(0.0, 1.0, n)

    last10 = {t: deque(maxlen=10) for t in teams}   # 'W'/'D'/'L'
    scored = {t: deque(maxlen=10) for t in teams}   # 進球數
    h2h: dict = {}
    season_record = {t: {"home": [0, 0, 0], "away": [0, 0, 0]} for t in teams}  # [W,D,L]

    rows: list[dict] = []
    gid = 0
    day_cursor = 0

    for s in range(1, args.seasons + 1):
        season = f"20{18 + s:02d}-20{19 + s:02d}"
        if s > 1:
            day_cursor += 60
        for t in teams:
            season_record[t]["home"] = [0, 0, 0]
            season_record[t]["away"] = [0, 0, 0]

        # 排程:每天每隊最多一場(隨機配對)——避免同一天兩場造成的特徵洩漏
        games: list[tuple[int, int, int]] = []
        per_day = n // 2
        day = 0
        while len(games) < args.games_per_season:
            perm = list(rng.permutation(n))
            for k in range(per_day):
                if len(games) >= args.games_per_season:
                    break
                games.append((day_cursor + day, int(perm[2 * k]), int(perm[2 * k + 1])))
            day += 1

        for day_idx, i, j in games:
            gid += 1
            h_rest, a_rest = int(rng.integers(1, 4)), int(rng.integers(1, 4))

            lam_h = BASE_GOALS * math.exp(0.32 * (atk[i] - dfn[j]) + HFA_LOG
                                          + 0.015 * (h_rest - 2.5))
            lam_a = BASE_GOALS * math.exp(0.32 * (atk[j] - dfn[i])
                                          + 0.015 * (a_rest - 2.5))
            M = score_matrix(lam_h, lam_a)
            p_true = one_x_two(M)

            # 誤價設計:~25% 開盤被扭曲,收盤修正 75%
            if rng.random() < 0.25:
                dh = rng.uniform(0.02, 0.06) * (1 if rng.random() < 0.5 else -1)
                da = rng.uniform(0.02, 0.06) * (1 if rng.random() < 0.5 else -1)
                p_open = _renorm3(p_true + np.array([dh, -(dh + da), da]))
            else:
                p_open = _renorm3(p_true + rng.normal(0.0, 0.008, 3))
            p_close = _renorm3(p_true + 0.25 * (p_open - p_true) + rng.normal(0.0, 0.005, 3))

            # 實際比分從同一分佈抽樣(市場機率與結果自洽)
            flat = M.ravel()
            x, y = divmod(int(rng.choice(len(flat), p=flat / flat.sum())), M.shape[1])
            margin = float(x - y)
            home_win = int(x > y)

            o_open = _odds(p_open, rng)
            o_close = _odds(p_close, rng)

            hw = sum(1 for r_ in last10[teams[i]] if r_ == "W")
            hd = sum(1 for r_ in last10[teams[i]] if r_ == "D")
            hl = len(last10[teams[i]]) - hw - hd
            aw = sum(1 for r_ in last10[teams[j]] if r_ == "W")
            ad = sum(1 for r_ in last10[teams[j]] if r_ == "D")
            al = len(last10[teams[j]]) - aw - ad
            h_avg = float(np.mean(scored[teams[i]])) if scored[teams[i]] else BASE_GOALS
            a_avg = float(np.mean(scored[teams[j]])) if scored[teams[j]] else BASE_GOALS
            pair = frozenset((teams[i], teams[j]))
            recent = list(h2h.get(pair, deque()))[-5:]
            hh = sum(1 for (w, _m) in recent if w == teams[i])
            ha = len(recent) - hh

            rows.append({
                "match_id": f"M{gid:05d}",
                "date": (date(2019, 1, 1) + timedelta(days=day_idx)).isoformat(),
                "season": season,
                "league": args.league,
                "home": teams[i], "away": teams[j],
                "home_score": x, "away_score": y,
                "home_win": home_win, "margin": margin,
                "open_spread": np.nan, "close_spread": np.nan,
                "open_total": np.nan, "close_total": np.nan,
                "open_ml_home": o_open[0], "open_ml_draw": o_open[1], "open_ml_away": o_open[2],
                "close_ml_home": o_close[0], "close_ml_draw": o_close[1], "close_ml_away": o_close[2],
                "home_form_w": hw, "home_form_d": hd, "home_form_l": hl,
                "away_form_w": aw, "away_form_d": ad, "away_form_l": al,
                "home_avg_pts": round(h_avg, 2), "away_avg_pts": round(a_avg, 2),
                "home_rest": h_rest, "away_rest": a_rest,
                "home_record": f"{season_record[teams[i]]['home'][0]}-{season_record[teams[i]]['home'][1]}-{season_record[teams[i]]['home'][2]}",
                "away_record": f"{season_record[teams[j]]['away'][0]}-{season_record[teams[j]]['away'][1]}-{season_record[teams[j]]['away'][2]}",
                "h2h_home_w": hh, "h2h_away_w": ha,
            })

            res_h = "W" if x > y else ("L" if x < y else "D")
            res_a = {"W": "L", "L": "W", "D": "D"}[res_h]
            last10[teams[i]].append(res_h)
            last10[teams[j]].append(res_a)
            scored[teams[i]].append(x)
            scored[teams[j]].append(y)
            season_record[teams[i]]["home"][{"W": 0, "D": 1, "L": 2}[res_h]] += 1
            season_record[teams[j]]["away"][{"W": 0, "D": 1, "L": 2}[res_a]] += 1
            winner = teams[i] if x > y else (teams[j] if y > x else None)
            if winner is not None:
                h2h.setdefault(pair, deque(maxlen=6)).append((winner, margin))
            atk = atk + rng.normal(0.0, 0.002, n)
            dfn = dfn + rng.normal(0.0, 0.002, n)

        day_cursor = games[-1][0] + 1

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    nh, nd = int(df["home_win"].sum()), int((df["margin"] == 0).sum())
    na = len(df) - nh - nd
    vig_o = float((1 / df["open_ml_home"] + 1 / df["open_ml_draw"] + 1 / df["open_ml_away"]).mean())
    print(f"saved {len(df)} games -> {args.out}")
    print(f"  home/draw/away: {nh / len(df):.1%} / {nd / len(df):.1%} / {na / len(df):.1%}")
    print(f"  avg goals: {(df['home_score'] + df['away_score']).mean():.2f}   open vig Σ(1/o)={vig_o:.3f}")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
