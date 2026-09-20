"""產生合成 demo 賽事資料(籃球風格):結果 + 開/收盤盤口 + 賠率 + 自訂特徵。

輸出單一 CSV(欄位規格見 data/SCHEMA.md)。
- 約 12% 的賽事實現「盤口誤價」(bookmaker 賠率偏離真實機率),讓模型有可學的信號。
- 盤口:收盤比開盤更有效(收盤更接近真實),開盤含更多雜訊 → 盤口移動本身是特徵。
- 隊實力隨時間漂移,模擬聯賽季節性變化。

範例:
  python data/generate_demo_data.py --seasons 4 --games-per-season 450 --out data/demo/matches.csv
"""
from __future__ import annotations

import argparse
import math
import os
from collections import deque
from datetime import date, timedelta

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, default=4)
    ap.add_argument("--teams", type=int, default=30)
    ap.add_argument("--games-per-season", type=int, default=450)
    ap.add_argument("--league", default="NBA")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/demo/matches.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n = args.teams
    teams = [f"T{i:02d}" for i in range(n)]
    strength = rng.normal(0.0, 1.0, n)
    HOME_ADV, MARGIN_SD = 3.0, 9.5

    last10 = {t: deque(maxlen=10) for t in teams}          # 1=贏 0=輸
    scored = {t: deque(maxlen=10) for t in teams}          # 得分
    h2h: dict = {}                                         # frozenset -> deque((winner, margin))
    season_record = {t: {"home": [0, 0], "away": [0, 0]} for t in teams}  # [W, L]

    rows: list[dict] = []
    gid = 0
    day_cursor = 0  # 自 2019-01-01 起的絕對天數

    def make_ml(p_book: float, side_home: bool) -> float:
        q = p_book if side_home else 1.0 - p_book
        q = min(0.95, max(0.05, q))
        odds = 1.0 / (q * 1.05)  # 約 5% 抽水
        return round(max(1.01, odds + rng.normal(0.0, 0.004)), 2)

    for s in range(1, args.seasons + 1):
        season = f"20{18 + s:02d}-20{19 + s:02d}"
        if s > 1:
            day_cursor += 60  # 休賽期
        for t in teams:
            season_record[t]["home"] = [0, 0]
            season_record[t]["away"] = [0, 0]

        # 隨機排程:每 3 場推進一天
        games: list[tuple[int, int, int]] = []
        while len(games) < args.games_per_season:
            i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
            if i == j:
                continue
            games.append((day_cursor + len(games) // 3, i, j))

        for day_idx, i, j in games:
            gid += 1

            # 實力差對分差的影響放大 3 倍(接近真實籃球:實力差 sd≈4-5 分,單場分差 sd≈11),
            # 否則勝率機率會被壓在 0.5~0.65、賠率沒有區隔度
            latent = 3.0 * (strength[i] - strength[j]) + HOME_ADV
            margin = latent + rng.normal(0.0, MARGIN_SD)
            p_true = 1.0 / (1.0 + math.exp(-latent / MARGIN_SD))

            # 盤口:~12% 出現誤價
            mis = (rng.uniform(0.02, 0.055) * (1.0 if rng.random() < 0.5 else -1.0)
                   if rng.random() < 0.12 else 0.0)
            p_close = float(np.clip(p_true + rng.normal(0.0, 0.008) + 0.25 * mis, 0.05, 0.95))
            p_open = float(np.clip(p_true + rng.normal(0.0, 0.035) + mis, 0.05, 0.95))

            close_spread = round(latent - 1.0, 1)
            open_spread = round(close_spread + rng.normal(0.0, 1.3), 1)
            total_true = 224.0 + 1.2 * (strength[i] + strength[j])
            close_total = round(total_true / 0.5) * 0.5 + 0.5
            open_total = round((total_true + rng.normal(0.0, 2.0)) / 0.5) * 0.5

            away_pts = int(round(max(60.0, 112.0 + 3.0 * strength[j] + rng.normal(0.0, 9.0))))
            home_pts = int(round(max(60.0, away_pts + margin)))
            margin_real = home_pts - away_pts
            home_win = 1 if margin_real > 0 else 0
            cover_home = 1 if margin_real > close_spread else 0

            # ---- 取「賽前」狀態做特徵(避免用到本場結果)----
            hw, hl = sum(last10[teams[i]]), len(last10[teams[i]]) - sum(last10[teams[i]])
            aw, al = sum(last10[teams[j]]), len(last10[teams[j]]) - sum(last10[teams[j]])
            h_avg = float(np.mean(scored[teams[i]])) if scored[teams[i]] else 112.0
            a_avg = float(np.mean(scored[teams[j]])) if scored[teams[j]] else 112.0
            h_rest, a_rest = int(rng.integers(1, 4)), int(rng.integers(1, 4))
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
                "home_score": home_pts, "away_score": away_pts,
                "home_win": home_win, "margin": float(margin_real),
                "open_spread": open_spread, "close_spread": close_spread,
                "open_total": open_total, "close_total": close_total,
                "open_ml_home": make_ml(p_open, True), "open_ml_away": make_ml(p_open, False),
                "close_ml_home": make_ml(p_close, True), "close_ml_away": make_ml(p_close, False),
                "home_form_w": hw, "home_form_l": hl,
                "away_form_w": aw, "away_form_l": al,
                "home_avg_pts": round(h_avg, 1), "away_avg_pts": round(a_avg, 1),
                "home_rest": h_rest, "away_rest": a_rest,
                "home_record": f"{season_record[teams[i]]['home'][0]}-{season_record[teams[i]]['home'][1]}",
                "away_record": f"{season_record[teams[j]]['away'][0]}-{season_record[teams[j]]['away'][1]}",
                "h2h_home_w": hh, "h2h_away_w": ha,
            })

            # ---- 更新賽後狀態 ----
            last10[teams[i]].append(home_win)
            last10[teams[j]].append(1 - home_win)
            scored[teams[i]].append(home_pts)
            scored[teams[j]].append(away_pts)
            season_record[teams[i]]["home"][0 if home_win else 1] += 1
            season_record[teams[j]]["away"][0 if (1 - home_win) else 1] += 1
            h2h.setdefault(pair, deque(maxlen=6)).append((teams[i] if home_win else teams[j], margin_real))
            strength = strength + rng.normal(0.0, 0.002, n)  # 實力緩慢漂移

        day_cursor = games[-1][0] + 1

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)

    # 摘要
    n_home = int(df["home_win"].sum())
    cov = int((df["margin"] > df["close_spread"]).sum())
    print(f"saved {len(df)} games -> {args.out}")
    print(f"  home wins: {n_home}/{len(df)} ({n_home / len(df):.1%})")
    print(f"  home covers: {cov}/{len(df)} ({cov / len(df):.1%})")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
