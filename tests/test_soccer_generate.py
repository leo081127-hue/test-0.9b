"""Dixon-Coles 足球生成器:統計性質 + 特徵不用未來資料。"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


@pytest.fixture(scope="module")
def soccer_df(tmp_path_factory):
    out = tmp_path_factory.mktemp("soccer_gen") / "soccer.csv"
    r = subprocess.run(
        [PY, "data/generate_demo_soccer.py",
         "--seasons", "2", "--games-per-season", "120", "--seed", "5", "--out", out],
        cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    return pd.read_csv(out)


def test_score_and_outcome_consistency(soccer_df):
    df = soccer_df
    assert (df["margin"] == df["home_score"] - df["away_score"]).all()
    assert ((df["home_win"] == (df["margin"] > 0).astype(int))).all()
    assert (df["home_score"] >= 0).all() and (df["away_score"] >= 0).all()
    # 進球是整數
    assert ((df["home_score"] % 1 == 0) & (df["away_score"] % 1 == 0)).all()


def test_base_rates_plausible(soccer_df):
    df = soccer_df
    p_home = df["home_win"].mean()
    p_draw = (df["margin"] == 0).mean()
    p_away = 1 - p_home - p_draw
    # 真實聯賽常見:主 0.40~0.55 / 和 0.18~0.35 / 客 0.20~0.40
    assert 0.38 <= p_home <= 0.55, p_home
    assert 0.15 <= p_draw <= 0.40, p_draw
    assert 0.15 <= p_away <= 0.45, p_away


def test_odds_vig_and_sanity(soccer_df):
    df = soccer_df
    for tag in ("open", "close"):
        o = 1 / df[f"{tag}_ml_home"] + 1 / df[f"{tag}_ml_draw"] + 1 / df[f"{tag}_ml_away"]
        assert (o > 0.95).all() and (o < 1.15).all(), f"{tag} vig {o.min()}~{o.max()}"
    assert (df[["open_ml_home", "open_ml_draw", "open_ml_away",
                "close_ml_home", "close_ml_draw", "close_ml_away"]] > 1.01).all().all()


def test_market_close_beats_coin(soccer_df):
    """去抽水後的收盤 1X2 必須優於均分 1/3(Brier < 1.0)——市場是有效的。"""
    from common import implied_prob_3way, outcome_from_margin
    df = soccer_df
    y = np.array([outcome_from_margin(m) for m in df["margin"].values])
    mkt = np.array([implied_prob_3way(r.close_ml_home, r.close_ml_draw, r.close_ml_away)
                    for r in df.itertuples()])
    brier = ((mkt - np.eye(3)[y]) ** 2).sum(1).mean()
    assert brier < 0.95, f"market brier {brier:.3f} 應該遠優於 coin(1.0)"
    # 收盤必須比開盤好或持平(誤價在收盤被修正)
    mkt_o = np.array([implied_prob_3way(r.open_ml_home, r.open_ml_draw, r.open_ml_away)
                      for r in df.itertuples()])
    brier_o = ((mkt_o - np.eye(3)[y]) ** 2).sum(1).mean()
    assert brier <= brier_o + 0.01, f"close {brier:.3f} 應該不差於 open {brier_o:.3f}"


def test_features_do_not_use_future(soccer_df):
    """抽兩筆:重算該隊「本場之前」的 last10 W-D-L,必須等於存下的特徵。"""
    from common import outcome_from_margin
    df = soccer_df
    picks = df.iloc[[40, 200]]
    for idx, row in picks.iterrows():
        for side, team, col_w, col_d, col_l in (
                ("home", row["home"], "home_form_w", "home_form_d", "home_form_l"),
                ("away", row["away"], "away_form_w", "away_form_d", "away_form_l")):
            prev = df[(df["home"] == team) | (df["away"] == team)]
            prev = prev[prev["date"] < row["date"]].tail(10)
            # 從 team 視角算 W/D/L(team 當客隊時,它的 margin = 負的比賽 margin)
            m = np.where(prev["home"].values == team, prev["margin"].values,
                         -prev["margin"].values)
            w = int((m > 0).sum()); d = int((m == 0).sum())
            assert row[col_w] == w and row[col_d] == d, (
                f"{team} 特徵用了未來資料或算錯: stored {row[col_w]}-{row[col_d]} vs {w}-{d}")


def test_mispricing_design_present(soccer_df):
    """開盤與收盤(去抽水後)必須有差異——這是誤價設計(模型的可學訊號)。"""
    from common import implied_prob_3way
    df = soccer_df
    po = np.array([implied_prob_3way(r.open_ml_home, r.open_ml_draw, r.open_ml_away)
                   for r in df.itertuples()])
    pc = np.array([implied_prob_3way(r.close_ml_home, r.close_ml_draw, r.close_ml_away)
                   for r in df.itertuples()])
    diff = np.abs(po - pc).max(1)
    assert (diff > 0.02).mean() > 0.10, "誤價設計失效:open/close 幾乎沒有差異"
