"""generate_demo_data.py 測試:確定性 + 資料不變量(含「特徵不可偷看本場結果」)。"""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_gen(out: Path, seasons: int = 2, gps: int = 150, seed: int = 42) -> pd.DataFrame:
    subprocess.run(
        [sys.executable, str(ROOT / "data/generate_demo_data.py"),
         "--seasons", str(seasons), "--games-per-season", str(gps),
         "--seed", str(seed), "--out", str(out)],
        check=True, cwd=ROOT, capture_output=True,
    )
    return pd.read_csv(out)


@pytest.fixture(scope="module")
def df(tmp_path_factory) -> pd.DataFrame:
    return _run_gen(tmp_path_factory.mktemp("gen") / "m.csv")


def test_determinism(tmp_path):
    a = _run_gen(tmp_path / "a.csv", seed=7)
    b = _run_gen(tmp_path / "b.csv", seed=7)
    pd.testing.assert_frame_equal(a, b)
    c = _run_gen(tmp_path / "c.csv", seed=8)
    assert not a["home_score"].equals(c["home_score"])


def test_dates_chronological_and_seasons_ordered(df):
    d = pd.to_datetime(df["date"])
    assert d.is_monotonic_increasing
    for s in sorted(df["season"].unique()):
        pass
    seas = sorted(df["season"].unique())
    assert len(seas) == 2
    max_first = pd.to_datetime(df.loc[df["season"] == seas[0], "date"]).max()
    min_second = pd.to_datetime(df.loc[df["season"] == seas[1], "date"]).min()
    assert min_second > max_first  # 賽季不能重疊/倒退


def test_score_margin_outcome_consistency(df):
    assert (df["home_score"] - df["away_score"] == df["margin"]).all()
    assert (df["home_win"] == (df["margin"] > 0)).all()
    assert (df["home_score"] > 0).all() and (df["away_score"] > 0).all()


def test_odds_sane(df):
    for col in ("open_ml_home", "open_ml_away", "close_ml_home", "close_ml_away"):
        assert (df[col] > 1.01).all(), col
    # 盤口區隔度:收盤隱含機率不該全擠在 0.5 附近(否則沒有可學的信號)
    q = 1 / df["close_ml_home"]
    qa = 1 / df["close_ml_away"]
    mkt = q / (q + qa)
    assert mkt.std() > 0.05
    # 市場線確實預測勝負(AUC 顯著大於 0.5;demo 刻意留漏洞所以不會太高)
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(df["home_win"], mkt) > 0.55


def test_cover_rate_around_half(df):
    cov = (df["margin"] > df["close_spread"]).mean()
    assert 0.38 < cov < 0.68, cov


def test_home_advantage(df):
    assert 0.50 < df["home_win"].mean() < 0.75


def test_rolling_windows_bounded(df):
    assert (df[["home_form_w", "home_form_l", "away_form_w", "away_form_l"]] <= 10).all().all()
    assert (df["home_form_w"] + df["home_form_l"] <= 10).all()
    assert (df["h2h_home_w"] + df["h2h_away_w"] <= 5).all()
    assert not df[["home_avg_pts", "away_avg_pts", "open_spread", "close_spread"]].isna().any().any()


def test_no_future_leak_in_first_game(tmp_path):
    """第一場賽事的 rolling 特徵必須為 0(還沒任何歷史)。"""
    d = _run_gen(tmp_path / "m1.csv", seasons=1, gps=50, seed=3)
    first = d.iloc[0]
    assert first["home_form_w"] == 0 and first["home_form_l"] == 0
    assert first["away_form_w"] == 0 and first["away_form_l"] == 0
    assert first["h2h_home_w"] == 0 and first["h2h_away_w"] == 0
    assert first["home_avg_pts"] == 112.0  # 無歷史時的預設
    assert first["home_record"] == "0-0"


def test_form_updates_after_first_game(tmp_path):
    """第二場若同一隊再出現,近10場應該累計到 1。"""
    d = _run_gen(tmp_path / "m2.csv", seasons=1, gps=200, seed=3)
    # 找同一隊連續兩場(作為主隊或客隊)
    teams_seen = {}
    for i, row in d.iterrows():
        for side, col in (("home", "home"), ("away", "away")):
            t = row[col]
            if t in teams_seen:
                prev_i = teams_seen[t]
                prev = d.iloc[prev_i]
                if t == prev["home"]:
                    exp_w, exp_l = prev["home_win"], 1 - prev["home_win"]
                else:
                    exp_w, exp_l = 1 - prev["home_win"], prev["home_win"]
                if t == row["home"]:
                    assert row["home_form_w"] == prev["home_form_w"] + exp_w
                    assert row["home_form_l"] == prev["home_form_l"] + exp_l
                else:
                    assert row["away_form_w"] == prev["away_form_w"] + exp_w
                    assert row["away_form_l"] == prev["away_form_l"] + exp_l
                return
        teams_seen[row["home"]] = i
        teams_seen[row["away"]] = i
    pytest.fail("沒找到連續出場的同一隊")
