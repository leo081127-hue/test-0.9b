"""common.py 單元測試:prompt/回應格式、解析、指標。"""
import math

import numpy as np
import pandas as pd
import pytest

from common import (
    P_HOME,
    P_AWAY,
    _acc_brier_logloss,
    build_prompt,
    build_response,
    ece,
    feature_matrix,
    implied_prob,
    normalize_game,
    parse_response,
)


# ---------------------------------------------------------------- build/parse 往返
@pytest.mark.parametrize("p_home,cover_home", [
    (0.85, 0.87), (0.50, 0.50), (0.02, 0.98), (0.98, 0.03), (0.37, 0.44),
])
def test_build_parse_roundtrip(p_home, cover_home):
    text = build_response(p_home, cover_home, ["理由一", "理由二"])
    r = parse_response(text)
    assert r["format_ok"] is True
    assert abs(r["p_home"] - round(p_home, 2)) < 1e-9
    assert abs(r["p_away"] - round(1 - round(p_home, 2), 2)) < 1e-9
    assert abs(r["cover_home"] - round(cover_home, 2)) < 1e-9
    expect_pred = P_HOME if round(p_home, 2) >= 0.5 else P_AWAY
    assert r["pred"] == expect_pred


def test_build_response_pred_matches_argmax():
    t = build_response(0.62, 0.55, ["x"])
    assert "最終預測: 主隊" in t
    t = build_response(0.41, 0.49, ["x"])
    assert "最終預測: 客隊" in t


def test_build_response_clamps_extremes():
    t = build_response(0.999, 0.001, ["x"])
    r = parse_response(t)
    assert 0.02 <= r["p_home"] <= 0.98


# ---------------------------------------------------------------- 解析的韌性
@pytest.mark.parametrize("text,expect_p_home", [
    ("最終預測: 主隊\n機率: 主隊 0.60, 客隊 0.40\n讓分覆蓋: 主隊 0.55, 客隊 0.45", 0.60),
    ("最終預測：客隊\n機率：主隊 0.35，客隊 0.65\n讓分覆蓋：主隊 0.4, 客隊 0.6", 0.35),
    ("blah blah\n機率: 主隊 0.10, 客隊 0.90", 0.10),
    ("機率: 主隊 1, 客隊 0", 1.0),
    ("機率: 主隊 0.00, 客隊 1.00", 0.0),
])
def test_parse_variants(text, expect_p_home):
    r = parse_response(text)
    assert r["format_ok"], text
    assert abs(r["p_home"] - expect_p_home) < 1e-9


def test_parse_unnormalized_sums_normalized():
    r = parse_response("機率: 主隊 0.6, 客隊 0.8")
    assert abs(r["p_home"] - 0.6 / 1.4) < 1e-9


def test_parse_garbage():
    r = parse_response("完全亂碼,沒有格式")
    assert r["format_ok"] is False
    assert r["p_home"] is None
    assert r["pred"] is None
    assert parse_response("")["format_ok"] is False


def test_parse_cover_only_no_winprob():
    r = parse_response("讓分覆蓋: 主隊 0.7, 客隊 0.3")
    assert r["format_ok"] is False  # 沒有勝率列
    assert abs(r["cover_home"] - 0.7) < 1e-9


# ---------------------------------------------------------------- implied_prob
def test_implied_prob_devig():
    # 1.85 / 1.90 含 ~5% 抽水 → 隱含約 0.5/0.5
    h, a = implied_prob(1.85, 1.90)
    assert abs(h + a - 1.0) < 1e-12
    assert abs(h - 0.5) < 0.01
    # 明顯主隊
    h2, _ = implied_prob(1.50, 2.80)
    assert h2 > 0.6


# ---------------------------------------------------------------- 指標
def test_metrics_perfect():
    m = _acc_brier_logloss([0.9, 0.1, 0.8], [1, 0, 1])
    assert m["acc"] == 1.0
    assert m["brier"] == pytest.approx((0.01 + 0.01 + 0.04) / 3, abs=1e-9)
    assert m["logloss"] > 0


def test_metrics_coin_brier_025():
    y = [1, 0, 1, 0, 1, 0]
    m = _acc_brier_logloss([0.5] * 6, y)
    assert m["brier"] == pytest.approx(0.25)


def test_metrics_empty():
    m = _acc_brier_logloss([], [])
    assert m["n"] == 0 and m["acc"] is None


def test_ece_calibrated_near_zero():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.4, 0.6, 4000)
    y = (rng.random(4000) < p).astype(float)
    assert ece(p, y) < 0.05


def test_ece_overconfident_high():
    p = np.array([0.95] * 100 + [0.05] * 100)
    y = np.array([0] * 100 + [1] * 100)  # 全錯但很自信
    assert ece(p, y) > 0.5


# ---------------------------------------------------------------- 特徵/normalization
def test_normalize_game_parses_records():
    g = normalize_game({"home_record": "12-8", "away_record": "5-15"})
    assert g["home_record_w"] == 12 and g["home_record_l"] == 8
    assert g["away_record_w"] == 5 and g["away_record_l"] == 15
    # 壞值不炸
    g2 = normalize_game({"home_record": "xx"})
    assert "home_record_w" not in g2


def test_feature_matrix():
    df = pd.DataFrame({
        "home_form_w": [6, np.nan], "home_form_l": [4, 3],
        "away_form_w": [7, 5], "away_form_l": [3, 5],
        "home_avg_pts": [118.0, 115.0], "away_avg_pts": [119.0, 112.0],
        "home_rest": [1, 2], "away_rest": [2, 1],
        "home_record": ["12-8", "9-9"], "away_record": ["9-10", "10-8"],
        "h2h_home_w": [2, 1], "h2h_away_w": [3, 4],
        "open_spread": [-2.0, 3.0], "close_spread": [-3.0, 4.0],
        "open_ml_home": [1.9, 2.3], "open_ml_away": [1.95, 1.6],
        "close_ml_home": [1.85, 2.4], "close_ml_away": [1.9, 1.55],
    })
    X, names = feature_matrix(df)
    assert X.shape == (2, 20)
    assert "home_record_w" in names
    assert X[0][names.index("home_record_w")] == 12
    assert not np.isnan(X).any()


def test_build_prompt_contains_key_facts():
    g = {
        "season": "2025-26", "home": "A隊", "away": "B隊",
        "home_form_w": 7, "home_form_l": 3, "away_form_w": 4, "away_form_l": 6,
        "home_avg_pts": 118.4, "away_avg_pts": 112.1, "home_rest": 1, "away_rest": 3,
        "home_record": "12-8", "away_record": "9-10",
        "h2h_home_w": 2, "h2h_away_w": 3,
        "open_spread": -2.5, "close_spread": -3.5, "open_total": 224.5, "close_total": 226.0,
        "open_ml_home": 1.92, "open_ml_away": 1.90,
        "close_ml_home": 1.98, "close_ml_away": 1.86,
    }
    text = build_prompt(g)
    assert "A隊" in text and "B隊" in text
    assert "-3.5" in text  # 收盤讓分(主隊讓 3.5)
    assert "1.98" in text


def test_format_spread_sign():
    g = {"home": "H", "away": "A", "close_spread": +3.0}
    text = build_prompt(g)
    assert "+3.0" in text  # 主隊受讓 3 分
