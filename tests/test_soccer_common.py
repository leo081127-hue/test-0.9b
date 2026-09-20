"""足球 1X2:common.py 的三結果解析、回應建構、多類指標(手算)。"""
import numpy as np
import pytest

from common import (implied_prob_3way, margin_from_score, outcome_from_margin,
                    parse_response, build_response, _acc_brier_logloss_mc, ece_mc)

GOOD = "最終預測: 主隊\n機率: 主隊 0.45, 和局 0.25, 客隊 0.30"
DRAW_PRED = "最終預測: 和局\n機率: 主隊 0.30, 和局 0.40, 客隊 0.30"


def test_parse_soccer_basic():
    p = parse_response(GOOD, "soccer")
    assert p["pred"] == "主隊"
    assert p["format_ok"]
    assert p["p_home"] == pytest.approx(0.45)
    assert p["p_draw"] == pytest.approx(0.25)
    assert p["p_away"] == pytest.approx(0.30)
    assert p["cover_home"] is None


def test_parse_soccer_draw_pred():
    p = parse_response(DRAW_PRED, "soccer")
    assert p["pred"] == "和局"
    assert p["p_draw"] == pytest.approx(0.40)


def test_parse_soccer_renormalized():
    p = parse_response("機率: 主隊 0.4, 和局 0.3, 客隊 0.5", "soccer")
    s = 1.2
    assert p["p_home"] == pytest.approx(0.4 / s)
    assert p["p_draw"] == pytest.approx(0.3 / s)
    assert p["p_away"] == pytest.approx(0.5 / s)


def test_parse_soccer_fullwidth():
    p = parse_response("最終預測: 客隊\n機率: 主隊 0.2, 和局 0.3, 客隊 0.5", "soccer")
    assert p["format_ok"] and p["pred"] == "客隊"


def test_parse_soccer_garbage_and_zero_sum():
    assert not parse_response("完全沒有格式的亂碼", "soccer")["format_ok"]
    assert not parse_response("機率: 主隊 0, 和局 0, 客隊 0", "soccer")["format_ok"]


def test_parse_cross_sport_mismatch():
    """二元格式不該被 soccer parser 接受(反過來也不行)。"""
    binary = "機率: 主隊 0.7, 客隊 0.3"
    assert not parse_response(binary, "soccer")["format_ok"]
    three = "機率: 主隊 0.4, 和局 0.3, 客隊 0.3"
    assert not parse_response(three, "basketball")["format_ok"]


def test_basketball_path_unchanged():
    p = parse_response("最終預測: 主隊\n機率: 主隊 0.7, 客隊 0.3\n讓分覆蓋: 主隊 0.6, 客隊 0.4")
    assert p["p_home"] == pytest.approx(0.7)
    assert p["cover_home"] == pytest.approx(0.6)
    assert p["p_draw"] is None


def test_build_response_soccer_roundtrip():
    r = build_response(0.45, None, ["理由"], sport="soccer", p_draw=0.25)
    p = parse_response(r, "soccer")
    assert p["format_ok"]
    assert p["pred"] == "主隊"
    assert p["p_home"] == pytest.approx(0.45, abs=0.011)
    assert p["p_draw"] == pytest.approx(0.25, abs=0.011)


def test_build_response_soccer_draw_pred():
    r = build_response(0.30, None, [], sport="soccer", p_draw=0.40)
    assert parse_response(r, "soccer")["pred"] == "和局"


def test_outcome_from_margin():
    assert outcome_from_margin(2) == 0
    assert outcome_from_margin(0) == 1
    assert outcome_from_margin(-1) == 2
    assert margin_from_score(3, 1) == 2.0


def test_feature_matrix_picks_up_soccer_cols():
    import pandas as pd
    from common import FEATURE_COLS, feature_matrix
    assert {"home_form_d", "away_form_d", "open_ml_draw", "close_ml_draw"} <= set(FEATURE_COLS)
    df = pd.DataFrame({
        "home_form_w": [5], "home_form_d": [2], "home_form_l": [3],
        "away_form_w": [4], "away_form_d": [1], "away_form_l": [5],
        "home_avg_pts": [1.3], "away_avg_pts": [1.1],
        "home_rest": [2], "away_rest": [1],
        "h2h_home_w": [1], "h2h_away_w": [1],
        "open_ml_home": [2.2], "open_ml_draw": [3.4], "open_ml_away": [3.1],
        "close_ml_home": [2.1], "close_ml_draw": [3.3], "close_ml_away": [3.2],
    })
    X, names = feature_matrix(df)
    for c in ("home_form_d", "away_form_d", "open_ml_draw", "close_ml_draw"):
        assert c in names
    assert X[0][names.index("home_form_d")] == 2.0
    assert X[0][names.index("close_ml_draw")] == 3.3


def test_implied_prob_3way():
    p = implied_prob_3way(2.0, 3.4, 3.6)
    assert sum(p) == pytest.approx(1.0, abs=1e-12)
    # 低賠率 = 高機率
    assert p[0] > p[1] and p[1] > p[2]
    # 均等賠率 → 均等機率
    assert implied_prob_3way(3.0, 3.0, 3.0) == pytest.approx((1 / 3, 1 / 3, 1 / 3), abs=1e-12)


def test_mc_metrics_hand_computed():
    p_mat = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]])
    y_mat = np.array([[1, 0, 0], [0, 1, 0]])
    m = _acc_brier_logloss_mc(p_mat, y_mat)
    assert m["acc"] == 1.0
    # brier = mean( (0.3²+0.2²+0.1²) + (0.2²+0.3²+0.1²) )
    assert m["brier"] == pytest.approx((0.14 + 0.14) / 2, abs=1e-12)
    # logloss = -mean( log0.7 + log0.7 )
    assert m["logloss"] == pytest.approx(-np.log(0.7), abs=1e-12)
    # coin(uniform 1/3) brier = (1/3-1)² + (1/3)² + (1/3)² = 2/3
    c = _acc_brier_logloss_mc(np.full((4, 3), 1 / 3),
                              np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    assert c["brier"] == pytest.approx(2 / 3, abs=1e-12)
    # ece:永遠輸出 0.9 且全對 → 過度自信 0.1 → ECE = 0.1
    assert ece_mc(np.array([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]]),
                  np.array([[1, 0, 0], [1, 0, 0]])) == pytest.approx(0.1, abs=1e-12)
    # 完全校準:conf 全 = 0.7 且 70% 對(7 主贏 + 3 和局)→ ECE = 0
    y_cal = np.tile([1, 0, 0], (7, 1))
    y_cal = np.vstack([y_cal, np.tile([0, 1, 0], (3, 1))])
    assert ece_mc(np.tile([0.7, 0.15, 0.15], (10, 1)), y_cal) == pytest.approx(0.0, abs=1e-12)


def test_mc_metrics_empty():
    m = _acc_brier_logloss_mc(np.empty((0, 3)), np.empty((0, 3)))
    assert m == {"n": 0, "acc": None, "brier": None, "logloss": None}
