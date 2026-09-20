"""eval/backtest.py 單元測試:手算小型案例驗證 ROI/CLV/drawdown/連敗。"""
import numpy as np
import pytest

from eval.backtest import implied_home, run_strategy


def test_implied_home():
    h = implied_home([1.85], [1.90])
    assert abs(h[0] - 0.5) < 0.01
    h2 = implied_home([1.50], [2.80])
    assert h2[0] > 0.6


def test_strategy_all_correct():
    p = np.array([0.8, 0.2, 0.8, 0.2])
    oh = np.array([1.5] * 4)
    oa = np.array([3.0] * 4)
    y = np.array([1, 0, 1, 0])
    ci = np.array([0.5] * 4)
    r = run_strategy(p, oh, oa, y, ci, 0.55)
    assert r["n_bets"] == 4
    # home @1.5 贏 +0.5;away @3.0 贏 +2.0 → 4 筆
    assert r["roi"] == pytest.approx((0.5 + 2.0 + 0.5 + 2.0) / 4)
    assert r["hit_rate"] == 1.0
    assert r["clv"] == pytest.approx(0.3)          # p_bet=0.8, close=0.5
    assert r["max_consec_loss"] == 0
    assert r["max_drawdown"] == 0.0
    assert r["net_profit"] == pytest.approx(5.0)
    assert r["avg_kelly"] == pytest.approx(0.25)   # 兩邊都被 clip 到 0.25
    # 全長 profit 曲線
    assert len(r["_profit"]) == 4
    assert r["_profit"][-1] == pytest.approx(5.0)


def test_strategy_all_skip():
    r = run_strategy(np.array([0.5, 0.5]), np.array([1.9, 1.9]),
                     np.array([1.9, 1.9]), np.array([1, 0]), np.array([0.5, 0.5]), 0.55)
    assert r["n_bets"] == 0
    assert r["roi"] is None
    assert np.all(r["_profit"] == 0.0)


def test_strategy_drawdown_and_streak():
    # 主 @1.2 贏/客 @4.0 贏/主 @1.2 輸/客 @4.0 輸
    p = np.array([0.9, 0.1, 0.9, 0.1])
    oh = np.array([1.2] * 4)
    oa = np.array([4.0] * 4)
    y = np.array([1, 0, 0, 1])
    ci = np.array([0.5] * 4)
    r = run_strategy(p, oh, oa, y, ci, 0.55)
    profits = [0.2, 3.0, -1.0, -1.0]
    cum = np.cumsum(profits)
    assert r["_profit"][-1] == pytest.approx(cum[-1])
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:]
    assert r["max_drawdown"] == pytest.approx((peak - cum).max())  # = 2.0
    assert r["max_drawdown"] == pytest.approx(2.0)
    assert r["max_consec_loss"] == 2
    assert r["roi"] == pytest.approx(np.mean(profits))
    assert r["hit_rate"] == pytest.approx(0.5)


def test_strategy_threshold_boundary():
    # 剛好 0.55 → 下注;0.549 → 跳過
    r = run_strategy(np.array([0.55]), np.array([2.0]), np.array([2.0]),
                     np.array([1]), np.array([0.5]), 0.55)
    assert r["n_bets"] == 1
    r2 = run_strategy(np.array([0.549]), np.array([2.0]), np.array([2.0]),
                      np.array([1]), np.array([0.5]), 0.55)
    assert r2["n_bets"] == 0
