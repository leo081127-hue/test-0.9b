"""run_strategy_3way(1X2 flat 1u 回測)手算驗證。"""
import numpy as np
import pytest

from eval.backtest import implied_3way, run_strategy_3way


def _mats():
    p = np.array([
        [0.6, 0.2, 0.2],   # row0 → 押主 (conf 0.6)
        [0.2, 0.6, 0.2],   # row1 → 押和 (conf 0.6)
        [0.2, 0.2, 0.7],   # row2 → 押客 (conf 0.7)
        [0.4, 0.3, 0.3],   # row3 → conf 0.4 < 0.55 跳過
    ])
    o = np.array([
        [1.8, 3.0, 3.0],
        [3.0, 1.9, 3.0],
        [3.0, 3.0, 1.6],
        [2.0, 3.0, 3.5],
    ])
    y = np.array([
        [1, 0, 0],  # 主贏 → row0 贏
        [1, 0, 0],  # 主贏 → row1(押和)輸
        [0, 0, 1],  # 客贏 → row2 贏
        [0, 1, 0],
    ])
    close = np.array([
        [0.55, 0.30, 0.15],
        [0.45, 0.50, 0.05],
        [0.20, 0.10, 0.60],
        [0.30, 0.40, 0.30],
    ])
    return p, o, y, close


def test_3way_strategy_hand_computed():
    p, o, y, c = _mats()
    r = run_strategy_3way(p, o, y, c, 0.55)
    assert r["n_bets"] == 3 and r["skipped"] == 1
    # profit = [0.8, -1.0, 0.6, 0]
    assert r["hit_rate"] == pytest.approx(2 / 3)
    assert r["roi"] == pytest.approx((0.8 - 1.0 + 0.6) / 3)
    assert r["net_profit"] == pytest.approx(0.4)
    # CLV = mean(0.6-0.55, 0.6-0.50, 0.7-0.60)
    assert r["clv"] == pytest.approx((0.05 + 0.10 + 0.10) / 3)
    # 資產曲線 [0.8, -0.2, 0.4, 0.4] → 最大回撤 1.0
    assert r["max_drawdown"] == pytest.approx(1.0)
    assert r["max_consec_loss"] == 1
    # Kelly: (0.6*1.8-1)/0.8=0.1, (0.6*1.9-1)/0.9, (0.7*1.6-1)/0.6=0.2
    k = (0.1 + (0.6 * 1.9 - 1) / 0.9 + 0.2) / 3
    assert r["avg_kelly"] == pytest.approx(k)
    # 全長 profit 向量
    assert r["_profit"] == pytest.approx(np.array([0.8, -0.2, 0.4, 0.4]))


def test_3way_all_skip():
    p, o, y, c = _mats()
    r = run_strategy_3way(p, o, y, c, 0.99)
    assert r["n_bets"] == 0 and r["skipped"] == 4
    assert r["roi"] is None and r["clv"] is None and r["hit_rate"] is None
    assert r["max_drawdown"] == 0.0


def test_implied_3way():
    q = implied_3way(np.array([[2.0, 3.0, 3.0]]))
    assert q.sum() == pytest.approx(1.0, abs=1e-12)
    # 原始 [1/2, 1/3, 1/3] 加總 7/6 → 正規化 [3/7, 2/7, 2/7]
    assert q == pytest.approx(np.array([[3 / 7, 2 / 7, 2 / 7]]), abs=1e-12)
    # 對稱賠率 → 均等
    q2 = implied_3way(np.array([[3.0, 3.0, 3.0]]))
    assert q2 == pytest.approx(np.full((1, 3), 1 / 3), abs=1e-12)
