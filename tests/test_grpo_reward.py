"""train/grpo.py 的 reward 函數:手算分數。"""
import pytest

trl = pytest.importorskip("trl")

from train.grpo import make_reward

GOOD_HOME = (
    "<推理>\n- 測試理由\n</推理>\n"
    "最終預測: 主隊\n機率: 主隊 0.80, 客隊 0.20\n讓分覆蓋: 主隊 0.70, 客隊 0.30"
)
WRONG_SIDE = (
    "最終預測: 客隊\n機率: 主隊 0.20, 客隊 0.80\n讓分覆蓋: 主隊 0.30, 客隊 0.70"
)


def test_reward_correct_prediction():
    r = make_reward()
    s = r(prompts=["p"], completions=[GOOD_HOME], outcome_home=[1], cover_home=[1])
    # +1 格式 + 1-(0.8-1)^2=0.96 Brier +0.5 方向 +0.3*(1-(0.7-1)^2)=0.273 覆蓋
    assert s[0] == pytest.approx(1.0 + 0.96 + 0.5 + 0.3 * (1 - 0.09), abs=1e-9)


def test_reward_wrong_side():
    r = make_reward()
    s = r(prompts=["p"], completions=[WRONG_SIDE], outcome_home=[1], cover_home=[1])
    # +1 格式 + 1-(0.2-1)^2=0.36 Brier +0 方向 +0.3*(1-(0.3-1)^2)=0.3*0.51
    assert s[0] == pytest.approx(1.0 + 0.36 + 0.0 + 0.3 * (1 - 0.49), abs=1e-9)
    # 且正確預測的分數一定較高
    s2 = r(prompts=["p"], completions=[GOOD_HOME], outcome_home=[1], cover_home=[1])
    assert s2[0] > s[0]


def test_reward_garbage_zero():
    r = make_reward()
    s = r(prompts=["p"], completions=["完全沒有格式的亂碼"], outcome_home=[1], cover_home=[1])
    assert s[0] == 0.0


def test_reward_missing_labels_returns_none():
    r = make_reward()
    assert r(prompts=["p"], completions=["x"]) is None
