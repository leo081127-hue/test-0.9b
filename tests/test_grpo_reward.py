"""train/grpo.py 的 reward 函數:手算分數 + 邊緣情況(格式解析的健壯性)。"""
import pytest

trl = pytest.importorskip("trl")

from train.grpo import make_reward, reward_components

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


# ------------------------------------------------------------- 邊緣情況

def test_reward_prob_renormalized():
    """機率行兩端相加不等於 1 → 解析時重新正規化,reward 用正規化後的 p。"""
    r = make_reward()
    s = r(prompts=["p"], completions=["最終預測: 主隊\n機率: 主隊 0.9, 客隊 0.5"],
          outcome_home=[1], cover_home=[0])
    p_home = 0.9 / 1.4  # 9/14
    expect = 1.0 + (1.0 - (p_home - 1.0) ** 2) + 0.5  # 無覆蓋行 → cover=0
    assert s[0] == pytest.approx(expect, abs=1e-9)


def test_reward_no_cover_line_drops_cover_only():
    r = make_reward()
    no_cover = "最終預測: 主隊\n機率: 主隊 0.80, 客隊 0.20"
    s = r(prompts=["p"], completions=[no_cover], outcome_home=[1], cover_home=[1])
    assert s[0] == pytest.approx(1.0 + 0.96 + 0.5, abs=1e-9)
    # 與完整版的差恰好是覆蓋分項
    full = r(prompts=["p"], completions=[GOOD_HOME], outcome_home=[1], cover_home=[1])
    assert full[0] - s[0] == pytest.approx(0.3 * (1 - 0.09), abs=1e-9)


def test_reward_fullwidth_punctuation():
    """全形冒號/逗號也要能解析(模型常輸出全形符號)。"""
    r = make_reward()
    s = r(prompts=["p"], completions=["機率: 主隊 0.8, 客隊 0.2"],
          outcome_home=[1], cover_home=[0])
    assert s[0] == pytest.approx(1.0 + 0.96 + 0.5, abs=1e-9)  # 無預測行→以機率定方向


def test_reward_away_winner_direction():
    """y=0(客隊贏):預測主隊 → 方向分 0。"""
    r = make_reward()
    s = r(prompts=["p"], completions=["最終預測: 主隊\n機率: 主隊 0.30, 客隊 0.70"],
          outcome_home=[0], cover_home=[0])
    assert s[0] == pytest.approx(1.0 + (1 - 0.09) + 0.0, abs=1e-9)


def test_reward_extreme_probs():
    r = make_reward()
    # p=1.0 全對:格式 1 + Brier 1 + 方向 0.5
    s = r(prompts=["p"], completions=["機率: 主隊 1, 客隊 0"], outcome_home=[1], cover_home=[1])
    assert s[0] == pytest.approx(2.5, abs=1e-9)
    # p=0.0 全錯(主隊 0 機率但主隊贏):Brier 0、方向 0
    s = r(prompts=["p"], completions=["機率: 主隊 0, 客隊 1"], outcome_home=[1], cover_home=[1])
    assert s[0] == pytest.approx(1.0, abs=1e-9)


def test_reward_zero_sum_probs_rejected():
    """0/0 無意義 → 不算 format_ok,總分 0。"""
    r = make_reward()
    s = r(prompts=["p"], completions=["機率: 主隊 0, 客隊 0"], outcome_home=[1], cover_home=[1])
    assert s[0] == 0.0


def test_reward_garbage_after_valid_line_still_scores():
    """解析是 search 不是 fullmatch:有效行後面亂碼不影響分數。"""
    r = make_reward()
    text = GOOD_HOME + "\n\n(後面接了一堆無意義的隨機文字 abc 123 xyz)"
    s = r(prompts=["p"], completions=[text], outcome_home=[1], cover_home=[1])
    s_ref = r(prompts=["p"], completions=[GOOD_HOME], outcome_home=[1], cover_home=[1])
    assert s[0] == pytest.approx(s_ref[0], abs=1e-12)


def test_reward_bytes_input():
    r = make_reward()
    s = r(prompts=["p"], completions=[GOOD_HOME.encode("utf-8")],
          outcome_home=[1], cover_home=[1])
    s_ref = r(prompts=["p"], completions=[GOOD_HOME], outcome_home=[1], cover_home=[1])
    assert s[0] == pytest.approx(s_ref[0], abs=1e-12)


def test_reward_batch_multiple_games():
    """一批多筆:回傳 list,每筆與單筆呼叫一致。"""
    r = make_reward()
    s = r(prompts=["p1", "p2"], completions=[GOOD_HOME, "亂碼"],
          outcome_home=[1, 0], cover_home=[1, 1])
    assert len(s) == 2
    assert s[0] == pytest.approx(
        r(prompts=["p1"], completions=[GOOD_HOME], outcome_home=[1], cover_home=[1])[0],
        abs=1e-12)
    assert s[1] == 0.0


def test_reward_components_sum_equals_reward():
    """reward_components 分項加總 == make_reward 總分(重構一致性)。"""
    r = make_reward()
    for text, y, c in [
        (GOOD_HOME, 1, 1), (WRONG_SIDE, 1, 0), ("亂碼", 0, 0),
        ("機率: 主隊 0.9, 客隊 0.5", 1, 1), ("機率: 主隊 0, 客隊 0", 1, 1),
    ]:
        comp = reward_components(text, float(y), float(c))
        total = comp["format"] + comp["brier"] + comp["direction"] + comp["cover"]
        got = r(prompts=["p"], completions=[text], outcome_home=[y], cover_home=[c])[0]
        assert total == pytest.approx(got, abs=1e-12), (text, comp, got)


def test_reward_components_range_and_garbage():
    comp = reward_components("完全沒有格式", 1.0, 1.0)
    assert comp == {"format": 0.0, "brier": 0.0, "direction": 0.0, "cover": 0.0}
    comp = reward_components(GOOD_HOME, 1.0, 1.0)
    assert 0.0 <= comp["brier"] <= 1.0
    assert comp["direction"] in (0.0, 0.5)
    assert 0.0 <= comp["cover"] <= 0.3
