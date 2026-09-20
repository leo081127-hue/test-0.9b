"""typed(Jev 風格)輸出、confidence 統計、聯賽 base-rate 錨定。"""
import json

import numpy as np
import pandas as pd
import pytest

from common import (FEATURE_COLS, attach_league_rates, build_prompt,
                    build_typed_response, confidence_from_probs,
                    confidence_gated_stats, feature_matrix,
                    format_game_features, parse_response, parse_response_any,
                    parse_typed_response)


# ------------------------------------------------------------- typed roundtrip

def test_typed_soccer_roundtrip():
    s = build_typed_response(0.45, p_draw=0.25, sport="soccer")
    obj = json.loads(s)
    assert obj["choice"] == "主隊"
    assert len(obj["probabilities"]) == 3
    assert abs(sum(obj["probabilities"]) - 1.0) < 1e-6
    assert obj["confidence"] == pytest.approx(0.45 - 0.30, abs=1e-3)
    p = parse_typed_response(s, "soccer")
    assert p["format_ok"] and p["typed"]
    assert p["p_home"] == pytest.approx(0.45, abs=1e-3)
    assert p["p_draw"] == pytest.approx(0.25, abs=1e-3)
    assert p["p_away"] == pytest.approx(0.30, abs=1e-3)
    assert p["pred"] == "主隊"


def test_typed_basketball_roundtrip():
    s = build_typed_response(0.62, 0.55, sport="basketball")
    obj = json.loads(s)
    assert obj["choice"] == "主隊"
    assert obj["probabilities"] == [0.62, 0.38]
    assert obj["cover_probabilities"] == [0.55, 0.45]
    p = parse_typed_response(s)
    assert p["format_ok"]
    assert p["p_home"] == pytest.approx(0.62, abs=1e-6)
    assert p["cover_home"] == pytest.approx(0.55, abs=1e-6)
    assert p["confidence"] == pytest.approx(0.24, abs=1e-6)


def test_typed_lenient_extraction():
    # 模型在 JSON 前後加字 / 包 code fence 也要能解析
    core = '{"choice": "和局", "probabilities": [0.30, 0.44, 0.26], "confidence": 0.18}'
    p = parse_typed_response(f"好的,我的判斷如下:\n```json\n{core}\n```", "soccer")
    assert p["format_ok"] and p["p_draw"] == pytest.approx(0.44, abs=1e-6)
    p2 = parse_typed_response(core + "\n(以上為我的預測)", "soccer")
    assert p2["format_ok"]


def test_typed_garbage_and_mismatch():
    assert not parse_typed_response("亂碼", "soccer")["format_ok"]
    assert not parse_typed_response('{"hello": 1}', "soccer")["format_ok"]
    # basketball 的 typed 給 soccer 解析 → 長度不符 → 拒
    s = build_typed_response(0.6, 0.5, sport="basketball")
    assert not parse_typed_response(s, "soccer")["format_ok"]
    # 負機率 / 和為 0 → 拒
    assert not parse_typed_response('{"choice":"主隊","probabilities":[-1,2,2],"confidence":0.5}',
                                    "soccer")["format_ok"]
    assert not parse_typed_response('{"probabilities":[0,0,0]}', "soccer")["format_ok"]


def test_parse_response_any_prefers_prose_falls_back_to_typed():
    prose = "最終預測: 主隊\n機率: 主隊 0.40, 和局 0.30, 客隊 0.30"
    out = parse_response_any(prose, "soccer")
    assert out["format_ok"] and not out.get("typed", False)
    typed = build_typed_response(0.5, p_draw=0.2, sport="soccer")
    out2 = parse_response_any(typed, "soccer")
    assert out2["format_ok"] and out2["typed"]
    out3 = parse_response_any("完全沒有格式", "soccer")
    assert not out3["format_ok"]
    # basketball:typed 帶 cover
    tb = build_typed_response(0.7, 0.6, sport="basketball")
    out4 = parse_response_any("blah " + tb)
    assert out4["format_ok"] and out4["cover_home"] == pytest.approx(0.6, abs=1e-6)


# ------------------------------------------------------------- confidence

def test_confidence_from_probs_hand():
    assert confidence_from_probs([0.45, 0.25, 0.30]) == pytest.approx(0.15)
    assert confidence_from_probs([0.5, 0.5]) == 0.0
    assert confidence_from_probs([0.9, 0.1]) == pytest.approx(0.8)


def test_confidence_gated_stats_hand():
    # 4 筆:兩筆高 conf(0.7, 0.6)全對、兩筆低 conf(0.02, 0.01)全錯 → acc 隨 conf 上升
    probs = [[0.8, 0.1, 0.1], [0.75, 0.15, 0.1], [0.35, 0.33, 0.32], [0.34, 0.33, 0.33]]
    ys = [0, 0, 2, 1]  # 第 4 筆平手 0.34/0.33/0.33 → argmax=主隊 ≠ 和局 → 錯
    stats = confidence_gated_stats(probs, ys, "soccer", n_bins=2)
    assert [s["n"] for s in stats] == [2, 2]
    assert stats[0]["acc"] == 0.0 and stats[1]["acc"] == 1.0
    # 空輸入
    assert confidence_gated_stats([], [], "soccer") == []


# ------------------------------------------------------------- league rates

def test_attach_league_rates_no_leak_and_prior():
    # 4 天、每天 1 場:W(0) D(1) L(2) W(0)
    df = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
        "margin": [1.0, 0.0, -1.0, 2.0],
    })
    out = attach_league_rates(df, "soccer", pseudo=20)
    # 第一場:只有先驗 9/5/6 / 20
    assert out["lr_home"].iloc[0] == pytest.approx(9 / 20)
    assert out["lr_draw"].iloc[0] == pytest.approx(5 / 20)
    assert out["lr_away"].iloc[0] == pytest.approx(6 / 20)
    # 第二場:先驗 + 第一場主勝
    assert out["lr_home"].iloc[1] == pytest.approx(10 / 21)
    # 最後一場:前 3 場 = 1 主勝 + 1 和 + 1 客 → (9+1, 5+1, 6+1) / 23(不含自己)
    assert out["lr_home"].iloc[3] == pytest.approx(10 / 23)
    assert out["lr_draw"].iloc[3] == pytest.approx(6 / 23)
    assert out["lr_away"].iloc[3] == pytest.approx(7 / 23)


def test_attach_league_rates_same_day_no_mutual_look():
    # 同一天兩場:兩場都只看「昨天之前」
    df = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-02"],
        "margin": [1.0, 1.0, -1.0],
    })
    out = attach_league_rates(df, "soccer", pseudo=20)
    # day2 兩場的 base rate 相同(都只含 day1)
    assert out["lr_home"].iloc[1] == pytest.approx(out["lr_home"].iloc[2])


def test_attach_league_rates_basketball_binary():
    df = pd.DataFrame({"date": ["a", "b"], "margin": [1.0, -2.0]})
    out = attach_league_rates(df, "basketball", pseudo=20)
    assert "lr_home" in out.columns and "lr_draw" not in out.columns
    assert out["lr_home"].iloc[0] == pytest.approx(0.55)
    assert out["lr_home"].iloc[1] == pytest.approx(11 / 21)


def test_feature_matrix_includes_lr_cols():
    assert {"lr_home", "lr_draw", "lr_away"} <= set(FEATURE_COLS)
    df = pd.DataFrame({
        "home_form_w": [3], "home_form_l": [2], "away_form_w": [1], "away_form_l": [4],
        "lr_home": [0.48], "lr_draw": [0.24], "lr_away": [0.28],
        "close_ml_home": [2.0], "close_ml_away": [3.0],
    })
    X, names = feature_matrix(df)
    assert "lr_home" in names
    assert X[0][names.index("lr_home")] == pytest.approx(0.48)


def test_prompt_shows_base_rate_line_only_when_present():
    g = {"season": "2024", "home": "A", "away": "B", "home_form_w": 5, "home_form_d": 1,
         "home_form_l": 4, "away_form_w": 3, "away_form_d": 2, "away_form_l": 5,
         "home_avg_pts": 1.4, "away_avg_pts": 1.0, "home_rest": 2, "away_rest": 3,
         "home_record": "5-1-4", "away_record": "3-2-5", "h2h_home_w": 2, "h2h_away_w": 3,
         "open_ml_home": 2.1, "open_ml_draw": 3.3, "open_ml_away": 3.4,
         "close_ml_home": 2.0, "close_ml_draw": 3.2, "close_ml_away": 3.3}
    text = format_game_features(g, "soccer")
    assert "本聯賽歷史" not in text
    g2 = dict(g, lr_home=0.47, lr_draw=0.23, lr_away=0.30)
    assert "本聯賽歷史結果分佈" in format_game_features(g2, "soccer")
    assert "47.0%" in format_game_features(g2, "soccer")


def test_typed_prompt_template():
    g = {"season": "2024", "home": "A", "away": "B"}
    prose = build_prompt(g, "soccer")
    typed = build_prompt(g, "soccer", fmt="typed")
    assert "最終預測" in prose and "JSON" not in prose
    assert "JSON" in typed and "最終預測: 主隊 / 和局" not in typed
    assert "probabilities" in typed
    tb = build_prompt(g, "basketball", fmt="typed")
    assert "cover_probabilities" in tb
