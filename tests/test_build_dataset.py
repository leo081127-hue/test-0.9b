"""build_dataset.py 測試:時間切分正確、jsonl schema、回應與 teacher 一致、DPO pairs。"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from common import P_AWAY, P_HOME, parse_response

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("build")
    matches = tmp / "matches.csv"
    subprocess.run(
        [sys.executable, str(ROOT / "data/generate_demo_data.py"),
         "--seasons", "2", "--games-per-season", "200", "--seed", "11",
         "--out", str(matches)],
        check=True, cwd=ROOT, capture_output=True,
    )
    out = tmp / "out"
    subprocess.run(
        [sys.executable, str(ROOT / "data/build_dataset.py"),
         "--matches", str(matches), "--out", str(out), "--max-pairs", "50"],
        check=True, cwd=ROOT, capture_output=True,
    )
    splits = {
        name: [json.loads(l) for l in open(out / f"{name}.jsonl", encoding="utf-8")]
        for name in ("train", "val", "test")
    }
    pairs = [json.loads(l) for l in open(out / "dpo_pairs.jsonl", encoding="utf-8")]
    return {"out": out, "matches": pd.read_csv(matches), **splits, "pairs": pairs}


def test_time_split_no_leakage(built):
    tr, va, te = built["train"], built["val"], built["test"]
    max_tr = max(r["date"] for r in tr)
    min_va = min(r["date"] for r in va)
    max_va = max(r["date"] for r in va)
    min_te = min(r["date"] for r in te)
    # 同一天可跨 split(按位置切),但不能反序
    assert max_tr <= min_va
    assert max_va <= min_te
    # match_id 不重疊
    ids = {r["match_id"] for r in tr + va + te}
    assert len(ids) == len(tr) + len(va) + len(te)
    # 比例約 70/15/15(±2)
    n = len(tr) + len(va) + len(te)
    assert abs(len(tr) - 0.70 * n) <= 2
    assert abs(len(va) - 0.15 * n) <= 2


def test_jsonl_schema(built):
    for name in ("train", "val", "test"):
        for row in built[name][:5]:
            assert set(row) >= {
                "match_id", "date", "prompt", "response",
                "outcome_home", "cover_home", "teacher_p_home",
                "teacher_cover_home", "market_p_home_close", "close_spread",
            }
            assert row["prompt"][0]["role"] == "user"
            assert len(row["prompt"][0]["content"]) > 100
            assert row["outcome_home"] in (0, 1)
            assert row["cover_home"] in (0, 1)


def test_response_format_and_teacher_consistency(built):
    for name in ("train", "val", "test"):
        for row in built[name][:20]:
            r = parse_response(row["response"])
            assert r["format_ok"], row["match_id"]
            # 回應裡的機率 ≈ teacher 機率(兩位小數;第 3 位 rounding 邊界容差 1 cent)
            assert abs(r["p_home"] - row["teacher_p_home"]) <= 0.011
            assert abs(r["cover_home"] - row["teacher_cover_home"]) <= 0.011
            # 預測方向 == 顯示機率 argmax(0.5 邊界 ±1 cent 內不檢查,rounding 可翻方向)
            if abs(r["p_home"] - 0.5) > 0.011:
                expect = P_HOME if r["p_home"] >= 0.5 else P_AWAY
                assert r["pred"] == expect
            # 機率在合法區間且加和=1
            assert 0.02 <= r["p_home"] <= 0.98
            assert abs(r["p_home"] + r["p_away"] - 1.0) < 1e-9


def test_outcome_fields_match_source(built):
    m = built["matches"].set_index("match_id")
    for row in built["test"]:
        src = m.loc[row["match_id"]]
        assert row["outcome_home"] == int(src["home_win"])
        assert row["cover_home"] == int(src["margin"] > src["close_spread"])


def test_dpo_pairs(built):
    assert len(built["pairs"]) == 50
    for p in built["pairs"][:10]:
        assert p["chosen"] != p["rejected"]
        rc, rr = parse_response(p["chosen"]), parse_response(p["rejected"])
        assert rc["format_ok"] and rr["format_ok"]
        # rejected 必須「實質不同」:預測方向翻轉,或機率大幅偏移
        assert rc["pred"] != rr["pred"] or abs(rc["p_home"] - rr["p_home"]) > 0.05
        # rejected 與 chosen 的讓分覆蓋方向相反
        assert abs(rc["cover_home"] + rr["cover_home"] - 1.0) < 1e-9


def test_prompt_uses_pre_game_features_only(built):
    """prompt 裡的近10場狀態必須等於 csv 的賽前值(生成器已保證),此處抽查欄位存在性。"""
    for row in built["train"][:5]:
        c = row["prompt"][0]["content"]
        assert "近10場" in c and "收盤" in c and "讓分" in c
