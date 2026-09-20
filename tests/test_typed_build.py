"""--format typed(Jev 風格)管線:build 出來的 SFT/DPO 資料都是可解析的嚴格 JSON。"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(*args, timeout=600):
    r = subprocess.run([PY, *map(str, args)], cwd=ROOT, capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode != 0:
        raise AssertionError(f"failed ({r.returncode}):\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r.stdout


@pytest.fixture(scope="module")
def typed_build(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("typed_build")
    matches = tmp / "soccer.csv"
    run("data/generate_demo_soccer.py", "--seasons", "2", "--games-per-season", "60",
        "--seed", "5", "--out", matches)
    out = tmp / "out"
    run("data/build_dataset.py", "--matches", matches, "--out", out,
        "--sport", "soccer", "--format", "typed", "--max-pairs", "50")
    data = {}
    for part in ("train", "val", "test"):
        data[part] = [json.loads(l) for l in (out / f"{part}.jsonl").read_text().splitlines()]
    data["pairs"] = [json.loads(l) for l in (out / "dpo_pairs.jsonl").read_text().splitlines()]
    return data


def test_typed_responses_are_valid_json(typed_build):
    for part in ("train", "val", "test"):
        for row in typed_build[part]:
            assert row["format"] == "typed"
            obj = json.loads(row["response"])  # 直接可 json.loads(嚴格 JSON)
            assert obj["choice"] in ("主隊", "和局", "客隊")
            assert len(obj["probabilities"]) == 3
            assert abs(sum(obj["probabilities"]) - 1.0) < 1e-6
            assert 0.0 <= obj["confidence"] <= 1.0
            # choice 必須與 probabilities argmax 一致
            assert obj["choice"] == ["主隊", "和局", "客隊"][obj["probabilities"].index(max(obj["probabilities"]))]


def test_typed_prompts_ask_for_json(typed_build):
    for row in typed_build["train"][:5]:
        prompt = row["prompt"][0]["content"]
        assert "JSON" in prompt
        assert "最終預測: 主隊 / 和局" not in prompt  # 不再是 prose 格式指令


def test_typed_dpo_pairs_parseable(typed_build):
    assert len(typed_build["pairs"]) > 0
    for pair in typed_build["pairs"]:
        c = json.loads(pair["chosen"]); rj = json.loads(pair["rejected"])
        assert c["choice"] != rj["choice"]  # rejected 是反方向


def test_typed_grpo_reward_scorable(typed_build):
    # GRPO reward 要能吃 typed 回應(格式分不為 0)
    from train.grpo import make_reward
    r = make_reward()
    row = typed_build["test"][0]
    s = r(prompts=["p"], completions=[row["response"]], sport=["soccer"],
          outcome=[row["outcome"]])
    assert s[0] > 1.0  # 至少格式 1.0
