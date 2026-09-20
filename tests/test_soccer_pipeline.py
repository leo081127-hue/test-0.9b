"""足球 1X2 整合管線(無網路、無 GPU、極小規模):

generate(soccer) → build(三類 teacher) → tiny → SFT → DPO →
評估(baseline + LLM)→ 單筆預測(soccer)→ 1X2 回測。
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(step: str, *args: str, timeout: int = 1200) -> str:
    cmd = [PY, *map(str, args)]
    print(f"\n[soccer-pipeline:{step}] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise AssertionError(f"step '{step}' failed ({r.returncode}):\n"
                             f"STDOUT:\n{r.stdout[-2500:]}\nSTDERR:\n{r.stderr[-2500:]}")
    return r.stdout


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("soccer_pipe")
    out = {}
    matches = tmp / "matches.csv"
    datadir = tmp / "data"
    datadir.mkdir()

    run("generate", "data/generate_demo_soccer.py",
        "--seasons", "2", "--games-per-season", "90", "--seed", "5", "--out", matches)
    run("build", "data/build_dataset.py",
        "--matches", matches, "--out", datadir, "--sport", "soccer", "--max-pairs", "20")
    run("tiny", "tools/smoke_tiny_model.py",
        "--train-jsonl", datadir / "train.jsonl", "--out", tmp / "tiny",
        "--hidden", "256", "--layers", "4", "--vocab", "2048")

    sft = tmp / "sft"
    out["sft_log"] = run("sft", "train/sft.py",
                         "--model-id", tmp / "tiny", "--dtype", "float32",
                         "--train-jsonl", datadir / "train.jsonl",
                         "--adapter-dir", sft,
                         "--max-steps", "4", "--batch-size", "2", "--max-len", "512",
                         "--logging-steps", "2")
    dpo = tmp / "dpo"
    run("dpo", "train/dpo.py",
        "--model-id", tmp / "tiny", "--dtype", "float32",
        "--adapter-dir", sft, "--pairs", datadir / "dpo_pairs.jsonl",
        "--output-dir", dpo, "--max-steps", "2", "--batch-size", "2",
        "--logging-steps", "2")

    rep_b = tmp / "report_b.json"
    run("eval_b", "eval/evaluate.py",
        "--model-id", tmp / "tiny", "--sport", "soccer",
        "--matches-csv", matches, "--test-jsonl", datadir / "test.jsonl",
        "--report", rep_b, "--no-model")
    out["report_b"] = json.loads(rep_b.read_text())

    rep_l = tmp / "report_l.json"
    plot = tmp / "cal.png"
    run("eval_l", "eval/evaluate.py",
        "--model-id", tmp / "tiny", "--dtype", "float32", "--sport", "soccer",
        "--adapter-dir", dpo, "--matches-csv", matches,
        "--test-jsonl", datadir / "test.jsonl", "--report", rep_l,
        "--plot", plot, "--limit", "2")
    out["report_l"] = json.loads(rep_l.read_text())
    out["plot"] = plot

    out["predict_log"] = run("predict", "infer/predict.py",
                             "--model-id", tmp / "tiny", "--dtype", "float32",
                             "--adapter-dir", dpo, "--csv", matches, "--row", "100")

    # 回測:teacher 機率當模型代理(與 demo 做法一致)
    import pandas as pd
    df = pd.read_csv(matches).set_index("match_id")
    recs = []
    for line in (datadir / "test.jsonl").read_text().splitlines():
        d = json.loads(line)
        m = df.loc[d["match_id"]]
        recs.append({"match_id": d["match_id"], "date": d["date"], "season": "",
                     "p_home": d["teacher_p"][0], "p_draw": d["teacher_p"][1],
                     "p_away": d["teacher_p"][2],
                     "open_ml_home": m.open_ml_home, "open_ml_draw": m.open_ml_draw,
                     "open_ml_away": m.open_ml_away,
                     "close_ml_home": m.close_ml_home, "close_ml_draw": m.close_ml_draw,
                     "close_ml_away": m.close_ml_away,
                     "outcome": d["outcome"]})
    preds = tmp / "preds.csv"
    pd.DataFrame(recs).to_csv(preds, index=False)
    bt = tmp / "backtest.json"
    equity = tmp / "equity.png"
    out["backtest_log"] = run("backtest", "eval/backtest.py",
                              "--preds", preds, "--matches-csv", matches,
                              "--sport", "soccer", "--threshold", "0.55",
                              "--report", bt, "--plot", equity)
    out["backtest"] = json.loads(bt.read_text())
    out["equity"] = equity
    out["tmp"] = tmp
    out["sft_dir"] = sft
    out["dpo_dir"] = dpo
    return out


def test_soccer_sft_dpo_artifacts(pipe):
    for d in (pipe["sft_dir"], pipe["dpo_dir"]):
        assert (d / "adapter_config.json").exists()
        assert (d / "adapter_model.safetensors").exists()
    meta = json.loads((pipe["dpo_dir"] / "metadata.json").read_text())
    assert meta["stage"] == "dpo"


def test_soccer_report_baselines(pipe):
    rep = pipe["report_b"]
    for k in ("Market_close", "LogReg", "Coin"):
        assert k in rep and rep[k]["brier"] is not None
    # 市場收盤必須優於均分 coin(三類 uniform brier = 2/3)
    assert rep["Market_close"]["brier"] < 0.66
    assert rep["Coin"]["brier"] == pytest.approx(2 / 3, abs=1e-6)
    assert rep["sport"] == "soccer"
    assert rep.get("by_season"), "report 應含 by_season"


def test_soccer_report_llm(pipe):
    rep = pipe["report_l"]
    assert "LLM" in rep and rep["LLM"] is not None
    assert "parse_error_rate" in rep["LLM"]
    assert 0.0 <= rep["LLM"]["parse_error_rate"] <= 1.0
    assert pipe["plot"].exists() and pipe["plot"].stat().st_size > 3000


def test_soccer_predict_output(pipe):
    log = pipe["predict_log"]
    assert "sport: soccer" in log
    assert "模型輸出" in log and "解析結果" in log
    assert "和局機率" in log


def test_soccer_backtest_structure(pipe):
    bt = pipe["backtest"]
    assert bt["sport"] == "soccer"
    for line in ("model", "market_open"):
        assert line in bt
        assert "clv" in bt[line] and "roi" in bt[line] and "n_bets" in bt[line]
    assert bt["threshold_scan"] and all("roi" in s for s in bt["threshold_scan"])
    assert pipe["equity"].exists() and pipe["equity"].stat().st_size > 3000
    # CLV 有限(不保證方向:demo 資料有設計誤價,但 tiny teacher 代理可能不穩)
    for line in ("model", "market_open"):
        v = bt[line]["clv"]
        if v is not None:
            assert v == v  # not NaN
