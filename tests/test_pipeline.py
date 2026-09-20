"""整合測試:無網路、無 GPU 的完整管線(極小規模)。

跑:generate → build_dataset → tiny model → SFT(含 val)→ DPO → GRPO →
    evaluate(baseline + LLM)→ predict → merge_adapter。
總時長在本機 2 核 CPU 約 2~4 分鐘。
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(step: str, *args: str) -> str:
    cmd = [PY, *map(str, args)]
    print(f"\n[pipeline:{step}] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1500)
    if r.returncode != 0:
        raise AssertionError(f"step '{step}' failed ({r.returncode}):\n"
                             f"STDOUT:\n{r.stdout[-3000:]}\nSTDERR:\n{r.stderr[-3000:]}")
    return r.stdout


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipe")
    out = {}
    matches = tmp / "matches.csv"
    datadir = tmp / "data"
    datadir.mkdir()

    run("generate", "data/generate_demo_data.py",
        "--seasons", "2", "--games-per-season", "120", "--seed", "5", "--out", matches)
    run("build", "data/build_dataset.py",
        "--matches", matches, "--out", datadir, "--max-pairs", "30")
    run("tiny", "tools/smoke_tiny_model.py",
        "--train-jsonl", datadir / "train.jsonl", "--out", tmp / "tiny")

    sft_dir = tmp / "sft"
    out["sft_log"] = run("sft", "train/sft.py",
                         "--model-id", tmp / "tiny", "--dtype", "float32",
                         "--train-jsonl", datadir / "train.jsonl",
                         "--val-jsonl", datadir / "val.jsonl",
                         "--adapter-dir", sft_dir,
                         "--max-steps", "6", "--batch-size", "2",
                         "--max-len", "512", "--warmup-steps", "2",
                         "--logging-steps", "2", "--eval-steps", "3")
    dpo_dir = tmp / "dpo"
    run("dpo", "train/dpo.py",
        "--model-id", tmp / "tiny", "--dtype", "float32",
        "--adapter-dir", sft_dir, "--pairs", datadir / "dpo_pairs.jsonl",
        "--output-dir", dpo_dir, "--max-steps", "3", "--batch-size", "2",
        "--logging-steps", "3")
    grpo_dir = tmp / "grpo"
    run("grpo", "train/grpo.py",
        "--model-id", tmp / "tiny", "--dtype", "float32",
        "--train-jsonl", datadir / "train.jsonl", "--output-dir", grpo_dir,
        "--lora", "--num-generations", "2", "--batch-size", "2",
        "--max-completion", "48", "--max-steps", "1")
    report = tmp / "report.json"
    plot = tmp / "cal.png"
    run("eval_nomodel", "eval/evaluate.py",
        "--model-id", tmp / "tiny", "--matches-csv", matches,
        "--test-jsonl", datadir / "test.jsonl", "--report", report,
        "--plot", plot, "--no-model")
    report_llm = tmp / "report_llm.json"
    run("eval_llm", "eval/evaluate.py",
        "--model-id", tmp / "tiny", "--dtype", "float32",
        "--adapter-dir", dpo_dir, "--matches-csv", matches,
        "--test-jsonl", datadir / "test.jsonl", "--report", report_llm,
        "--limit", "2")
    out["predict_log"] = run("predict", "infer/predict.py",
                             "--model-id", tmp / "tiny", "--dtype", "float32",
                             "--adapter-dir", dpo_dir, "--csv", matches, "--row", "100")
    merged = tmp / "merged"
    run("merge", "tools/merge_adapter.py",
        "--model-id", tmp / "tiny", "--dtype", "float32",
        "--adapter-dir", dpo_dir, "--out-dir", merged)

    out["tmp"] = tmp
    out["sft_dir"] = sft_dir
    out["dpo_dir"] = dpo_dir
    out["grpo_dir"] = grpo_dir
    out["report"] = report
    out["report_llm"] = report_llm
    out["plot"] = plot
    out["merged"] = merged
    return out


def test_sft_artifacts(pipeline):
    d = pipeline["sft_dir"]
    assert (d / "adapter_config.json").exists()
    assert (d / "adapter_model.safetensors").exists()
    assert (d / "metadata.json").exists()
    assert "adapter saved" in pipeline["sft_log"]
    # 有 loss 記錄,且 SFT 有跑 val(--val-jsonl 路徑)
    assert re.search(r"'loss':", pipeline["sft_log"])


def test_dpo_artifacts(pipeline):
    d = pipeline["dpo_dir"]
    assert (d / "adapter_config.json").exists()
    assert (d / "metadata.json").exists()
    meta = json.loads((d / "metadata.json").read_text())
    assert meta["stage"] == "dpo"


def test_grpo_artifacts(pipeline):
    assert (pipeline["grpo_dir"] / "adapter_config.json").exists()
    assert (pipeline["grpo_dir"] / "adapter_model.safetensors").exists()


def test_report_baselines(pipeline):
    rep = json.loads(pipeline["report"].read_text())
    for k in ("Market_close", "LogReg", "Coin"):
        assert k in rep
        assert rep[k]["brier"] is not None
        assert 0.0 < rep[k]["brier"] < 0.5
    # 分季統計(2 個賽季 → test 至少 1 季)
    assert rep.get("by_season"), "report 應含 by_season"
    # 校準圖
    assert pipeline["plot"].exists() and pipeline["plot"].stat().st_size > 5000


def test_report_llm_keys(pipeline):
    rep = json.loads(pipeline["report_llm"].read_text())
    assert "LLM" in rep and rep["LLM"] is not None
    # tiny 隨機模型:parse 多半失敗,但欄位結構必須在
    assert "parse_error_rate" in rep["LLM"]
    assert 0.0 <= rep["LLM"]["parse_error_rate"] <= 1.0


def test_predict_output(pipeline):
    log = pipeline["predict_log"]
    assert "模型輸出" in log
    assert "解析結果" in log


def test_merged_model_deployment_ready(pipeline):
    d = pipeline["merged"]
    assert (d / "config.json").exists()
    assert (d / "model.safetensors").exists()
    assert (d / "tokenizer.json").exists()
    # 合併後可直接用 HF 載入(不需 PEFT)
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "import torch\n"
        "m = AutoModelForCausalLM.from_pretrained(%r, dtype=torch.float32)\n"
        "t = AutoTokenizer.from_pretrained(%r)\n"
        "ids = t('測試', return_tensors='pt')\n"
        "out = m.generate(**ids, max_new_tokens=4)\n"
        "print('MERGED_OK', out.shape)" % (str(ROOT), str(d), str(d))
    )
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "MERGED_OK" in r.stdout
