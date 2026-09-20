"""data/qa_report.py 測試:乾淨資料 0 error;壞資料要抓得出來。"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_qa(csv: Path, out_json: Path) -> dict:
    r = subprocess.run(
        [PY, str(ROOT / "data/qa_report.py"), "--matches", str(csv), "--json", str(out_json)],
        cwd=ROOT, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stderr
    return json.loads(out_json.read_text())


@pytest.fixture(scope="module")
def clean_csv(tmp_path_factory) -> Path:
    tmp = tmp_path_factory.mktemp("qa")
    csv = tmp / "clean.csv"
    subprocess.run(
        [PY, str(ROOT / "data/generate_demo_data.py"),
         "--seasons", "1", "--games-per-season", "80", "--seed", "9", "--out", str(csv)],
        cwd=ROOT, capture_output=True, check=True,
    )
    return csv


def test_clean_data_no_errors(clean_csv, tmp_path):
    rep = run_qa(clean_csv, tmp_path / "qa.json")
    assert rep["n_errors"] == 0, rep["issues"]
    assert rep["n_rows"] == 80


def test_dirty_data_detected(clean_csv, tmp_path):
    df = pd.read_csv(clean_csv)
    # 1) 重複 id
    df.loc[2, "match_id"] = df.loc[0, "match_id"]
    # 2) margin 與比分不符
    df.loc[3, "margin"] = df.loc[3, "home_score"] - df.loc[3, "away_score"] + 5
    # 3) home_win 與 margin 矛盾
    df.loc[4, "home_win"] = 1 - int(df.loc[4, "home_win"])
    # 4) 賠率 <= 1.01
    df.loc[5, "open_ml_home"] = 1.0
    bad = tmp_path / "bad.csv"
    df.to_csv(bad, index=False)

    rep = run_qa(bad, tmp_path / "qa_bad.json")
    checks = {i["check"] for i in rep["issues"] if i["severity"] == "ERROR"}
    assert "duplicate_id" in checks
    assert "margin_consistency" in checks
    assert "win_consistency" in checks
    assert "odds_low:open_ml_home" in checks
    assert rep["n_errors"] >= 4


def test_missing_column_flagged(clean_csv, tmp_path):
    df = pd.read_csv(clean_csv).drop(columns=["close_ml_home"])
    bad = tmp_path / "missing.csv"
    df.to_csv(bad, index=False)
    rep = run_qa(bad, tmp_path / "qa_miss.json")
    checks = {i["check"] for i in rep["issues"] if i["severity"] == "ERROR"}
    assert "required_columns" in checks
