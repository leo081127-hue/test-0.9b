"""GRPO(RL 階段)測試:reward 機制 + trainer 行為 + RL 審計工具。

為什麼這裡不斷言「RL 一定提升 reward」:CPU 上隨機初始化的 tiny 模型在
幾十步 SFT 內學不會輸出格式(format_rate≈0)→ reward 全 0 → 群組內方差 0 →
GRPO 沒有任何梯度訊號(TRL log 的 frac_reward_zero_std=1 即此狀態)。
這是 CPU smoke 的已知上限;真實機器上的 RL 收益驗證流程是:
  1. SFT(+DPO)後先跑 --reward-audit 存下 before 基線
  2. GRPO 训练
  3. 再跑 --reward-audit,確認 reward_mean 上升且不是只靠 format_rate

本檔覆蓋:
  1. reward 单调性:完整光譜(最好>對方向無覆蓋>錯方向>亂碼)分數遞減
  2. trainer 跑完:loss 有限、log 含 reward/kl 指標、無 NaN、adapter 存檔
  3. reward 饑餓(群組方差=0)時不崩、loss≈0(數值穩定)
  4. rows 不能被 num_generations 整除 → 自動 pad
  5. run_audit(stub model 回固定 token)→ 與手算分項完全一致
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
torch = pytest.importorskip("torch")
trl = pytest.importorskip("trl")

from common import build_prompt
from train.grpo import make_reward, run_audit

GOOD_HOME = (
    "<推理>\n- 測試理由\n</推理>\n"
    "最終預測: 主隊\n機率: 主隊 0.80, 客隊 0.20\n讓分覆蓋: 主隊 0.70, 客隊 0.30"
)
WRONG_SIDE = (
    "最終預測: 客隊\n機率: 主隊 0.20, 客隊 0.80\n讓分覆蓋: 主隊 0.30, 客隊 0.70"
)


# ------------------------------------------------------------- 1. reward 光譜

def _score(text, y=1, c=1):
    return make_reward()(prompts=["p"], completions=[text],
                         outcome_home=[y], cover_home=[c])[0]


def test_reward_monotonic_spectrum():
    """GRPO 的梯度訊號來自 reward 差異:分數必須隨預測品質單調遞減。"""
    best = _score(GOOD_HOME)                       # 1 + 0.96 + 0.5 + 0.273
    good_no_cover = _score("最終預測: 主隊\n機率: 主隊 0.80, 客隊 0.20")
    wrong = _score(WRONG_SIDE)                     # 方向、機率、覆蓋全反
    garbage = _score("完全沒有格式的亂碼")
    assert best == pytest.approx(2.733, abs=1e-9)
    assert good_no_cover == pytest.approx(2.46, abs=1e-9)
    assert best > good_no_cover > wrong > garbage
    assert garbage == 0.0
    # 群組內(同 prompt、不同 completion)必有差異 → GRPO 有訊號
    assert (best - wrong) > 0.5


# ------------------------------------------------------------- 2~4. trainer

def _game(i: int) -> dict:
    return {
        "season": 2024, "home": f"H{i}", "away": f"A{i}",
        "home_form_w": 5, "home_form_l": 5, "away_form_w": 4, "away_form_l": 6,
        "home_avg_pts": 100.0, "away_avg_pts": 99.0, "home_rest": 2, "away_rest": 3,
        "home_record": "15-10", "away_record": "12-13",
        "h2h_home_w": 2, "h2h_away_w": 3,
        "open_spread": -3.5, "close_spread": -3.0, "open_total": 220, "close_total": 219,
        "open_ml_home": 1.80, "open_ml_away": 2.00,
        "close_ml_home": 1.75, "close_ml_away": 2.05,
    }


def run_grpo(*extra, n_rows=6, steps=4, out, timeout=900) -> str:
    j = out / "rows.jsonl"
    rows = [{"prompt": [{"role": "user", "content": build_prompt(_game(i))}],
             "outcome_home": i % 2, "cover_home": (i + 1) % 2} for i in range(n_rows)]
    j.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                 encoding="utf-8")
    cmd = [PY, "train/grpo.py", "--model-id", str(out / "tiny"), "--dtype", "float32",
           "--train-jsonl", j, "--output-dir", out / "grpo",
           "--lora",
           "--num-generations", "2", "--batch-size", "2",
           "--max-completion", "32", "--max-steps", str(steps),
           "--beta", "0.04", "--lr", "5e-6", "--logging-steps", "2", *map(str, extra)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise AssertionError(f"GRPO run failed ({r.returncode}):\n{r.stdout[-2000:]}\n"
                             f"{r.stderr[-2000:]}")
    return r.stdout


@pytest.fixture(scope="module")
def grpo_env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("grpo_rl")
    # 資料(smoke_tiny_model 需要 jsonl 來訓練 tokenizer)
    rows = [{"prompt": [{"role": "user", "content": build_prompt(_game(i))}],
             "response": GOOD_HOME} for i in range(8)]
    (tmp / "x.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows), encoding="utf-8")
    r = subprocess.run([PY, "tools/smoke_tiny_model.py",
                        "--train-jsonl", str(tmp / "x.jsonl"), "--out", tmp / "tiny",
                        "--hidden", "256", "--layers", "4", "--vocab", "2048"],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    log = run_grpo(out=tmp, n_rows=6, steps=4)
    return {"tmp": tmp, "log": log}


def test_grpo_trainer_runs_and_logs_finite_metrics(grpo_env):
    log = grpo_env["log"]
    # 有 per-step log,含 GRPO 關鍵指標
    assert "'loss':" in log and "'reward'" in log and "'kl'" in log
    assert "nan" not in log.lower()
    # loss 有限且很小(隨機 tiny 模型 reward 全 0 → 群組 advantage≈0 → 幾乎不更新)
    losses = [float(x) for x in re.findall(r"'loss': '([0-9eE.+-]+)'", log)]
    assert losses and all(l == l and l < 1e-2 for l in losses)
    # adapter 存檔
    d = grpo_env["tmp"] / "grpo"
    assert (d / "adapter_config.json").exists()
    assert (d / "adapter_model.safetensors").exists()


def test_grpo_zero_variance_reward_is_stable(grpo_env):
    """reward 饑餓(所有 completion 同分)→ GRPO 無訊號但必須數值穩定。"""
    log = grpo_env["log"]
    assert "'frac_reward_zero_std': '1'" in log  # 確認確實處於零方差狀態
    assert "'reward_std': '0'" in log
    assert "GRPO adapter saved" in log


def test_grpo_row_count_padding(grpo_env, tmp_path_factory):
    """rows 不能被 num_generations 整除 → 自動 pad(TRL 硬性要求)。"""
    import shutil
    tmp = tmp_path_factory.mktemp("grpo_pad")
    shutil.copytree(grpo_env["tmp"] / "tiny", tmp / "tiny")
    log = run_grpo(out=tmp, n_rows=5, steps=2)
    assert "pad" in log
    assert "GRPO rows: 6" in log


# ------------------------------------------------------------- 5. run_audit

class _StubModel:
    """generate 固定回「prompt tokens + 固定回應 tokens」,不真正训练。"""

    def __init__(self, tok, response_text):
        self._resp_ids = torch.tensor(
            [tok(response_text, add_special_tokens=False)["input_ids"]] * 1)
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def generate(self, input_ids, **_kw):
        """TRL/transformers 以 generate(**inputs, ...) 呼叫 → input_ids 為 kwarg。"""
        return torch.cat([input_ids,
                          self._resp_ids.expand(len(input_ids), -1)], dim=1)


def test_run_audit_stub_matches_hand_computed(grpo_env):
    """run_audit 的每分項都必須等於手算值(審計工具的可信度)。"""
    tok = grpo_env["tmp"] / "tiny"
    from transformers import AutoTokenizer
    ftok = AutoTokenizer.from_pretrained(str(tok))
    rows = [{"prompt": [{"role": "user", "content": build_prompt(_game(i))}],
             "outcome_home": 1, "cover_home": 1} for i in range(4)]
    s = run_audit(_StubModel(ftok, GOOD_HOME), ftok, rows, n=4, max_new=96)
    # GOOD_HOME @ y=1, c=1:format 1 + brier 0.96 + direction 0.5 + cover 0.273
    assert s["n"] == 4
    assert s["format_rate"] == pytest.approx(1.0)
    assert s["brier_mean"] == pytest.approx(0.96, abs=1e-9)
    assert s["direction_mean"] == pytest.approx(0.5, abs=1e-9)
    assert s["cover_mean"] == pytest.approx(0.3 * (1 - 0.09), abs=1e-9)
    assert s["reward_mean"] == pytest.approx(1.0 + 0.96 + 0.5 + 0.3 * 0.91, abs=1e-9)


def test_run_audit_garbage_zero(grpo_env):
    from transformers import AutoTokenizer
    ftok = AutoTokenizer.from_pretrained(str(grpo_env["tmp"] / "tiny"))
    rows = [{"prompt": [{"role": "user", "content": "x"}],
             "outcome_home": 1, "cover_home": 1} for _ in range(3)]
    s = run_audit(_StubModel(ftok, "完全沒有格式的亂碼"), ftok, rows, n=3, max_new=96)
    assert s["reward_mean"] == 0.0
    assert s["format_rate"] == 0.0
