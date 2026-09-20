"""評估:LLM 輸出解析 → 準確率 / Brier / LogLoss / ECE,並對照三個基準:
  1. Market close — 收盤賠率去除抽水後的市場隱含機率(真正要贏的對手)
  2. LogReg     — 線性對數回歸(用 train 時間之前的資料擬合)
  3. Coin       — 固定 0.5

同時回報:讓分覆蓋準確率、格式解析失敗率。輸出 report.json,可選畫 calibration.png。

範例:
  python eval/evaluate.py \
      --model-id IFM/K2-Horizon-0.9B --trust-remote-code --adapter-dir output/dpo \
      --test-jsonl data/out/test.jsonl --matches-csv data/demo/matches.csv \
      --report output/report.json --plot output/calibration.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

from common import _acc_brier_logloss, ece, feature_matrix, implied_prob, parse_response
from train.sft import DTYPES, load_causal_lm


def run_model_eval(args, rows: list[dict]):
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = load_causal_lm(args.model_id, DTYPES[args.dtype], args.trust_remote_code)
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    llm_p, llm_y, llm_cov_p, llm_cov_y, llm_pred_ok = [], [], [], [], []
    llm_rows: list[dict] = []  # 每筆解析成功的:match_id + p_home(分季統計用)
    n_err = 0
    for k, row in enumerate(rows):
        text = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tok.pad_token_id,
            )
        out = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        p = parse_response(out)
        if p["p_home"] is None:
            n_err += 1
            print(f"[{k + 1}/{len(rows)}] {row['match_id']}  parse FAIL (raw: {out[:80]!r})")
            continue
        llm_p.append(p["p_home"])
        llm_y.append(row["outcome_home"])
        llm_rows.append({"match_id": row["match_id"], "p_home": p["p_home"]})
        if p["cover_home"] is not None:
            llm_cov_p.append(p["cover_home"])
            llm_cov_y.append(row["cover_home"])
        pred = p["pred"] or ("主隊" if p["p_home"] >= 0.5 else "客隊")
        llm_pred_ok.append(pred == ("主隊" if row["outcome_home"] else "客隊"))
        if (k + 1) % 10 == 0 or k + 1 == len(rows):
            print(f"[{k + 1}/{len(rows)}] generated")

    m = _acc_brier_logloss(llm_p, llm_y)
    m["ece"] = ece(llm_p, llm_y)
    m["parse_error_rate"] = n_err / max(1, len(rows))
    m["pred_acc"] = float(np.mean(llm_pred_ok)) if llm_pred_ok else None
    if llm_cov_p:
        m["cover_acc"] = float(
            (((np.asarray(llm_cov_p) >= 0.5).astype(float)) ==
             (np.asarray(llm_cov_y) == 1)).mean()
        )
    return m, llm_p, llm_y, llm_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="IFM/K2-Horizon-0.9B")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--matches-csv", required=True, help="完整資料(baseline 要用 train 段擬合)")
    ap.add_argument("--report", default="output/report.json")
    ap.add_argument("--plot", default=None, help="校準曲線 png 路徑")
    ap.add_argument("--limit", type=int, default=None, help="只評估前 N 筆(smoke test)")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--no-model", action="store_true", help="只跑 baseline(LLM 沒訓練好時對照用)")
    args = ap.parse_args()

    rows = []
    with open(args.test_jsonl, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]
    print(f"test rows: {len(rows)}")

    report = {"model_id": args.model_id, "adapter_dir": args.adapter_dir, "n_test": len(rows)}
    llm_p, llm_y, llm_rows = None, None, None

    if not args.no_model:
        report["LLM"], llm_p, llm_y, llm_rows = run_model_eval(args, rows)
    else:
        report["LLM"] = None

    # ---- baselines ----
    df = pd.read_csv(args.matches_csv)
    test_ids = {r["match_id"] for r in rows}
    tm = df[df["match_id"].isin(test_ids)].reset_index(drop=True)
    ym = tm["home_win"].astype(int).values

    mkt_p = np.array([implied_prob(r.close_ml_home, r.close_ml_away)[0] for r in tm.itertuples()])
    mkt = _acc_brier_logloss(mkt_p, ym)
    mkt["ece"] = ece(mkt_p, ym)
    report["Market_close"] = mkt

    boundary = min(r["date"] for r in rows)
    trn = df[df["date"] < boundary]
    X_tr, names = feature_matrix(trn)
    y_tr = trn["home_win"].astype(int).values
    X_te, _ = feature_matrix(tm)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=3000).fit(X_tr, y_tr)
    lr_p = lr.predict_proba(X_te)[:, 1]
    lrm = _acc_brier_logloss(lr_p, ym)
    lrm["ece"] = ece(lr_p, ym)
    report["LogReg"] = lrm

    coin = _acc_brier_logloss(np.full(len(ym), 0.5), ym)
    coin["ece"] = ece(np.full(len(ym), 0.5), ym)
    report["Coin"] = coin

    # ---- 分季統計(看模型在更晚賽季是否退化)----
    if "season" in df.columns:
        season_of = dict(zip(df["match_id"], df["season"]))
        outcome_of = dict(zip(df["match_id"], df["home_win"].astype(int)))
        by_season: dict = {}
        for s in sorted(set(season_of.values())):
            mids = {m for m, se in season_of.items() if se == s}
            entry: dict = {}
            if llm_rows:
                sel = [r for r in llm_rows if r["match_id"] in mids]
                if sel:
                    entry["LLM"] = _acc_brier_logloss(
                        [r["p_home"] for r in sel], [outcome_of[r["match_id"]] for r in sel])
            mt = tm[tm["match_id"].isin(mids)]
            if len(mt) > 0:
                mp = np.array([implied_prob(r.close_ml_home, r.close_ml_away)[0] for r in mt.itertuples()])
                entry["Market"] = _acc_brier_logloss(mp, mt["home_win"].astype(int).values)
            if entry:
                by_season[s] = entry
        report["by_season"] = by_season

    # ---- 輸出 ----
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    def fmt(v, nd=4):
        return f"{v:.{nd}f}" if v is not None else "  n/a "

    print(f"\n{'model':<14}{'acc':>7}{'brier':>9}{'logloss':>10}{'ece':>8}  extra")
    for name in ("LLM", "Market_close", "LogReg", "Coin"):
        m = report[name]
        if m is None:
            continue
        extra = ""
        if name == "LLM":
            extra = f"pred_acc={fmt(m.get('pred_acc'), 3)} cover_acc={fmt(m.get('cover_acc'), 3)} parse_err={fmt(m['parse_error_rate'], 3)}"
        print(f"{name:<14}{fmt(m['acc'], 3):>7}{fmt(m['brier']):>9}{fmt(m['logloss']):>10}{fmt(m['ece']):>8}  {extra}")
    if report.get("by_season"):
        print(f"\n{'season':<12}{'model':<10}{'acc':>7}{'brier':>9}{'n':>6}")
        for s, entry in report["by_season"].items():
            for name in ("LLM", "Market"):
                if name in entry:
                    m = entry[name]
                    print(f"{s:<12}{name:<10}{fmt(m['acc'], 3):>7}{fmt(m['brier']):>9}{m['n']:>6}")
    print(f"\nreport -> {args.report}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def curve(p, y, n_bins=10):
            p = np.asarray(p); y = np.asarray(y)
            conf = np.where(p >= 0.5, p, 1 - p)
            correct = ((p >= 0.5).astype(float) == y)
            edges = np.linspace(0, 1, n_bins + 1)
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (conf >= lo) & (conf <= hi)
                if m.sum() >= 2:
                    xs.append((lo + hi) / 2); ys.append(correct[m].mean())
            return xs, ys

        fig, ax = plt.subplots(figsize=(6, 5))
        if llm_p is not None and len(llm_p) > 0:
            xs, ys = curve(llm_p, llm_y)
            ax.plot(xs, ys, "^-", color="tab:red", label="LLM (ours)")
        xs, ys = curve(mkt_p, ym)
        ax.plot(xs, ys, "o-", label="Market close")
        xs, ys = curve(lr_p, ym)
        ax.plot(xs, ys, "s-", label="LogReg")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
        ax.set_title("Calibration (test, win prob)")
        ax.legend(); ax.grid(alpha=0.3)
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"plot   -> {args.plot}")


if __name__ == "__main__":
    main()
