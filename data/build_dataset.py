"""把 matches.csv 變成 SFT / DPO 訓練資料。

做法:
1. 依時間切分 train/val/test(預設 70/15/15)——絕不讓模型偷看未來。
2. teacher 模型:HistGradientBoostingClassifier(只用 train 擬合),負責產出
   「目標機率」(勝負 + 讓分覆蓋各一個 teacher)。
3. 每筆樣本 = prompt(中文特徵描述) + response(推理 + 固定格式機率)。
   推理文字引用真實特徵數字,讓 LLM 學會「看盤口與狀態」而不是背答案。
4. DPO pairs:chosen = teacher 回應;rejected = 把機率往反方向扭曲的回應。

輸出: <out>/{train,val,test}.jsonl 與 <out>/dpo_pairs.jsonl

範例:
  python data/build_dataset.py --matches data/demo/matches.csv --out data/out
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from common import build_prompt, build_response, feature_matrix, implied_prob


def make_reasons(r: pd.Series, p_home: float, mkt_home: float) -> list[str]:
    """依真實特徵數字產生 2~4 條推理(模板化但引用實際值)。"""
    out: list[str] = []
    try:
        hw, hl = int(r["home_form_w"]), int(r["home_form_l"])
        aw, al = int(r["away_form_w"]), int(r["away_form_l"])
        out.append(f"近10場狀態: 主隊 {hw}-{hl},客隊 {aw}-{al}")
        if abs(aw - hw) >= 2:
            out.append(f"狀態比較: {'客隊' if aw > hw else '主隊'}近10場明顯佔優")
    except (KeyError, TypeError, ValueError):
        pass
    try:
        hr, ar = int(r["home_rest"]), int(r["away_rest"])
        if hr > ar:
            out.append(f"輪休: 主隊休息 {hr} 天 > 客隊 {ar} 天")
        elif ar > hr:
            out.append(f"輪休: 客隊休息 {ar} 天 > 主隊 {hr} 天")
    except (KeyError, TypeError, ValueError):
        pass
    try:
        if int(r["h2h_home_w"]) + int(r["h2h_away_w"]) > 0:
            out.append(f"近5次對決: 主隊 {int(r['h2h_home_w'])}-{int(r['h2h_away_w'])} 客隊")
    except (KeyError, TypeError, ValueError):
        pass
    try:
        if abs(float(r["close_spread"]) - float(r["open_spread"])) >= 1.0:
            dirn = "主隊" if float(r["close_spread"]) < float(r["open_spread"]) else "客隊"
            out.append(
                f"盤口移動: 讓分從 {float(r['open_spread']):+.1f} 變為 {float(r['close_spread']):+.1f},"
                f"資金流向{dirn}"
            )
    except (KeyError, TypeError, ValueError):
        pass
    if abs(p_home - mkt_home) >= 0.02:
        side = "主隊" if p_home > mkt_home else "客隊"
        out.append(f"市場隱含主隊勝率 {mkt_home:.2f},模型評估 {p_home:.2f},差距在{side}一方")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matches", default="data/demo/matches.csv")
    ap.add_argument("--out", default="data/out")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-pairs", type=int, default=4000, help="DPO pairs 上限(0=全部 train)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.matches)
    df = df.sort_values(["date", "match_id"]).reset_index(drop=True)
    n = len(df)
    i_tr = int(n * args.train_frac)
    i_va = int(n * (args.train_frac + args.val_frac))
    train, val, test = df.iloc[:i_tr], df.iloc[i_tr:i_va], df.iloc[i_va:]
    print(f"split by date: train={len(train)} (..{train['date'].iloc[-1]}), "
          f"val={len(val)}, test={len(test)} (from {test['date'].iloc[0]})")

    # ---- teacher(只吃 train 的特徵)----
    X_all, names = feature_matrix(df)
    print(f"teacher features ({len(names)}): {names}")
    y_win = train["home_win"].astype(int).values
    y_cov = (train["margin"] > train["close_spread"]).astype(int).values
    rng = np.random.default_rng(args.seed)
    # 規則化較強的設定:小資料上避免過擬合(過擬合會讓 test AUC 低於市場 baseline)
    common_kw = dict(max_iter=150, learning_rate=0.03, max_depth=3, l2_regularization=10.0)
    m_win = HistGradientBoostingClassifier(**common_kw, random_state=args.seed).fit(X_all[:i_tr], y_win)
    m_cov = HistGradientBoostingClassifier(**common_kw, random_state=args.seed).fit(X_all[:i_tr], y_cov)
    p_home_all = m_win.predict_proba(X_all)[:, 1]
    cov_home_all = m_cov.predict_proba(X_all)[:, 1]

    def _auc(y, p):
        try:
            return float(roc_auc_score(y, p))
        except ValueError:
            return None

    print(f"teacher AUC (train): win={_auc(y_win, p_home_all[:i_tr]):.4f}  "
          f"cover={_auc(y_cov, cov_home_all[:i_tr]):.4f}")
    print(f"teacher AUC (test) : win={_auc(test['home_win'].astype(int).values, p_home_all[i_va:]):.4f}  "
          f"cover={_auc((test['margin'] > test['close_spread']).astype(int).values, cov_home_all[i_va:]):.4f}")

    # ---- 寫 SFT jsonl ----
    os.makedirs(args.out, exist_ok=True)
    split_of = {}
    for part, d, p in (("train", train, p_home_all[:i_tr]),
                       ("val", val, p_home_all[i_tr:i_va]),
                       ("test", test, p_home_all[i_va:])):
        cov_part = {"train": cov_home_all[:i_tr], "val": cov_home_all[i_tr:i_va],
                    "test": cov_home_all[i_va:]}[part]
        path = os.path.join(args.out, f"{part}.jsonl")
        rows = []
        for pos, (_, r) in enumerate(d.iterrows()):
            ph = float(p[pos])
            ch = float(cov_part[pos])
            mkt_h, _mkt_a = implied_prob(r["close_ml_home"], r["close_ml_away"])
            game = r.to_dict()
            rows.append({
                "match_id": r["match_id"],
                "date": r["date"],
                "prompt": [{"role": "user", "content": build_prompt(game)}],
                "response": build_response(ph, ch, make_reasons(r, ph, mkt_h)),
                "outcome_home": int(r["home_win"]),
                "cover_home": int(r["margin"] > r["close_spread"]),
                "teacher_p_home": round(ph, 4),
                "teacher_cover_home": round(ch, 4),
                "market_p_home_close": round(float(mkt_h), 4),
                "close_spread": float(r["close_spread"]),
            })
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        split_of[part] = rows
        print(f"wrote {path} ({len(rows)} rows)")

    # ---- DPO pairs(train 部分)----
    train_by_id = {r["match_id"]: r for r in train.to_dict("records")}
    pairs = []
    for row in split_of["train"]:
        p = row["teacher_p_home"]
        # 把機率往反方向扭曲:強的 teacher 判斷會被翻成弱/錯誤的判斷
        rp = 0.5 - (p - 0.5) * 0.8 if p >= 0.5 else 0.5 + (0.5 - p) * 0.8
        rp = float(np.clip(rp, 0.05, 0.95))
        game = train_by_id.get(row["match_id"])
        if game is None:
            continue
        mkt_h, _ = implied_prob(game["close_ml_home"], game["close_ml_away"])
        pairs.append({
            "match_id": row["match_id"],
            "prompt": row["prompt"][0]["content"],
            "chosen": row["response"],
            "rejected": build_response(rp, 1.0 - row["teacher_cover_home"],
                                       make_reasons(pd.Series(game), p, mkt_h)),
            "outcome_home": row["outcome_home"],
        })
    if args.max_pairs and len(pairs) > args.max_pairs:
        idx = np.sort(rng.choice(len(pairs), args.max_pairs, replace=False))
        pairs = [pairs[i] for i in idx]
    path = os.path.join(args.out, "dpo_pairs.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for p_ in pairs:
            f.write(json.dumps(p_, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(pairs)} pairs)")

    # 展示一筆
    sample = split_of["test"][0] if split_of["test"] else split_of["train"][0]
    print("\n===== sample (test) =====")
    print("PROMPT:\n" + sample["prompt"][0]["content"])
    print("\nRESPONSE:\n" + sample["response"])


if __name__ == "__main__":
    main()
