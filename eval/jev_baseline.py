"""Jev(TypeSafe AI「System One」模型)接軌:把它當成一個 baseline 來對照我們的 0.9B。

Jev 回傳「typed 決策 + 校準機率」(Choice/Score/Noul),不產生文字——
正好是本倉庫 typed 輸出格式的外包版。這裡用它的 Choice 原始語義:

  soccer     → 1 題 Choice{主隊,和局,客隊}
  basketball → 1 題 Choice{主隊,客隊} + 1 題 Noul「主隊蓋收盤讓分盤?」

API(2026-09-15 發布):
  POST https://api.typesafe.ai/v1/systemone
  Authorization: Bearer $TYPESAFE_API_KEY
  body: {"state": ..., "model": "jev-latest", "questions": {...}}

state 只放「該題需要的欄位」(Jev 的 jaggedness 警告:state 塞雜訊會讓準確率掉)。
沒有 API key 時所有函式 fail-fast;單測以 monkeypatch 假回應驗證。

用法(要 key):
  export TYPESAFE_API_KEY=sk-...
  python eval/jev_baseline.py --test-jsonl data/out/test.jsonl \
      --matches-csv data/demo/soccer.csv --sport soccer --report output/jev.json
  # 或在 eval/evaluate.py 加 --jev,自動把 Jev 當成一個 baseline 放進 report.json
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from common import (attach_league_rates, confidence_from_probs,
                    format_game_features, implied_prob, implied_prob_3way)

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = "jev-latest"


class JevError(RuntimeError):
    pass


def choice(name: str, instructions: str, criteria: Dict[str, str]) -> dict:
    """Jev Choice 原始語義:固定選項集,一次回 choice + probabilities + confidence。"""
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def noul(name: str, instructions: str) -> dict:
    """Jev Noul:是/否的機率。"""
    return {"type": "noul", "instructions": instructions}


def jev_call(state, questions: Dict[str, dict], api_key: str = None,
             model: str = JEV_MODEL, endpoint: str = JEV_ENDPOINT,
             timeout: float = 30.0) -> Dict[str, dict]:
    """一次並行問所有題(Jev 的賣點:多題近乎不增加延遲)。回傳 answers dict。"""
    api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise JevError("缺 TYPESAFE_API_KEY(env)或 api_key 參數")
    body = json.dumps({"state": state, "model": model, "questions": questions},
                      ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=body, method="POST",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise JevError(f"Jev API HTTP {e.code}: {e.read()[:200]!r}") from e
    except urllib.error.URLError as e:
        raise JevError(f"Jev API 連線失敗: {e.reason}") from e
    return data.get("answers", data)


def _choice_answer(ans: dict) -> Tuple[List[float], float]:
    """Choice 回應 → (probabilities 依 options 順序, confidence)。"""
    opts = list(ans.get("options", {}).keys()) if isinstance(ans.get("options"), dict) else None
    probs = ans.get("probabilities")
    if opts is None:
        # 某些 SDK 版本把 probabilities 放成 {option: p}
        probs_map = ans.get("probabilities")
        if isinstance(probs_map, dict):
            opts = list(probs_map.keys())
            probs = [probs_map[o] for o in opts]
        else:
            raise JevError(f"無法解析 Choice 回應: {ans!r}")
    probs = [float(p) for p in probs]
    conf = ans.get("confidence")
    conf = float(conf) if conf is not None else confidence_from_probs(probs)
    return probs, conf


def jev_soccer_state(game: dict) -> dict:
    """state = 只放 1X2 決策需要的欄位(Jev:retrieve/filter first,別塞雜訊)。"""
    return {
        "sport": "soccer",
        "season": game.get("season"),
        "features": format_game_features(game, "soccer"),
        "open_ml": [game.get("open_ml_home"), game.get("open_ml_draw"), game.get("open_ml_away")],
        "close_ml": [game.get("close_ml_home"), game.get("close_ml_draw"), game.get("close_ml_away")],
        "market_close_implied": list(implied_prob_3way(
            game["close_ml_home"], game["close_ml_draw"], game["close_ml_away"])),
    }


def jev_basketball_state(game: dict) -> dict:
    return {
        "sport": "basketball",
        "season": game.get("season"),
        "features": format_game_features(game, "basketball"),
        "close_spread_home": game.get("close_spread"),
        "market_close_implied_home": implied_prob(
            game["close_ml_home"], game["close_ml_away"])[0],
    }


def predict_soccer_row(game: dict, api_key: str = None) -> Tuple[List[float], float]:
    """單場 1X2 → (probabilities [主,和,客], confidence)。"""
    ans = jev_call(jev_soccer_state(game), {
        "outcome": choice("outcome", "這場足球比賽的 90 分鐘結果是什麼?",
                          {"home": "主隊贏(1)", "draw": "和局(X)", "away": "客隊贏(2)"}),
    }, api_key=api_key)
    probs, conf = _choice_answer(ans["outcome"])
    s = sum(probs)
    return [p / s for p in probs], conf


def predict_basketball_row(game: dict, api_key: str = None) -> Tuple[float, float, float]:
    """單場 → (p_home_win, p_home_cover_spread, confidence)。"""
    ans = jev_call(jev_basketball_state(game), {
        "outcome": choice("outcome", "哪一隊會贏這场比赛?",
                          {"home": "主隊贏", "away": "客隊贏"}),
        "cover": noul("cover", "主隊在收盤讓分盤上覆蓋(cover)嗎?"),
    }, api_key=api_key)
    probs, conf = _choice_answer(ans["outcome"])
    p_home = probs[0] / (probs[0] + probs[1])
    p_cover = float(ans["cover"].get("noul", 0.5))
    return p_home, p_cover, conf


def run_jev_baseline(rows: List[dict], matches_csv: str, sport: str,
                     api_key: str = None, limit: int = None) -> Dict:
    """對 test.jsonl 的每筆跑 Jev,回傳 {metrics, n, parse_error_rate, probs}。

    rows 需含 match_id + outcome(soccer)或 outcome_home(basketball)。
    """
    from common import (_acc_brier_logloss, _acc_brier_logloss_mc, ece, ece_mc)
    df = attach_league_rates(pd.read_csv(matches_csv), sport)
    src = df.set_index("match_id")
    rows = rows[:limit] if limit else rows
    P, Y, n_err = [], [], 0
    for row in rows:
        try:
            g = src.loc[row["match_id"]].to_dict()
        except KeyError:
            n_err += 1
            continue
        try:
            if sport == "soccer":
                probs, _conf = predict_soccer_row(g, api_key)
                y = int(row["outcome"])
            else:
                p_home, _pc, _conf = predict_basketball_row(g, api_key)
                probs, y = [p_home, 1.0 - p_home], int(row["outcome_home"])
        except JevError as e:
            print(f"[jev] {row.get('match_id')} fail: {e}")
            n_err += 1
            continue
        P.append(probs)
        Y.append(y)
    if sport == "soccer":
        Pm = np.array(P)
        Ym = np.zeros((len(P), 3))
        Ym[np.arange(len(P)), [int(y) for y in Y]] = 1.0
        m = _acc_brier_logloss_mc(Pm, Ym)
        m["ece"] = ece_mc(Pm, Ym)
    else:
        p = np.array([x[0] for x in P])
        m = _acc_brier_logloss(p, np.array(Y))
        m["ece"] = ece(p, np.array(Y))
    m["n"] = len(P)
    m["api_error_rate"] = n_err / max(1, len(rows))
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-jsonl", required=True)
    ap.add_argument("--matches-csv", required=True)
    ap.add_argument("--sport", choices=("basketball", "soccer"), default="soccer")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--report", default="output/jev_baseline.json")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.test_jsonl, encoding="utf-8") if l.strip()]
    m = run_jev_baseline(rows, args.matches_csv, args.sport, limit=args.limit)
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(json.dumps(m, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
