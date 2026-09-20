"""共用工具:賽事特徵格式化、prompt/回應建構、輸出解析、特徵矩陣、評估指標。

所有模組(資料產生、SFT/DPO、評估、推論)都使用這一份 prompt 與解析格式,
確保「訓練格式 = 評估格式 = 推論格式」,避免格式飄移。
"""
from __future__ import annotations

import math
import re
from typing import Dict, List

import numpy as np
import pandas as pd

P_HOME = "主隊"
P_AWAY = "客隊"

# ---------------------------------------------------------------- prompt
PROMPT_TEMPLATE = """以下是待預測的賽事資料:

{features}

請先簡短推理(2~4 條,必須引用上面的具體數字),再輸出:
最終預測: 主隊 或 客隊
機率: 主隊 0.00, 客隊 0.00
讓分覆蓋: 主隊 0.00, 客隊 0.00

(機率 = 該隊贏得比賽的機率;讓分覆蓋 = 該隊蓋過「收盤讓分盤」的機率。兩列各隊數字之和都必須等於 1)"""


def _num(x, nd: int = 1, default: str = "無") -> str:
    try:
        v = float(x)
        if math.isnan(v):
            return default
        return f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return default


def _signed(x, nd: int = 1, default: str = "無") -> str:
    try:
        v = float(x)
        if math.isnan(v):
            return default
        return f"{v:+.{nd}f}"
    except (TypeError, ValueError):
        return default


def format_game_features(g: Dict) -> str:
    """把一筆賽事特徵(dict,欄位見 data/SCHEMA.md)轉成中文描述。

    讓分慣例:數值為「主隊讓分」。-5.5 表示主隊讓 5.5 分;
    正值表示主隊受讓(客隊讓分)。
    """
    lines = [
        f"賽季: {g.get('season', 'N/A')}",
        f"主隊: {g.get('home', 'N/A')}(近10場 {_num(g.get('home_form_w'), 0)}勝{_num(g.get('home_form_l'), 0)}負, 近10場平均得分 {_num(g.get('home_avg_pts'))})",
        f"客隊: {g.get('away', 'N/A')}(近10場 {_num(g.get('away_form_w'), 0)}勝{_num(g.get('away_form_l'), 0)}負, 近10場平均得分 {_num(g.get('away_avg_pts'))})",
        f"休息天數: 主 {_num(g.get('home_rest'), 0)} / 客 {_num(g.get('away_rest'), 0)}",
        f"本季主/客場紀錄: 主隊主場 {_num(g.get('home_record_w'), 0)}勝{_num(g.get('home_record_l'), 0)}負, 客隊客場 {_num(g.get('away_record_w'), 0)}勝{_num(g.get('away_record_l'), 0)}負",
        f"近5次對決: 主隊 {_num(g.get('h2h_home_w'), 0)} - {_num(g.get('h2h_away_w'), 0)} 客隊",
        f"開盤: 讓分(主) {_signed(g.get('open_spread'))}, 總分 {_num(g.get('open_total'), 0)}",
        f"收盤: 讓分(主) {_signed(g.get('close_spread'))}, 總分 {_num(g.get('close_total'), 0)}",
        f"開盤賠率(主/客): {_num(g.get('open_ml_home'), 2)} / {_num(g.get('open_ml_away'), 2)}",
        f"收盤賠率(主/客): {_num(g.get('close_ml_home'), 2)} / {_num(g.get('close_ml_away'), 2)}",
    ]
    return "\n".join(lines)


def normalize_game(g: Dict) -> Dict:
    """補上 home_record_w/l、away_record_w/l(原始欄位是 'W-L' 字串)。"""
    g = dict(g)
    for k in ("home_record", "away_record"):
        if k in g and f"{k}_w" not in g:
            try:
                w, l = str(g[k]).split("-")
                g[k + "_w"], g[k + "_l"] = float(w), float(l)
            except ValueError:
                pass
    return g


def build_prompt(g: Dict) -> str:
    return PROMPT_TEMPLATE.format(features=format_game_features(normalize_game(g)))


def build_messages(g: Dict) -> List[Dict]:
    return [{"role": "user", "content": build_prompt(g)}]


# ---------------------------------------------------------------- response
def build_response(p_home: float, cover_home: float, reasons: List[str]) -> str:
    """以固定格式產生目標回應(SFT 的 assistant 文字)。"""
    p_home = round(min(0.98, max(0.02, float(p_home))), 2)
    cover_home = round(min(0.98, max(0.02, float(cover_home))), 2)
    p_away = round(1.0 - p_home, 2)
    cover_away = round(1.0 - cover_home, 2)
    pred = P_HOME if p_home >= 0.5 else P_AWAY
    body = "\n".join(f"- {r}" for r in reasons[:4]) if reasons else "- 資料有限,依盤口與狀態綜合判斷"
    return (
        f"<推理>\n{body}\n</推理>\n"
        f"最終預測: {pred}\n"
        f"機率: 主隊 {p_home:.2f}, 客隊 {p_away:.2f}\n"
        f"讓分覆蓋: 主隊 {cover_home:.2f}, 客隊 {cover_away:.2f}"
    )


_NUM = r"(1(?:\.0+)?|0(?:\.\d+)?)"
RE_PRED = re.compile(r"最終預測[:：]\s*(主隊|客隊)")
RE_WINPROB = re.compile(r"機率[:：]\s*主隊\s*" + _NUM + r"\s*[,，]\s*客隊\s*" + _NUM)
RE_COVER = re.compile(r"讓分覆蓋[:：]\s*主隊\s*" + _NUM + r"\s*[,，]\s*客隊\s*" + _NUM)


def parse_response(text: str) -> Dict:
    """解析模型輸出。format_ok=True 表示至少解析到完整的一行勝率。"""
    out = {
        "pred": None, "p_home": None, "p_away": None,
        "cover_home": None, "cover_away": None, "format_ok": False,
    }
    if not text:
        return out
    m = RE_PRED.search(text)
    if m:
        out["pred"] = m.group(1)
    m = RE_WINPROB.search(text)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if a + b > 0:
            out["p_home"], out["p_away"] = a / (a + b), b / (a + b)
            out["format_ok"] = True
    m = RE_COVER.search(text)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if a + b > 0:
            out["cover_home"], out["cover_away"] = a / (a + b), b / (a + b)
    return out


# ---------------------------------------------------------------- 特徵矩陣
FEATURE_COLS = [
    "home_form_w", "home_form_l", "away_form_w", "away_form_l",
    "home_avg_pts", "away_avg_pts", "home_rest", "away_rest",
    "home_record_w", "home_record_l", "away_record_w", "away_record_l",
    "h2h_home_w", "h2h_away_w",
    "open_spread", "close_spread",
    "open_ml_home", "open_ml_away", "close_ml_home", "close_ml_away",
]


def feature_matrix(df: pd.DataFrame):
    """DataFrame(欄位見 data/SCHEMA.md)→ (X ndarray, feature names)。"""
    d = df.copy()
    for src in ("home_record", "away_record"):
        if src in d.columns and f"{src}_w" not in d.columns:
            parts = d[src].astype(str).str.split("-")
            d[f"{src}_w"] = pd.to_numeric(parts.str[0], errors="coerce")
            d[f"{src}_l"] = pd.to_numeric(parts.str[1], errors="coerce")
    names = [c for c in FEATURE_COLS if c in d.columns]
    X = d[names].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(float)
    return X, names


# ---------------------------------------------------------------- 評估指標
def implied_prob(odds_home, odds_away):
    """去除抽水後的市場隱含機率 → (主隊, 客隊)。"""
    qh = 1.0 / float(odds_home)
    qa = 1.0 / float(odds_away)
    s = qh + qa
    return qh / s, qa / s


def _acc_brier_logloss(p, y, eps: float = 1e-4) -> Dict:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(p)
    if n == 0:
        return {"n": 0, "acc": None, "brier": None, "logloss": None}
    acc = float((((p >= 0.5).astype(float) == 1) == (y == 1)).mean())
    brier = float(((p - y) ** 2).mean())
    pc = np.clip(p, eps, 1.0 - eps)
    logloss = float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean())
    return {"n": int(n), "acc": acc, "brier": brier, "logloss": logloss}


def ece(p, y, n_bins: int = 10):
    """Expected Calibration Error(以 confidence = max(p, 1-p) 分箱)。"""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return None
    conf = np.where(p >= 0.5, p, 1.0 - p)
    correct = ((p >= 0.5).astype(float) == y)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf <= hi) if lo > 0 else (conf == 0) | (conf <= hi)
        if m.sum() == 0:
            continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)
