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
P_DRAW = "和局"
SPORTS = ("basketball", "soccer")

# ---------------------------------------------------------------- prompt
PROMPT_TEMPLATE = """以下是待預測的賽事資料:

{features}

請先簡短推理(2~4 條,必須引用上面的具體數字),再輸出:
最終預測: 主隊 或 客隊
機率: 主隊 0.00, 客隊 0.00
讓分覆蓋: 主隊 0.00, 客隊 0.00

(機率 = 該隊贏得比賽的機率;讓分覆蓋 = 該隊蓋過「收盤讓分盤」的機率。兩列各隊數字之和都必須等於 1)"""

# 足球 1X2(主勝/和/客勝);v1 不輸出让分覆蓋(盤口資訊用 1X2 賠率欄)
def _prompt_template(sport: str) -> str:
    if sport == "soccer":
        return (
            "以下是待預測的足球賽事資料:\n\n"
            "{features}\n\n"
            "請先簡短推理(2~4 條,必須引用上面的具體數字),再輸出:\n"
            "最終預測: 主隊 / 和局 / 客隊\n"
            "機率: 主隊 0.00, 和局 0.00, 客隊 0.00\n\n"
            "(機率 = 該結果發生的機率,三個數字之和必須等於 1)"
        )
    return PROMPT_TEMPLATE


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


def format_game_features(g: Dict, sport: str = "basketball") -> str:
    """把一筆賽事特徵(dict,欄位見 data/SCHEMA.md)轉成中文描述。

    籃球讓分慣例:數值為「主隊讓分」。-5.5 表示主隊讓 5.5 分;
    正值表示主隊受讓(客隊讓分)。足球 v1 用 1X2 賠率欄(含和局)。
    """
    if sport == "soccer":
        def _wdl(w, d, l):
            return f"{_num(w, 0)}勝{_num(d, 0)}平{_num(l, 0)}負"
        lines = [
            f"賽季: {g.get('season', 'N/A')}",
            f"主隊: {g.get('home', 'N/A')}(近10場 {_wdl(g.get('home_form_w'), g.get('home_form_d'), g.get('home_form_l'))}, 近10場平均每場進球 {_num(g.get('home_avg_pts'))})",
            f"客隊: {g.get('away', 'N/A')}(近10場 {_wdl(g.get('away_form_w'), g.get('away_form_d'), g.get('away_form_l'))}, 近10場平均每場進球 {_num(g.get('away_avg_pts'))})",
            f"休息天數: 主 {_num(g.get('home_rest'), 0)} / 客 {_num(g.get('away_rest'), 0)}",
            f"本季主/客場紀錄: 主隊主場 {_wdl(g.get('home_record_w'), g.get('home_record_d'), g.get('home_record_l'))}, 客隊客場 {_wdl(g.get('away_record_w'), g.get('away_record_d'), g.get('away_record_l'))}",
            f"近5次對決: 主隊 {_num(g.get('h2h_home_w'), 0)} - {_num(g.get('h2h_away_w'), 0)} 客隊",
            f"開盤 1X2 賠率(主/和/客): {_num(g.get('open_ml_home'), 2)} / {_num(g.get('open_ml_draw'), 2)} / {_num(g.get('open_ml_away'), 2)}",
            f"收盤 1X2 賠率(主/和/客): {_num(g.get('close_ml_home'), 2)} / {_num(g.get('close_ml_draw'), 2)} / {_num(g.get('close_ml_away'), 2)}",
        ]
        return "\n".join(lines)
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


def normalize_game(g: Dict, sport: str = "basketball") -> Dict:
    """補上 home_record_w/l(/d)、away_record_w/l(/d)。

    原始欄位:籃球 'W-L';足球 'W-D-L'。
    """
    g = dict(g)
    for k in ("home_record", "away_record"):
        if k in g and f"{k}_w" not in g:
            try:
                parts = [float(x) for x in str(g[k]).split("-")]
                if len(parts) == 2:
                    g[k + "_w"], g[k + "_l"] = parts
                    g[k + "_d"] = 0.0
                elif len(parts) == 3:
                    g[k + "_w"], g[k + "_d"], g[k + "_l"] = parts
            except ValueError:
                pass
    return g


def build_prompt(g: Dict, sport: str = "basketball") -> str:
    return _prompt_template(sport).format(features=format_game_features(normalize_game(g, sport), sport))


def build_messages(g: Dict, sport: str = "basketball") -> List[Dict]:
    return [{"role": "user", "content": build_prompt(g, sport)}]


# ---------------------------------------------------------------- response
def build_response(p_home: float, cover_home: float, reasons: List[str],
                   sport: str = "basketball", p_draw: float = None) -> str:
    """以固定格式產生目標回應(SFT 的 assistant 文字)。

    soccer: p_draw 必給(三結果 1X2,不輸出让分覆蓋)。
    """
    body = "\n".join(f"- {r}" for r in reasons[:4]) if reasons else "- 資料有限,依盤口與狀態綜合判斷"
    if sport == "soccer":
        ph = min(0.95, max(0.02, float(p_home)))
        pd_ = min(0.90, max(0.02, float(p_draw) if p_draw is not None else 0.0))
        pa = max(0.02, 1.0 - ph - pd_)
        s = ph + pd_ + pa
        ph, pd_, pa = round(ph / s, 2), round(pd_ / s, 2), round(pa / s, 2)
        pred = P_HOME if ph >= max(pd_, pa) else (P_DRAW if pd_ >= pa else P_AWAY)
        return (
            f"<推理>\n{body}\n</推理>\n"
            f"最終預測: {pred}\n"
            f"機率: 主隊 {ph:.2f}, 和局 {pd_:.2f}, 客隊 {pa:.2f}"
        )
    p_home = round(min(0.98, max(0.02, float(p_home))), 2)
    cover_home = round(min(0.98, max(0.02, float(cover_home))), 2)
    p_away = round(1.0 - p_home, 2)
    cover_away = round(1.0 - cover_home, 2)
    pred = P_HOME if p_home >= 0.5 else P_AWAY
    return (
        f"<推理>\n{body}\n</推理>\n"
        f"最終預測: {pred}\n"
        f"機率: 主隊 {p_home:.2f}, 客隊 {p_away:.2f}\n"
        f"讓分覆蓋: 主隊 {cover_home:.2f}, 客隊 {cover_away:.2f}"
    )


_NUM = r"(1(?:\.0+)?|0(?:\.\d+)?)"
RE_PRED = re.compile(r"最終預測[:：]\s*(主隊|客隊)")
RE_PRED_S = re.compile(r"最終預測[:：]\s*(主隊|和局|客隊)")
RE_WINPROB = re.compile(r"機率[:：]\s*主隊\s*" + _NUM + r"\s*[,，]\s*客隊\s*" + _NUM)
RE_PROB3 = re.compile(
    r"機率[:：]\s*主隊\s*" + _NUM + r"\s*[,，]\s*和局\s*" + _NUM + r"\s*[,，]\s*客隊\s*" + _NUM)
RE_COVER = re.compile(r"讓分覆蓋[:：]\s*主隊\s*" + _NUM + r"\s*[,，]\s*客隊\s*" + _NUM)

# 三結果 one-hot 順序:0=主勝 1=和 2=客勝
OUTCOME_LABELS = (P_HOME, P_DRAW, P_AWAY)


def outcome_from_margin(margin: float) -> int:
    """margin = home_score - away_score → 0=主勝 1=和 2=客勝。"""
    m = float(margin)
    return 0 if m > 0 else (1 if m == 0 else 2)


def parse_response(text: str, sport: str = "basketball") -> Dict:
    """解析模型輸出。format_ok=True 表示至少解析到完整的一行勝率。

    basketball: 二結果 + 覆蓋;soccer: 三結果 1X2(p_draw)。
    """
    out = {
        "pred": None, "p_home": None, "p_away": None, "p_draw": None,
        "cover_home": None, "cover_away": None, "format_ok": False,
    }
    if not text:
        return out
    if sport == "soccer":
        m = RE_PRED_S.search(text)
        if m:
            out["pred"] = m.group(1)
        m = RE_PROB3.search(text)
        if m:
            a, b, c = float(m.group(1)), float(m.group(2)), float(m.group(3))
            s = a + b + c
            if s > 0:
                out["p_home"], out["p_draw"], out["p_away"] = a / s, b / s, c / s
                out["format_ok"] = True
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
    # 足球 1X2 額外的「和」狀態(籃球沒有這些欄位 → 自動被跳過)
    "home_form_d", "away_form_d",
    "home_avg_pts", "away_avg_pts", "home_rest", "away_rest",
    "home_record_w", "home_record_l", "away_record_w", "away_record_l",
    "home_record_d", "away_record_d",
    "h2h_home_w", "h2h_away_w",
    "open_spread", "close_spread",
    "open_ml_home", "open_ml_away", "close_ml_home", "close_ml_away",
    "open_ml_draw", "close_ml_draw",
]


def feature_matrix(df: pd.DataFrame):
    """DataFrame(欄位見 data/SCHEMA.md)→ (X ndarray, feature names)。"""
    d = df.copy()
    for src in ("home_record", "away_record"):
        if src in d.columns and f"{src}_w" not in d.columns:
            parts = d[src].astype(str).str.split("-")
            d[f"{src}_w"] = pd.to_numeric(parts.str[0], errors="coerce")
            d[f"{src}_d"] = pd.to_numeric(parts.str[1], errors="coerce") if (
                parts.str.len() >= 3).any() else 0.0
            d[f"{src}_l"] = pd.to_numeric(parts.str[-1], errors="coerce")
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


def implied_prob_3way(odds_home, odds_draw, odds_away):
    """1X2 三結果去抽水(比例法)→ (主, 和, 客),和為 1。"""
    qh = 1.0 / float(odds_home)
    qd = 1.0 / float(odds_draw)
    qa = 1.0 / float(odds_away)
    s = qh + qd + qa
    return qh / s, qd / s, qa / s


def margin_from_score(home_score, away_score) -> float:
    return float(home_score) - float(away_score)


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
    """Expected Calibration Error(以 confidence = max(p, 1-p) 分箱,半開區間)。"""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(p) == 0:
        return None
    conf = np.where(p >= 0.5, p, 1.0 - p)
    correct = ((p >= 0.5).astype(float) == y)
    idx = np.clip(np.digitize(conf, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]),
                  0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


# -------------------------------------------------- 多類(足球 1X2)指標
def _acc_brier_logloss_mc(p_mat, y_mat, eps: float = 1e-4) -> Dict:
    """多類版:acc(argmax)/ Brier(mean Σ(p_i-y_i)²)/ logloss。

    p_mat: [N, K];y_mat: [N, K] one-hot。
    """
    p_mat = np.atleast_2d(np.asarray(p_mat, dtype=float))
    y_mat = np.atleast_2d(np.asarray(y_mat, dtype=float))
    if p_mat.size == 0:  # 空輸入(atleast_2d 會變成 (1,0))
        return {"n": 0, "acc": None, "brier": None, "logloss": None}
    n = len(p_mat)
    acc = float((p_mat.argmax(1) == y_mat.argmax(1)).mean())
    brier = float(((p_mat - y_mat) ** 2).sum(1).mean())
    pc = np.clip(p_mat, eps, 1.0 - eps)
    logloss = float(-(y_mat * np.log(pc)).sum(1).mean())
    return {"n": int(n), "acc": acc, "brier": brier, "logloss": logloss}


def ece_mc(p_mat, y_mat, n_bins: int = 10):
    """多類 ECE:confidence = max p,correct = argmax p == argmax y。"""
    p_mat = np.atleast_2d(np.asarray(p_mat, dtype=float))
    y_mat = np.atleast_2d(np.asarray(y_mat, dtype=float))
    if p_mat.size == 0:
        return None
    conf = p_mat.max(1)
    correct = (p_mat.argmax(1) == y_mat.argmax(1)).astype(float)
    # 半開區間分箱(最右 bin 含 1.0),避免邊界值被雙重計入
    idx = np.clip(np.digitize(conf, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]),
                  0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)
