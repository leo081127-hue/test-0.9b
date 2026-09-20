"""Jev(TypeSafe System One)client:請求形狀、回應解析、baseline 流程(mock,無網路)。"""
import io
import json
import urllib.error

import numpy as np
import pandas as pd
import pytest

import eval.jev_baseline as jb
from common import attach_league_rates


class FakeResp:
    def __init__(self, payload: dict):
        self._data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def capture(monkeypatch):
    """monkeypatch urlopen → 回預設的 soccer 回應;並記錄每次請求。"""
    calls = []

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        calls.append({"url": req.full_url, "auth": req.get_header("Authorization"),
                      "body": body})
        probs = {"home": 0.40, "draw": 0.25, "away": 0.35}
        if "outcome" in body["questions"] and body["questions"]["outcome"].get("criteria", {}).get("away") == "客隊贏":
            probs = {"home": 0.62, "away": 0.38}
            payload = {"answers": {
                "outcome": {"choice": "home", "probabilities": probs, "confidence": 0.24},
                "cover": {"noul": 0.58},
            }}
        else:
            payload = {"answers": {
                "outcome": {"choice": "home", "probabilities": probs, "confidence": 0.10},
            }}
        return FakeResp(payload)

    monkeypatch.setattr(jb.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")
    return calls


def test_jev_call_request_shape(capture):
    ans = jb.jev_call({"a": 1}, {"outcome": jb.choice("outcome", "x", {"home": "h", "draw": "d", "away": "a"})})
    req = capture[-1]
    assert req["url"] == jb.JEV_ENDPOINT
    assert req["auth"] == "Bearer sk-test"
    assert req["body"]["model"] == "jev-latest"
    assert req["body"]["state"] == {"a": 1}
    assert req["body"]["questions"]["outcome"]["type"] == "choice"
    assert set(ans.keys()) == {"outcome"}
    assert ans["outcome"]["choice"] == "home"


def test_jev_call_no_key_fails(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(jb.JevError):
        jb.jev_call({}, {"q": jb.noul("q", "is it?")})


def test_jev_call_http_error(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 401, "unauthorized", {}, io.BytesIO(b"nope"))
    monkeypatch.setattr(jb.urllib.request, "urlopen", boom)
    with pytest.raises(jb.JevError, match="401"):
        jb.jev_call({}, {"q": jb.noul("q", "x")}, api_key="sk-x")


def test_choice_answer_dict_form(monkeypatch):
    def fake(req, timeout=None):
        return FakeResp({"answers": {"outcome": {
            "choice": "draw", "probabilities": {"home": 0.3, "draw": 0.4, "away": 0.3},
            "confidence": 0.1}}})
    monkeypatch.setattr(jb.urllib.request, "urlopen", fake)
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-t")
    probs, conf = jb.predict_soccer_row({"close_ml_home": 2.0, "close_ml_draw": 3.0,
                                         "close_ml_away": 3.0, "open_ml_home": 2.1,
                                         "open_ml_draw": 3.1, "open_ml_away": 3.2}, api_key="sk-t")
    assert probs[1] == pytest.approx(0.4)
    assert conf == pytest.approx(0.1)


def test_state_contains_no_outcome_fields(capture):
    game = {"season": "2024", "home": "A", "away": "B", "home_form_w": 5,
            "open_ml_home": 2.1, "open_ml_draw": 3.3, "open_ml_away": 3.4,
            "close_ml_home": 2.0, "close_ml_draw": 3.2, "close_ml_away": 3.3,
            "margin": 1.0, "home_score": 2, "away_score": 1, "home_win": 1}
    jb.predict_soccer_row(game)
    state = capture[-1]["body"]["state"]
    # Jev 的 jaggedness:state 不得含答案
    for bad in ("margin", "home_score", "away_score", "home_win"):
        assert bad not in state
    assert "market_close_implied" in state


def _soccer_fixture(tmp_path):
    df = pd.DataFrame({
        "match_id": ["M1", "M2"], "date": ["2020-01-01", "2020-01-02"],
        "season": ["s", "s"], "home": ["A", "B"], "away": ["B", "A"],
        "home_score": [2, 0], "away_score": [1, 1], "home_win": [1, 0], "margin": [1, -1],
        "open_spread": [np.nan, np.nan], "close_spread": [np.nan, np.nan],
        "open_total": [np.nan, np.nan], "close_total": [np.nan, np.nan],
        "open_ml_home": [2.0, 2.5], "open_ml_draw": [3.2, 3.2], "open_ml_away": [3.4, 2.8],
        "close_ml_home": [1.9, 2.6], "close_ml_draw": [3.1, 3.1], "close_ml_away": [3.3, 2.7],
        "home_form_w": [5, 3], "home_form_d": [1, 1], "home_form_l": [4, 6],
        "away_form_w": [3, 4], "away_form_d": [2, 3], "away_form_l": [5, 3],
        "home_avg_pts": [1.5, 1.0], "away_avg_pts": [1.0, 1.2],
        "home_rest": [2, 2], "away_rest": [1, 1],
        "home_record": ["5-1-4", "3-1-6"], "away_record": ["3-2-5", "4-3-3"],
        "h2h_home_w": [2, 1], "h2h_away_w": [3, 4],
    })
    csv = tmp_path / "m.csv"
    df.to_csv(csv, index=False)
    rows = [{"match_id": "M1", "outcome": 0}, {"match_id": "M2", "outcome": 2}]
    return str(csv), rows


def test_run_jev_baseline_soccer(capture, tmp_path):
    csv, rows = _soccer_fixture(tmp_path)
    m = jb.run_jev_baseline(rows, csv, "soccer")
    assert m["n"] == 2 and m["api_error_rate"] == 0.0
    assert m["acc"] is not None and 0.0 <= m["acc"] <= 1.0
    assert "brier" in m and "ece" in m
    # 兩題都問到(每場一題 Choice)
    assert len(capture) == 2


def test_run_jev_baseline_api_errors_counted(monkeypatch, tmp_path):
    csv, rows = _soccer_fixture(tmp_path)
    def fail(req, timeout=None):
        raise urllib.error.URLError("down")
    monkeypatch.setattr(jb.urllib.request, "urlopen", fail)
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-x")
    m = jb.run_jev_baseline(rows, csv, "soccer")
    assert m["n"] == 0 and m["api_error_rate"] == 1.0


def test_maybe_add_jev_no_key_skips(monkeypatch):
    import eval.evaluate as ev
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    args = type("A", (), {"jev": True, "matches_csv": "x.csv", "sport": "soccer"})()
    report = {}
    ev._maybe_add_jev(report, [], args)
    assert report == {}  # 沒 key → 不 crash、不改 report


def test_maybe_add_jev_off_noop(monkeypatch):
    import eval.evaluate as ev
    args = type("A", (), {"jev": False, "matches_csv": "x.csv", "sport": "soccer"})()
    report = {}
    ev._maybe_add_jev(report, [], args)
    assert report == {}
