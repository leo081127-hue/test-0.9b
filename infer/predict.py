"""對「新賽事」做預測(推論 CLI)。

輸入二擇一:
  --features-json path.json   單筆賽事特徵(欄位見 data/SCHEMA.md;結果欄位可不填)
  --csv path --row N          直接取 matches.csv 的第 N 行(0-based)來模擬

範例:
  python infer/predict.py --model-id IFM/K2-Horizon-0.9B --trust-remote-code \
      --adapter-dir output/dpo --csv data/demo/matches.csv --row 1500
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd

from common import attach_league_rates, build_messages, parse_response_any
from train.sft import DTYPES, load_causal_lm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="IFM/K2-Horizon-0.9B")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    ap.add_argument("--adapter-dir", default=None)
    ap.add_argument("--features-json", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--row", type=int, default=None)
    ap.add_argument("--sport", choices=("basketball", "soccer", "auto"), default="auto")
    ap.add_argument("--format", choices=("prose", "typed"), default="prose",
                    help="prose=推理+固定格式行;typed=Jev 風格嚴格 JSON(需與訓練格式一致)")
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    if args.features_json:
        with open(args.features_json, encoding="utf-8") as f:
            game = json.load(f)
    elif args.csv is not None and args.row is not None:
        df = pd.read_csv(args.csv)
        # 附上「截至該行開賽前」的聯賽 base rate(與訓練時相同)
        df = attach_league_rates(df, "soccer" if "open_ml_draw" in df.columns else "basketball")
        game = df.iloc[args.row].to_dict()
    else:
        ap.error("需要 --features-json 或 (--csv + --row)")

    sport = args.sport
    if sport == "auto":
        try:
            sport = "soccer" if game.get("open_ml_draw") not in (None, "") \
                and pd.notna(game.get("open_ml_draw")) else "basketball"
        except Exception:
            sport = "basketball"
    print(f"sport: {sport}")

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

    text = tok.apply_chat_template(build_messages(game, sport, args.format),
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tok.pad_token_id,
        do_sample=args.temperature > 0,
    )
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature
    with torch.no_grad():
        gen = model.generate(**ids, **gen_kwargs)
    out = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    p = parse_response_any(out, sport)

    print("=" * 60)
    print("模型輸出:")
    print(out)
    print("=" * 60)
    print("解析結果:")
    print(f"  最終預測   : {p['pred'] or '解析失敗'}")
    print(f"  主隊勝率   : {p['p_home']:.3f}" if p["p_home"] is not None else "  主隊勝率   : n/a")
    if sport == "soccer":
        print(f"  和局機率   : {p['p_draw']:.3f}" if p["p_draw"] is not None else "  和局機率   : n/a")
    print(f"  客隊勝率   : {p['p_away']:.3f}" if p["p_away"] is not None else "  客隊勝率   : n/a")
    if sport != "soccer":
        print(f"  讓分覆蓋主 : {p['cover_home']:.3f}" if p["cover_home"] is not None else "  讓分覆蓋主 : n/a")
    if p.get("confidence") is not None:
        print(f"  confidence : {p['confidence']:.3f}(Jev 風格:分佈的勝負手差距;gate 下停用)")
    if not p["format_ok"]:
        print("  ⚠️ 格式解析失敗:模型輸出沒有符合訓練格式,請檢查 adapter 或 max-new-tokens。")


if __name__ == "__main__":
    main()
