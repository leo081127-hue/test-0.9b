"""把 LoRA adapter 合併進基座模型,存成標準 HF 格式——給 vLLM/SGLang 部署用。

合併後的目錄可以直接:
  vllm serve <out-dir> --trust-remote-code --dtype bfloat16 --reasoning-parser k2_horizon
(參數照 IFM/K2-Horizon-0.9B model card 的 serving quickstart)

範例:
  python tools/merge_adapter.py --model-id IFM/K2-Horizon-0.9B --trust-remote-code \
      --adapter-dir output/dpo --out-dir output/dpo_merged
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from train.sft import DTYPES, load_causal_lm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default="IFM/K2-Horizon-0.9B")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=sorted(DTYPES))
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    base = load_causal_lm(args.model_id, DTYPES[args.dtype], args.trust_remote_code)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    merged = model.merge_and_unload()

    os.makedirs(args.out_dir, exist_ok=True)
    merged.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print(f"merged model saved -> {args.out_dir}")
    print("部署範例(K2-Horizon):")
    print(f"  vllm serve {args.out_dir} --trust-remote-code --dtype bfloat16 \\")
    print("      --max-model-len 16384 --reasoning-parser k2_horizon")


if __name__ == "__main__":
    main()
