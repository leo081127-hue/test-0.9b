"""驗證 DPO 的 reference 技巧:關閉 adapter 層 == 純 base 模型(省一份 VRAM 的正確性)。"""
import torch
import pytest

from tokenizers import Tokenizer, models as tmodels, pre_tokenizers, trainers as ttrainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from peft import LoraConfig, TaskType, get_peft_model

from train.dpo import completion_logprob, ref_completion_logprob

torch.manual_seed(0)


@pytest.fixture(scope="module")
def tiny():
    texts = [
        "主隊 近10場 6勝4負,客隊 近10場 4勝6負,收盤讓分 -3.5,機率 主隊 0.60 客隊 0.40"
    ] * 30
    tok = Tokenizer(tmodels.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    tok.train_from_iterator(texts, ttrainers.BpeTrainer(vocab_size=400,
                                                        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"]))
    ftok = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>",
                                   bos_token="<bos>", eos_token="<eos>")
    cfg = LlamaConfig(vocab_size=len(ftok), hidden_size=256, intermediate_size=512,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, tie_word_embeddings=True)
    base = LlamaForCausalLM(cfg)
    return ftok, base


def test_disable_adapter_equals_base(tiny):
    ftok, base = tiny
    policy = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, bias="none",
                                             task_type=TaskType.CAUSAL_LM))
    prompt = "主隊 6勝4負 客隊 4勝6負"
    comp = "最終預測: 主隊 機率 主隊 0.60 客隊 0.40"

    with torch.no_grad():
        p_base = completion_logprob(base, ftok, prompt, comp, 512)
        p_ref = ref_completion_logprob(policy, ftok, prompt, comp, 512)
    assert torch.allclose(p_base, p_ref, atol=1e-5), (p_base.item(), p_ref.item())

    # adapter 啟用時(權重已動過)兩者應該不同
    policy.train()
    with torch.no_grad():
        for p in policy.parameters():
            if p.requires_grad:
                p.data.add_(torch.randn_like(p) * 0.1)
    with torch.no_grad():
        p_pol = completion_logprob(policy, ftok, prompt, comp, 512)
        p_ref2 = ref_completion_logprob(policy, ftok, prompt, comp, 512)
    assert not torch.allclose(p_pol, p_ref2, atol=1e-3)


def test_completion_logprob_shape_and_finite(tiny):
    ftok, base = tiny
    v = completion_logprob(base, ftok, "abc 測試", "0.5 0.5", 256)
    assert v.dim() == 0  # scalar
    assert torch.isfinite(v)
    # 空 completion → 0
    v0 = completion_logprob(base, ftok, "abc 測試", "", 256)
    assert v0.item() == 0.0
