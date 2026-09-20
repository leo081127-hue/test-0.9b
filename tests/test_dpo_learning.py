"""DPO 行為測試:偏好學習真的發生(loss 下降、chosen 隱含優勢變大)。

不用 mock——直接跑 train/dpo.py 的核心元件(completion_logprob /
ref_completion_logprob)與它同款的訓練迴圈,在一個「答案明確」的玩具任務上
驗證:訓練後 policy 應該 (a) DPO loss 下降,(b) 更偏好 chosen。
"""
import torch
import torch.nn.functional as F
import pytest

trl = pytest.importorskip("trl")  # 確保與 CI 相同的依賴環境

from tokenizers import Tokenizer, models as tmodels, pre_tokenizers, trainers as ttrainers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from peft import LoraConfig, TaskType, get_peft_model
from train.dpo import completion_logprob, ref_completion_logprob

torch.manual_seed(0)

CHOSEN = "最終預測: 主隊\n機率: 主隊 0.62, 客隊 0.38"
REJECTED = "最終預測: 客隊\n機率: 主隊 0.38, 客隊 0.62"


def _prompt(i: int) -> str:
    return (f"主隊 近10場 {(i % 7) + 2}勝{9 - (i % 7) - 2}負, 客隊 近10場 {(i % 5) + 1}勝"
            f"{9 - (i % 5) - 1}負, 收盤讓分 -{(i % 4) + 1}.5")


@pytest.fixture(scope="module")
def tiny_dpo():
    texts = [_prompt(i) + CHOSEN for i in range(40)] + [_prompt(i) + REJECTED for i in range(40)]
    tok = Tokenizer(tmodels.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel()
    tok.train_from_iterator(texts, ttrainers.BpeTrainer(vocab_size=512,
                                                        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"]))
    ftok = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>",
                                   bos_token="<bos>", eos_token="<eos>")
    cfg = LlamaConfig(vocab_size=len(ftok), hidden_size=256, intermediate_size=512,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=256, tie_word_embeddings=True)
    base = LlamaForCausalLM(cfg)
    policy = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, bias="none",
                                             task_type=TaskType.CAUSAL_LM))
    policy.train()
    return ftok, policy


def _batch_loss(policy, ftok, pairs, beta):
    losses, margins = [], []
    for pr, c, r in pairs:
        pc = completion_logprob(policy, ftok, pr, c, 256)
        pj = completion_logprob(policy, ftok, pr, r, 256)
        rc = ref_completion_logprob(policy, ftok, pr, c, 256)
        rr = ref_completion_logprob(policy, ftok, pr, r, 256)
        losses.append(-F.logsigmoid(beta * ((pc - rc) - (pj - rr))))
        margins.append((pc - rc) - (pj - rr))
    return torch.stack(losses).mean(), torch.stack(margins).mean()


def test_dpo_loss_decreases_and_policy_prefers_chosen(tiny_dpo):
    ftok, policy = tiny_dpo
    pairs = [(_prompt(i), CHOSEN, REJECTED) for i in range(40)]
    beta, lr, steps, bsz = 0.3, 1e-3, 30, 8
    opt = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=lr)
    gen = torch.Generator().manual_seed(7)

    with torch.no_grad():
        loss0, margin0 = _batch_loss(policy, ftok, pairs[:bsz], beta)
    hist = [(loss0.item(), margin0.item())]

    for step in range(steps):
        idx = torch.randperm(len(pairs), generator=gen)[:bsz].tolist()
        batch = [pairs[j] for j in idx]
        loss, margin = _batch_loss(policy, ftok, batch, beta)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        hist.append((loss.item(), margin.item()))

    losses = [h[0] for h in hist]
    margins = [h[1] for h in hist]
    for v in losses:
        assert torch.isfinite(torch.tensor(v))
    # (a) loss 明顯下降(初始 = -logsigmoid(0) = log 2 ≈ 0.693)
    assert losses[0] == pytest.approx(torch.log(torch.tensor(2.0)).item(), abs=1e-3)
    assert losses[-1] < losses[0] - 0.05, losses
    # (b) chosen 的隱含優勢 margin = (logπ/πref chosen) - (logπ/πref rejected) 變大
    assert margins[-1] > margin0 + 0.05, (margins[0], margins[-1])
    assert margins[-1] > 0.0
    # (c) policy 直接偏好 chosen 的 logprob 高於 rejected
    with torch.no_grad():
        pc = completion_logprob(policy, ftok, _prompt(3), CHOSEN, 256)
        pr = completion_logprob(policy, ftok, _prompt(3), REJECTED, 256)
        assert pc > pr
