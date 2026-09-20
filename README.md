# 0.9B 體育賽事預測模型 — 微調開源基座完整 Pipeline

用 **0.9B 級開源模型**(預設 [IFM/K2-Horizon-0.9B](https://huggingface.co/IFM/K2-Horizon-0.9B))微調一個
**體育賽事預測模型**:輸入「歷史賽事特徵 + 盤口/賠率 + 自訂特徵」,輸出
**勝負/讓分結果 + 機率 + 推理理由**,並帶有**校準機率**(可以用 Brier 分數評估)。

```
你的資料(3 張表)          合成 demo 資料(沒資料時)
      │  join 成 matches.csv          │
      └──────────────┬────────────────┘
                     ▼
        data/build_dataset.py
        時間切分 train/val/test + teacher(GBM)蒸餾目標機率
                     ▼
        train/sft.py      階段1 LoRA SFT:學「盤口+特徵 → 機率+理由」
                     ▼
        train/dpo.py      階段2 (選用) DPO:修正過度自信 / 方向錯誤
                     ▼
        train/grpo.py     階段3 (選用) GRPO RL:reward = Brier + 格式
                     ▼
        eval/evaluate.py  acc / Brier / LogLoss / ECE,對照市場收盤賠率 & LogReg
                     ▼
        infer/predict.py  單筆新賽事預測 CLI
```

---

## 1. 先說三句實話(重要)

1. **真正的對手是收盤賠率。** 市場收盤賠率(去除抽水後)就是目前所有公開資訊的綜合。
   穩定贏過收盤線(closing line value)是專業量化團隊的事。本 repo 的評估**一定會把
   市場收盤賠率列為基準**——你的模型只有贏過它,才談得上有價值。
2. **0.9B 的價值在哪?** ① 便宜快(單張消费級 GPU 就能微調);② 可解釋——每筆預測附推理;
   ③ 能透過「teacher 蒸餾 + RL」吸收盤口特徵 → 機率 的非線性模式;④ K2-Horizon 本身就是
   RL(GRPO)訓練出來的 reasoning 模型,接續 RL 階段最順。
3. **demo 合成資料裡故意放了 ~25% 的「盤口誤價」與盤口看不到的休息天數效應**,
   所以 demo 上模型會「贏過市場」——那只是驗證管線用,換上你的真實資料後,
   成績對照市場基準才有意義(真實市場的漏洞通常少得多,甚至沒有)。

## 2. 基座模型選擇

| 模型 | 參數 | 說明 |
|---|---|---|
| `IFM/K2-Horizon-0.9B`(預設) | 0.9B | 128K context、reasoning 型、GRPO 血統。⚠️ 自帶 custom code,所有命令加 `--trust-remote-code`;bf16 使用;推論建議 vLLM(見該 model card) |
| `Qwen/Qwen3-1.7B`(備用) | 1.7B | 生態支援最好、效果通常更好;8GB 显存可 LoRA 微調 |
| `Qwen/Qwen2.5-1.5B`(備用) | 1.5B | 同上,更省資源 |

換基座只需改 `MODEL_ID`(見下)。本 repo 的 prompt 格式、解析格式與基座無關。

## 3. 硬體 / 時間估計(1,500 筆樣本 × 3 epochs,seq≤1024)

| 顯存 | 可行內容 |
|---|---|
| 8GB (RTX 3060/4060) | SFT + DPO(建議 `--grad-ckpt`);GRPO 需調小 `--num-generations 4` |
| 16GB (3070/4060 Ti) | SFT + DPO 從容 |
| 24GB (3090/4090/A5000) | 全三階段,GRPO `--num-generations 8` |

免費方案:Colab Pro(T4 16GB)、Kaggle Notebook(24GB,每週 30h GPU)都行。
沙盒內(無 GPU)可用 `scripts/smoke_test.sh` 驗證管線(本地小模型,CPU)。

## 4. 快速開始

```bash
pip install -r requirements.txt

# A) 有真實資料:先把「歷史賽事 + 賠率/盤口 + 特徵表」join 成一個 csv(欄位見 data/SCHEMA.md)
# B) 沒有:用合成 demo 資料跑通管線
export MATCHES=data/demo/matches.csv          # A) 時改成你的路徑
export MODEL_ID=IFM/K2-Horizon-0.9B           # 預設值
export EXTRA_ARGS="--trust-remote-code"       # K2-Horizon 需要;換 Qwen 就留空

bash scripts/run_all.sh
# 輸出: output/sft、output/dpo、output/report.json、output/calibration.png、
#       output/preds.csv、output/backtest.json、output/equity.png
```

### 沒有 GPU?用 Colab

`colab/sport_predict_colab.ipynb`:開 Colab → 上傳/打開這個 notebook → 填入你的 repo
URL → 跑(免費 T4 可跑 SFT+DPO+評估;GRPO 建議 A100)。

### 你的真實資料開跑前

```bash
python data/qa_report.py --matches 你的資料.csv   # 重複 id/比分矛盾/賠率 vig/覆蓋率…
```

### 單筆新賽事預測

```bash
python infer/predict.py --model-id "$MODEL_ID" $EXTRA_ARGS \
    --adapter-dir output/dpo --features-json my_game.json
# my_game.json 欄位見 data/SCHEMA.md(結果欄位 home_win/margin 等可以不填)
```

### CPU smoke test(本沙盒 / CI,無網路無 GPU 也能跑)

```bash
bash scripts/smoke_test.sh
```

## 5. 你的資料怎麼放進來

你說的三類資料 → 合併成 **一個 csv**,以 `match_id` join:

| 你的資料 | 對應欄位 |
|---|---|
| 歷史賽事數據 | `match_id, date, season, league, home, away, home_score, away_score, home_win, margin` |
| 賠率 / 盤口 | `open_spread, close_spread, open_total, close_total, open_ml_home/away, close_ml_home/away` |
| 自訂特徵表 | `home_form_w/l, away_form_w/l, home_avg_pts, away_avg_pts, home_rest, away_rest, home_record, away_record, h2h_home_w, h2h_away_w` |

- 完整欄位說明:`data/SCHEMA.md`,範例:`data/examples/matches_sample.csv`
- **讓分慣例**:數值 = 主隊讓分;`-5.5` = 主隊讓 5.5 分。
- **防資料洩漏**:`build_dataset.py` 一律依 `date` 時間切分;特徵欄(近10場狀態、紀錄)
  必須是「該場開打前」的值——生成 demo 資料時已如此處理,你的特徵表也請照此對齊。
- 想加自訂特徵:數值型欄位加進 `common.py` 的 `FEATURE_COLS` 就會進 teacher 與 baseline;
  想讓模型「看得到」,再把它加進 `common.format_game_features()`。
- 沒有盤口資料?讓分填 `0`、ML 填 `1.85/1.90` 即可跑(但評估時 market 基準會失效)。

## 6. 訓練細節

### 階段 1:SFT(`train/sft.py`)
- LoRA(r=16, α=32)只训 attention/MLP projections,0.9B 下約 +0.5% 可訓參數。
- **機率目標從哪來**:用 `HistGradientBoostingClassifier`(只用 train 段擬合)對
  「勝負」與「讓分覆蓋」各擬合一個 teacher,把它的機率 + 模板推理文字當 SFT 目標——
  這是「tabular teacher → LLM 蒸餾」,0.9B 模型學「看哪些特徵」比從原始標籤學更穩。
- 只對 assistant 段算 loss(prompt 段 mask 成 -100)。
- 建議:`--epochs 3` 起,盯 `eval_loss`;val loss 開始回升就停(小模型很容易背答案)。

### 階段 2:DPO(`train/dpo.py`)
- pairs 由 teacher 產出:`chosen` = 正確判斷的回應;`rejected` = 把機率往反方向扭曲的回應。
- 目的:壓制過度自信、把方向錯誤的輸出概率降下來。1 個 epoch、lr 5e-5 通常足夠。

### 階段 3:GRPO(`train/grpo.py`,選用)
- 你的「enhance RL」需求就放在這:體育預測有**可驗證的 reward**(結果出來就知道準不準),
  正適合 on-policy RL。reward = 格式分 + 勝率 Brier + 方向分 + 讓分覆蓋 Brier(權重 0.3)。
- 這與 K2-Horizon 自己的訓練路線一致(它的 math/code 專家都是 GRPO 練出來再 merge)。
- 需 `pip install trl datasets`;腳本已對 **TRL 1.x 的 GRPOTrainer API 實跑驗證**
  (reward 函數簽名、`datasets.Dataset`、batch 整除限制)。
- ⚠️ TRL 1.x 要求 `--batch-size` 能被 `--num-generations` 整除(如 8×2、16×8);
  資料行數不能被 `--num-generations` 整除時腳本會**自動 pad** 並印出警告。
- 0.9B + `--num-generations 8` 建議 24GB 顯存;8GB 用 `--num-generations 2 --batch-size 2`。

#### RL 準備度審計(`--reward-audit`):確認 RL 真的有用

GRPO 的梯度訊號來自**同一 prompt 的群組內 reward 差異**。若模型連格式都輸出不出來
(reward 全 0),群組方差 = 0,RL 什麼都學不到(log 會顯示
`frac_reward_zero_std: 1`)。所以正式流程是:

```bash
# 1) SFT(+DPO)後,先量「RL 前」基線:格式率/Brier/方向/覆蓋各差多少
python train/grpo.py $EXTRA_ARGS --model-id "$MODEL_ID" --adapter-dir output/dpo \
    --train-jsonl data/out/train.jsonl --reward-audit --audit-n 50 \
    --audit-out output/reward_audit_before.json
# 2) 跑 GRPO 訓練
python train/grpo.py $EXTRA_ARGS --model-id "$MODEL_ID" --adapter-dir output/dpo \
    --train-jsonl data/out/train.jsonl --output-dir output/grpo \
    --num-generations 8 --batch-size 8
# 3) 再量「RL 後」:reward_mean 必須上升(且不是只靠 format_rate 上升)
python train/grpo.py $EXTRA_ARGS --model-id "$MODEL_ID" --adapter-dir output/grpo \
    --train-jsonl data/out/train.jsonl --reward-audit --audit-n 50 \
    --audit-out output/reward_audit_after.json
```

`run_all.sh`(USE_GRPO=1)已自動串好這三步。

### 部署:`tools/merge_adapter.py`
把 LoRA 合併進基座 → 標準 HF 目錄,直接給 vLLM/SGLang 用:
```bash
python tools/merge_adapter.py --model-id "$MODEL_ID" $EXTRA_ARGS \
    --adapter-dir output/dpo --out-dir output/dpo_merged
vllm serve output/dpo_merged --trust-remote-code --dtype bfloat16 --reasoning-parser k2_horizon
```

## 7. 評估指標(`eval/evaluate.py`)

| 指標 | 意義 | 怎么看 |
|---|---|---|
| acc / pred_acc | 預測主/客是否正確 | 體育 ~52-55% 就算不錯 |
| **brier** | 機率校準品質(越小越好,0.25=coin) | **最重要的指標** |
| logloss | 同 brier 的 log 版本 | 越小越好 |
| ECE | 置信度 vs 實際準確率 | 越接近 0 越誠實 |
| cover_acc | 讓分覆蓋準確率 | 盤口約 50%,看是否 >50% |
| parse_error_rate | 格式解析失敗率 | 應該 <2%,否則調 `--max-new-tokens` 或重訓 |

`report.json` 會同時給 **LLM / Market close / LogReg / Coin** 四組數字——
**LLM 的 brier 要贏過 Market close 才算有意義**。`calibration.png` 畫校準曲線。
另外 `report["by_season"]` 給**分季** LLM vs Market 的 acc/brier,看模型在更晚的賽季
(時間外推)是否退化——體育模型最怕這個。

### 回測:`eval/backtest.py`(贏錢了嗎?)

`evaluate.py --pred-out` 會把逐筆模型機率 + 開/收盤賠率匯出 CSV;backtest 用
「|p-0.5| ≥ 門檻才下注、flat 1u、買進價=開盤賠率」的策略算:

| 指標 | 意義 |
|---|---|
| ROI | 每下注一場的期望獲利(flat 1u) |
| **CLV** | 下注方機率 vs 收盤市場隱含機率的平均差。**> 0 才是長期可贏的訊號**(價格贏過收盤市場) |
| max_drawdown / 連敗 | 資金曲線的最大回撤與最長連敗(能不能撐得住) |
| 門檻掃描 | 0.50~0.70 各門檻的 bets/ROI(選你的下注風格) |

會同時跑 **market_open** 基準線(賭場開盤價去抽水後下注)做對照——
它的 CLV 理論上 ≈ 0,可用來 sanity check。

## 8. 故障排除

| 症狀 | 解法 |
|---|---|
| `trust_remote_code` 載入失敗(K2-Horizon) | 確認 `transformers>=4.44`、PyTorch 2.x;該模型官方建議 bf16(`--dtype bfloat16`,預設就是) |
| OOM | `--grad-ckpt`、降 `--batch-size`/`--grad-accum`、DPO 用 4bit reference(目前用 disable-adapter 技巧,已省一份 VRAM) |
| parse_error_rate 高 | 調大 `--max-new-tokens`(預設 384)、確認用對 adapter;推理用 `temperature=0` |
| GRPO 報 `generation_batch_size ... divisible by num_generations` | TRL 1.x 限制:`--batch-size` 要能被 `--num-generations` 整除(如 2/2、8/8、16/8) |
| GRPO 報 `train_dataset must be a Dataset` | 腳本已用 `datasets.Dataset.from_list`;若改動資料輸入,記得保持該轉換 |
| GRPO log 顯示 `frac_reward_zero_std: 1`、reward 全 0 | **reward 饑餓**:模型輸出解析不出格式 → 群組內無差異、RL 無訊號。先跑 `--reward-audit` 看 `format_rate`;通常代表 SFT 沒學好(加 epoch/查 val loss)或 `--max-completion` 太短 |
| val loss 下降但 test 贏不了 market | 正常!檢查是否資料太少/盤口品質差;先確認 LogReg baseline 本身能否贏 market(它贏不了,LLM 更難) |
| 想加速推論 | vLLM serve(見 K2-Horizon model card 的 quickstart),adapter 合併後部署 |

## 9. 專案結構

```
common.py                    # prompt 格式 / 回應解析 / 特徵矩陣 / 指標(所有模組共用)
data/
  SCHEMA.md                  # matches.csv 欄位規格
  examples/matches_sample.csv
  generate_demo_data.py      # 合成 demo 資料
  build_dataset.py           # 時間切分 + teacher 蒸餾 → train/val/test.jsonl + dpo_pairs.jsonl
train/
  sft.py                     # LoRA SFT(主)
  dpo.py                     # DPO(選用)
  grpo.py                    # GRPO RL(選用,需 trl)
eval/evaluate.py             # 評估 + baseline 對照 + 分季統計 + report.json + calibration.png + pred-out
eval/backtest.py             # 回測:ROI / CLV / drawdown / 門檻掃描(對照 market_open)
data/qa_report.py            # 真實資料訓練前品質檢查(重複 id/矛盾標籤/vig/…)
infer/predict.py             # 單筆預測 CLI
tools/smoke_tiny_model.py    # 本地 tiny 模型(無網路 smoke test 用)
tools/merge_adapter.py       # LoRA 合併進基座 → vLLM/SGLang 部署
colab/sport_predict_colab.ipynb  # Colab 訓練 notebook
tests/                       # pytest:單元 + 資料不變量 + 全管線整合(無網路無 GPU 可跑)
scripts/run_all.sh           # 一條龍(GPU 機器)
scripts/smoke_test.sh        # CPU smoke(無 GPU 也能跑)
.github/workflows/ci.yml     # CI:push/PR 自動跑全部測試
```

## 10. 測試

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q          # 全部(含整合管線,2 核 CPU 約 5~6 分鐘)
python -m pytest tests/ -q -k "not pipeline"   # 只快測(約 30 秒)
```

覆蓋(11 個檔、80 項):
- `test_common.py` — prompt/回應格式、解析(含全形符號、未加和為 1 的正規化)、Brier/LogLoss/ECE、特徵矩陣
- `test_generate.py` — 生成器確定性、時間序列不變量、**近10場/對決特徵不用未來資料**、盤口區隔度(AUC>0.6)
- `test_build_dataset.py` — 時間切分無反序、jsonl schema、回應機率 == teacher 值、DPO pairs 方向扭曲
- `test_dpo_ref.py` — DPO 的「disable adapter == base」reference 技巧的數值正確性
- `test_dpo_learning.py` — **DPO 行為測試**:偏好學習真的發生(loss 下降、chosen 隱含優勢變大、policy 更偏好 chosen)
- `test_grpo_reward.py` — GRPO reward 手算驗證(格式/Brier/方向/覆蓋)+ 11 個解析健壯性邊緣案例
  (正規化、全形符號、極端機率、0/0 拒判、batch、bytes、分項加總一致性)
- `test_grpo_rl.py` — **GRPO 行為測試**:reward 光譜單調性(有訊號)、trainer log 指標有限無 NaN、
  reward 饑餓(群組方差=0)時數值穩定、rows 自動 pad、`run_audit` 與手算分項一致
- `test_backtest.py` — ROI/CLV/drawdown/連敗 手算案例 + 門檻邊界
- `test_qa.py` — 乾淨資料 0 error;重複 id/矛盾標籤/壞賠率要被抓出來
- `test_pipeline.py` — 整合:generate→build→tiny→SFT(含 val)→DPO→**GRPO**→評估→推論(csv+json 兩路)→**merge 部署**

> RL 測試的誠實邊界:CPU 上隨機初始化的 tiny 模型學不會輸出格式(reward 全 0、群組方差 0),
> 所以套件不斷言「GRPO 一定提升 reward」——那需要 GPU + 已 SFT 的模型,流程見 §6
> 「RL 準備度審計(`--reward-audit`)」小節。

## 11. 足球 1X2(三結果:主勝/和/客勝)

```bash
# 有真實足球資料(matches.csv 含 open_ml_draw/close_ml_draw 即自動走 soccer 路徑;
# 欄位差異見 data/SCHEMA.md「足球」節)
export SPORT=soccer
bash scripts/run_all.sh
# 沒資料:Dixon-Coles 風格合成 demo(與 repo 內 data/demo/soccer.csv 相同參數)
python data/generate_demo_soccer.py --seasons 5 --games-per-season 200 --out data/demo/soccer.csv
```

- **格式**:回應 = `最終預測: 主隊/和局/客隊` + `機率: 主隊 x, 和局 x, 客隊 x`(三者和=1)。
- **統計核心**(demo 生成器):bivariate Poisson(每隊攻擊/防守強度 + 主場優勢)+
  **Dixon-Coles(1997)低分修正**(κ=0.10:獨立 Poisson 低估 0-0/1-1、高估 1-0/0-1,
  修正後和局率才接近真實的 ~21-26%)。
- **teacher**:三類 HGB(主/和/客)蒸餾;**評估**:三結果 acc / 多類 Brier / LogLoss / ECE,
  對照 **Market close(1X2 去抽水)**、LogReg(多項)、Coin(均分 1/3,Brier=2/3)。
- **回測**:max 機率 ≥ 門檻才下注、flat 1u @ 開盤 1X2 賠率,CLV vs 收盤市場
  (market_open 基準 CLV≈0 當 sanity check)。
- **GRPO reward**:格式 1 + 多類 Brier + 方向 0.5(soccer jsonl 帶 `outcome` 欄,自動切換)。
- 誠實預期:1X2 的 argmax acc 天花板約 50-55%(和局幾乎永遠不是 argmax),**看 Brier**;
  v1 不含讓分/大小分(下一版加 Asian handicap)。

**demo 結果**(`data/demo/soccer.csv`,5 季 × 200 場,seed 7;70/15/15 時間切分,test 150 場):

| model | acc | Brier | LogLoss | ECE |
|---|---|---|---|---|
| Market close(1X2 去抽水) | 0.527 | **0.5643** | **0.9543** | 0.067 |
| teacher(我們模型的代理,三類 HGB) | **0.547** | 0.6077 | 1.0187 | — |
| LogReg(多項) | 0.527 | 0.5847 | 0.9906 | 0.053 |
| Coin(均分 1/3) | 0.480 | 0.6667 | 1.0986(=ln 3) | 0.147 |

1X2 回測(flat 1u @ 開盤賠率,門檻 = max 機率):

| 策略 | n | 命中率 | ROI | CLV | maxDD |
|---|---|---|---|---|---|
| teacher @ 0.55 | 79 | 60.8% | +2.9% | **+0.131** | 7.4u |
| market_open 基準(收盤 favorite) | 55 | 72.7% | +8.9% | ≈ +0.004(sanity) | 2.6u |

> teacher 的 argmax 方向略優於市場,但機率校準仍不及市場收盤(多類 Brier 0.61 vs 0.56)——
> 這正是 RL(GRPO/DPO)該去推的部分:讓 0.9B 的機率「更貼近真實分佈」。

## 12. Roadmap(還沒做)

- 足球讓分(Asian handicap)+ 大小分(Over/Under)輸出
- 多聯賽聯合訓練、傷兵/陣容特徵
- GRPO 之上的 on-policy DPO 迭代、KL 調參

## 13. 免責

本專案僅供**學習與研究**。體育賠率由持牌機構提供,任何預測都不保證獲利;
請遵守所在地法律,不要把它當成投注建議。
