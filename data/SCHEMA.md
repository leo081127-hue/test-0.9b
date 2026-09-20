# 賽事資料規格(matches.csv)

整條 pipeline(訓練、評估、推論)都吃同一份 CSV。你可以把「歷史賽事數據 + 賠率/盤口 + 自訂特徵表」合併成這一個檔案(以 `match_id` 為主鍵 join 起來)。

## 欄位說明

| 欄位 | 類型 | 必填 | 說明 |
|---|---|---|---|
| `match_id` | str | ✅ | 唯一賽事實 ID |
| `date` | ISO date | ✅ | 比賽日期(用於時間切分,避免資料洩漏) |
| `season` | str | ✅ | 賽季(如 `2025-26`) |
| `league` | str | ✅ | 聯賽(NBA / 英超 …) |
| `home` / `away` | str | ✅ | 主/客隊名 |
| `home_score` / `away_score` | int | ✅ | 最終比分(只有歷史賽事有;新賽事前留空) |
| `home_win` | 0/1 | ✅ | 主隊是否贏(平局請定義規則後轉成 0/1) |
| `margin` | float | ✅ | 主隊 - 客隊 分差(可正可負) |
| `open_spread` / `close_spread` | float | ✅ | 讓分盤,**數值 = 主隊讓分**。`-5.5` = 主隊讓 5.5 分;`+3` = 主隊受讓 3 分 |
| `open_total` / `close_total` | float | ⭕ | 大小分(total) |
| `open_ml_home` / `open_ml_away` | float | ✅ | 開盤 moneyline 賠率(十進位,主/客各一欄) |
| `close_ml_home` / `close_ml_away` | float | ✅ | 收盤 moneyline 賠率(評估時當「市場基準」用) |
| `home_form_w` / `home_form_l` | int | ⭕ | 主隊近 10 場勝/負 |
| `away_form_w` / `away_form_l` | int | ⭕ | 客隊近 10 場勝/負 |
| `home_avg_pts` / `away_avg_pts` | float | ⭕ | 近 10 場平均得分(或平均進球數) |
| `home_rest` / `away_rest` | int | ⭕ | 休息天數 |
| `home_record` / `away_record` | str `W-L` | ⭕ | 本季主場/客場紀錄(字串,如 `12-8`) |
| `h2h_home_w` / `h2h_away_w` | int | ⭕ | 近 5 次對決各隊勝場 |
| `lr_home`(/`lr_draw`/`lr_away`) | float | 🤖 自動 | **不需手動提供**:`common.attach_league_rates()` 依時間序列累計「該場開賽前」的聯賽結果分佈(先驗平滑)。只回看過去,無洩漏。是 Jev 足球預測的「league 先驗」技巧,防止模型系統性低估和局 |

- ⭕ = 選填;缺漏會以 `0` 當特徵、prompt 中顯示「無」。🤖 = pipeline 自動計算。
- **輸出格式**:`data/build_dataset.py --format {prose,typed}`。`prose`(預設)= 推理 + 固定格式行;`typed`(Jev 風格)= 嚴格 JSON `{"choice":…, "probabilities":[…], "confidence":…}`,程式可直接 `json.loads`。`common.parse_response_any()` 兩種都吃,GRPO reward / 評估 / 推論皆然。
- 自訂特徵表:只要你的欄位是數值型,加進 `common.py` 的 `FEATURE_COLS` 就能進 teacher 模型與 baseline;想進 prompt 文字,改 `common.format_game_features()`。
- 沒有盤口資料的賽事:讓分欄填 `0`、ML 填 `1.85/1.90`(≈五五波)即可,pipeline 仍會跑。

範例:`data/examples/matches_sample.csv`

## 足球(sport = soccer,1X2 三結果)

足球走「主勝 / 和 / 客勝」三結果路線(v1 不輸出让分覆蓋)。欄位差異:

| 欄位 | 說明 |
|---|---|
| `home_win` | 主隊是否**贏**(平局 = 0);三結果由 `margin` 決定:>0 主勝、=0 和、<0 客勝 |
| `open_ml_draw` / `close_ml_draw` | **必填**:1X2 的和局賠率(十進位)。有這欄 pipeline 自動走 soccer 路徑 |
| `home_form_d` / `away_form_d` | 近 10 場平局數(籃球留空) |
| `home_record` / `away_record` | 用 `W-D-L` 格式(如 `10-4-6`) |
| `open_spread` / `close_spread` / totals | v1 不需要,留空(NaN) |

`home_avg_pts` 在足球 = 平均每場進球數。

```bash
python data/generate_demo_soccer.py --seasons 5 --games-per-season 200 --out data/demo/soccer.csv
python data/build_dataset.py --matches data/demo/soccer.csv --out data/out_soccer --sport soccer
python eval/evaluate.py ... --sport soccer                        # 三結果 acc/Brier/LogLoss
python eval/backtest.py ... --sport soccer                         # 1X2 flat 1u 回測 + CLV
```

統計核心:`data/generate_demo_soccer.py` 用 bivariate Poisson + Dixon-Coles(1997)
低分修正(κ=0.10 放大 0-0/1-1、壓縮 1-0/0-1)+ 主場優勢;~25% 賽事開盤誤價、
收盤修正 75%(讓 teacher/模型有可學的 edge,market close 依然是強 baseline)。

## 合成 demo 資料

`python data/generate_demo_data.py` 產生籃球風格的假資料(含開/收盤、賠率、狀態特徵、~25% 盤口誤價與盤口看不到的休息天數效應,讓模型有可學的 edge)。
**正式使用請換成你的真實資料**——合成資料只驗證管線,不保證市場上的表現。
