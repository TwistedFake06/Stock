# Copilot Instructions – Stock Analysis App

## 專案概述
這是一個混合式股票分析工具，支援美股、港股與 A 股。
- 完整版：Streamlit（`app.py` + `views/`）
- 精簡版：靜態 GitHub Pages（`index.html` + `web/`），使用快取的 `quotes.json`
目標：協助紀律化交易（技術分析、SOP 規則、風險控制），不是自動交易系統。

## 技術棧
- Python + Streamlit
- Yahoo Finance（搭配本地快取）
- 靜態前端：HTML / JS / CSS
- 部署：Docker + GitHub Actions + GitHub Pages

## 主要結構
- `analysis.py`、`indicators.py`、`trade_sop.py`、`trade_plan.py` → 核心業務邏輯
- `stock_service.py` → 資料抓取與快取
- `views/` → Streamlit UI
- `web/` + `index.html` → 靜態精簡版
- `backtest/` → 策略驗證
- `scripts/` → 資料生成與建置腳本

## 編碼準則
- 優先使用清晰易讀的程式碼，避免過度聰明的寫法。
- 函式保持小而專注。
- 修改核心分析或 SOP 邏輯時，盡量維持現有公開函式簽名不變。
- 任何功能變更都要同時考慮 Streamlit 完整版與靜態精簡版。
- 優先顧及手機體驗（特別是 iPhone Safari）。
- 未經明確要求，不要引入付費 API 或沉重的新依賴。

## 邊界限制
- 絕對不要硬編碼密鑰或 API Key。
- 不要在沒有充分理由的情況下破壞風險控制或 SOP 相關邏輯。
- 避免一次大幅重構同時影響完整版與精簡版，除非必要。

## 建議做法
1. 先定位負責的模組（資料 → 分析 → UI）。
2. 優先做漸進式改進，而非大規模重寫。
3. 進行功能強化時，保持「風險優先」與「SOP 驅動」原則。
4. 提出架構變更時，請說明取捨。

<!-- 
## 未來可加區塊範例（需要時再打開）

## Commands
- 本地執行：`streamlit run app.py`
- 建置靜態版：...

## 個人 SOP 重點
- ...

## 已知技術債
- ...
-->