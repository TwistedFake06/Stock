# 股票分析助手 (Stock)

基于 **Python + Streamlit + Yahoo Finance** 的本地股票分析应用。

支持：

- 🇺🇸 美股（如 `AAPL`、`NVDA`）
- 🇭🇰 港股（如 `0700.HK`、`9988.HK`）
- 🇨🇳 A股（如 `600519` 茅台、`000001` 平安银行；也可用 `600519.SS` / `000001.SZ`）

## 功能（Streamlit 完整版）

> 下表是 **Streamlit 完整版**（`app.py`）功能。  
> 如果你当前使用的是 GitHub Pages 静态版（`index.html`），请看下一节「功能覆盖总览（完整版 vs 静态版）」。

| 模块             | 说明                                                                                | Streamlit 完整版 | GitHub Pages 静态版 |
| ---------------- | ----------------------------------------------------------------------------------- | ---------------- | ------------------- |
| **行情看板**     | 最新价、涨跌、市值、PE、K线 + 成交量 + 多空摘要                                     | ✅               | ✅（Lite）          |
| **多空分析**     | 综合判断 **看多 / 看空 / 中性**（得分仪表盘 + 分项依据）                            | ✅               | ✅（Lite）          |
| **入场与目标价** | 入场评级、买卖区、止损、短中期目标、**仓位计算**、**交易计划卡**、**财报/除息提醒** | ✅               | ✅（简化版）        |
| **策略验证**     | Walk-forward 重放 Bias 规则分，检查未来回报、超额回报、样本量与目标/止损路径        | ✅               | ❌                  |
| **综合分析**     | 评分卡 + 风险 / 支撑阻力 / 趋势 / 基本面 / 相对强弱 / 量价（Streamlit 专属）        | ✅               | ❌                  |
| **技术分析**     | SMA、布林带、RSI、MACD                                                              | ✅               | ✅（Lite 指标版）   |
| **多股对比**     | 多只股票归一化走势对比（Streamlit 专属）                                            | ✅               | ❌                  |
| **自选股**       | 本地 watchlist，含多空标签与得分排序（Streamlit 专属）                              | ✅               | ❌                  |

## 功能覆盖总览（完整版 vs 静态版）

为避免混淆，当前仓库有两种可用形态：

- **完整版（Streamlit / `app.py`）**：功能最完整，适合深度分析
- **静态版（GitHub Pages / `index.html` + `web/`）**：免后端、开箱即用，适合手机快速查看

| 能力                                                       | Streamlit 完整版 | GitHub Pages 静态版 |
| ---------------------------------------------------------- | ---------------- | ------------------- |
| 多市场代码输入（美股/港股/A股）                            | ✅               | ✅                  |
| K 线/价格趋势图                                            | ✅（Plotly）     | ✅（Chart.js）      |
| 成交量图                                                   | ✅               | ✅                  |
| SMA20 / SMA50                                              | ✅               | ✅                  |
| RSI                                                        | ✅               | ✅（RSI14）         |
| MACD                                                       | ✅               | ✅（MACD Hist）     |
| 多空评分（Bias Score）                                     | ✅               | ✅                  |
| 入场区 / 止损 / 目标位（计划卡）                           | ✅               | ✅（简化版）        |
| 综合分析（评分卡/风险/支撑阻力/趋势/基本面/相对强弱/量价） | ✅               | ❌                  |
| Bias 规则分 Walk-forward 验证                              | ✅               | ❌                  |
| 多股对比                                                   | ✅               | ❌                  |
| 自选股 Watchlist 持久化                                    | ✅               | ❌                  |
| iPhone Safari 适配                                         | ✅               | ✅                  |
| 无后端直接通过 GitHub Pages 使用                           | ❌               | ✅                  |

### 静态版当前特性（Stock Analyzer Lite）

- 快速输入或一键示例代码（`AAPL`、`NVDA`、`0700.HK`、`9988.HK`、`600519`、`000001`）
- 指标卡片：价格、涨跌、当日高低、RSI(14)、MACD Hist、SMA20、SMA50
- 图表：价格趋势 + 成交量
- 方向判断：`Strong Bullish / Bullish / Neutral / Bearish / Strong Bearish`
- 交易计划：买入/做空区间、止损、T1/T2 目标位（基于 ATR 的简化模型）
- 数据策略：优先读取同域缓存（GitHub Actions 生成的 Yahoo 数据），再尝试网络回退
- 浏览器缓存：本地缓存最近请求结果，减轻重复请求失败概率

### 综合分析包含

| 子模块       | 内容                                                         |
| ------------ | ------------------------------------------------------------ |
| **评分卡**   | 技术 / 基本面 / 风险调整 / 相对强弱 → 综合分与倾向           |
| **风险**     | 年化波动、最大回撤、夏普、VaR、Calmar、胜率、净值回撤图      |
| **支撑阻力** | 波段高低点聚类、均线、Pivot R/S、距离现价                    |
| **趋势结构** | 短中期方向、高低点结构、强度指数                             |
| **基本面**   | PE/PB/PS/PEG、ROE、净利率、营收/盈利增长、负债、股息、目标价 |
| **相对强弱** | 自动基准（SPY / 上证 / 恒生）、超额收益、β、相关性           |
| **量价**     | 量比、量价齐升/放量下跌、OBV 方向                            |

### 入场与目标价

| 项目         | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| **入场评级** | 较佳入场 / 可关注 / 观望 / 不宜追高 / 偏空回避               |
| **买卖区**   | 综合回踩、突破、超卖等给出的关注价格带                       |
| **止损**     | 近低点 / 均线 / ATR 参考止损                                 |
| **超短目标** | 约 **1 周内**：ATR×0.6–1、近5/10日高低、Pivot、SMA5、5日动量 |
| **短期目标** | 约 2 周–1 月：ATR、布林、20 日高低、Pivot、动量              |
| **中期目标** | 约 1–2 月：ATR×2.5–3.5、60 日高低、波段测算、分析师均价      |
| **辅助分析** | 本周剧本、观察清单、关键价位、T1/T2/T3 分批止盈              |
| **盈亏比**   | 相对止损到超短/短/中期看多目标的 R:R                         |

### 多空怎么算？

系统对下列信号加权打分（约 -100 ~ +100）：

- 均线排列 / 金叉死叉
- MACD 金叉死叉与柱状动能
- RSI 超买超卖与强弱区
- 布林带位置
- 近 5 / 20 根 K 线动量
- 放量涨跌

| 得分区间  | 结论     |
| --------- | -------- |
| ≥ +45     | 强烈看多 |
| +18 ~ +45 | 看多     |
| -18 ~ +18 | 中性     |
| -45 ~ -18 | 看空     |
| ≤ -45     | 强烈看空 |

### 策略验证页

- 每个历史日期只使用当日及以前的 K 线计算 Bias 规则分
- 分别显示未来 5 / 10 / 20 / 30 个交易日回报与相对 SPY/QQQ 超额回报
- 按分数区间报告样本量、上涨比例及方向调整后回报
- ATR 目标先触、止损先触、到期未触、同 K 歧义分开统计
- 规则分只代表规则强弱；没有经过校准前，不应解读为成功概率

## 手机浏览器 (iPhone)

- 布局与侧栏已按小屏优化：默认**收起侧栏**，点左上角打开菜单
- 图表自适应宽度；表格可左右滑动
- iPhone Safari 已加入 viewport/safe-area 适配，输入框聚焦不会自动放大
- 建议用 Safari / Chrome 全屏访问部署后的链接

## 通过 GitHub Actions 部署（推荐）

本项目是 Python + Streamlit 动态应用，**不适合 GitHub Pages 静态托管**。
推荐方式：用 GitHub Actions 自动构建并发布 Docker 镜像到 GHCR。

仓库已包含：`.github/workflows/github-action-deploy.yml`

### 首次启用

1. 把代码推到 GitHub 仓库（默认分支 `main`）
2. 打开仓库 **Settings → Actions → General**，确保允许 Workflow 运行
3. 推送到 `main` 或手动触发 **Actions → CI and Deploy Container**
4. 成功后会发布镜像到：

```text
ghcr.io/<你的GitHub用户名或组织>/stock-app:latest
```

### 运行镜像（任意可访问公网的服务器）

```bash
docker run -d --name stock-app -p 8501:8501 ghcr.io/<你的GitHub用户名或组织>/stock-app:latest
```

运行后，用 iPhone Safari 打开：

```text
http(s)://你的服务器域名或IP:8501
```

如果你希望我下一步直接补一个「自动部署到具体云平台（如 Render/Fly.io/Azure）」的 GitHub Action，我可以按你选的平台继续配置。

## GitHub Pages 访问说明（纯静态版）

- 已新增 `index.html` + `web/` 静态前端页面，支持在 GitHub Pages 直接运行（不依赖 Streamlit 服务端）。
- 工作流：`.github/workflows/pages-deploy.yml`，推送到 `main` 会自动部署到 Pages。
- 部署后访问：`https://twistedfake06.github.io/Stock/`
- iPhone Safari 可直接访问并输入代码分析（如 `AAPL`、`0700.HK`、`600519`）。
- 该静态版使用浏览器侧指标计算（SMA/RSI/MACD/多空评分），用于快速参考。
- 行情数据来自 Yahoo Finance：GitHub Actions 先拉取并生成同域缓存 `web/data/quotes.json`（同时写入 `quotes.json` 兼容路径），前端优先读缓存，绕过浏览器 CORS 限制。

## 部署到 Streamlit Community Cloud

### Streamlit Cloud 就绪检查（推荐先确认）

- 入口文件：`app.py`
- 依赖文件：`requirements.txt`
- Python 版本：建议 **3.11**
- 数据源：Yahoo Finance（若临时失败可重试）
- 本地语法检查：

```bash
py -m compileall -q .
```

> 说明：Streamlit 为服务端 Python 运行模式，浏览器不直接请求第三方行情 API，
> 因此不会像 GitHub Pages 静态版那样受前端 CORS 限制。

1. 把本项目推到 **GitHub** 公开或私有仓库（**不要**提交 `.env`）
2. 打开 [share.streamlit.io](https://share.streamlit.io) 登录
3. **New app** → 选仓库
4. **Main file path** 填：`app.py`
5. Python 版本建议 **3.11**
6. （可选）App settings → **Secrets** 填免费 API key，例如：

```toml
ALPHAVANTAGE_API_KEY = "你的key"
# FRED_API_KEY = "..."
# FINNHUB_API_KEY = "..."
```

7. Deploy 后打开 `https://xxx.streamlit.app`

可选：在 Cloud 的 Advanced settings 里确认 `requirements.txt` 被识别。

**说明：**

- 不设 Secrets 也能跑（yfinance 行情 + VIX/SPY 等）。
- Alpha Vantage / FRED / Finnhub 仅在 Secrets 或本地 `.env` 有 key 时启用。
- Cloud **读不到** 你电脑上的 `.env`，必须在网页 Secrets 里再贴一次 key。
- 免费 Alpha Vantage 额度很紧；Cloud 多用户刷新会更快耗尽，请依赖缓存、少扫大量股票。

### 常见问题（Streamlit Cloud）

- 启动失败（ImportError）：确认 `requirements.txt` 已包含依赖并重新 Deploy。
- 拉取行情失败：通常为 Yahoo 临时网络/频率限制，稍后重试。
- 页面空白或异常：在 Cloud 的 App logs 查看报错栈并按行修复。
- Secrets 无效：键名必须完全一致，例如 `ALPHAVANTAGE_API_KEY`（TOML 字符串加引号）。

> 期权/行情依赖 Yahoo Finance，Cloud 服务器网络需能访问 Yahoo；若拉取失败请稍后重试。

## 安装

需要 Python 3.10+。

```bash
cd Stock
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## 运行

```bash
streamlit run app.py
```

浏览器会自动打开（默认 `http://localhost:8501`）。

## 代码格式说明

| 市场     | 示例输入            | Yahoo 代码                |
| -------- | ------------------- | ------------------------- |
| 美股     | `AAPL`              | `AAPL`                    |
| 港股     | `0700.HK`           | `0700.HK`                 |
| 沪市 A股 | `600519`            | `600519.SS`               |
| 深市 A股 | `000001` / `300750` | `000001.SZ` / `300750.SZ` |

A 股 6 位代码会自动补全交易所后缀。

## 项目结构

```
Stock/
├── app.py                              # Streamlit 入口（Cloud / 本地；薄路由）
├── views/                              # 各功能页（勿改名 pages/，避免 multipage）
│   ├── common.py                       # 共用 UI 助手
│   ├── options_page.py / dashboard.py  # …
│   └── …
├── stock_service.py                    # 行情数据（yfinance + 缓存）
├── indicators.py                       # 技术指标
├── analysis.py                         # 多空评分（完整版）
├── charts.py                           # Plotly 图表
├── strategy_validation.py              # Bias 规则分 walk-forward 验证核心
├── options_*.py                        # 保留的旧期权计算模块（当前 UI 未挂载）
├── trade_plan.py                       # 交易计划与仓位
├── ui_mobile.py                        # 手机端样式与适配
├── tests/                              # unittest（无网络）
├── index.html                          # GitHub Pages 静态入口（Lite）
├── web/
│   ├── app.js                          # 静态版逻辑与指标计算
│   ├── styles.css                      # 静态版 UI 样式
│   └── data/quotes.json                # 静态版行情缓存（Actions 生成）
├── scripts/build_static_data.py        # 生成静态版 Yahoo 缓存
├── .github/workflows/
│   ├── github-action-deploy.yml        # GHCR 容器发布（Streamlit）
│   └── pages-deploy.yml                # GitHub Pages 部署（静态版）
├── requirements.txt
├── watchlist.json                      # 运行后自动生成（完整版）
└── README.md
```

## 注意

- 数据来自 Yahoo Finance，可能有延迟或个别 A 股字段不全。
- **定位：实盘辅助工具**
  - POP + **到期 EV**（分段盈亏积分）+ **管理 EV**（50% 止盈 / 2R 止损路径）
  - 回测引擎 `bs_mtm_v3`：Bull Put / Bear Call、BS 盯市、佣金+摩擦、单仓不重叠
  - 数据源可能延迟，**不构成投资建议 / 非自动下单**
- 若网络访问 Yahoo 受限，请检查代理 / 防火墙设置。
