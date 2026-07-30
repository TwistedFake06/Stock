# 股票分析助手 (Stock)

基于 **Python + Streamlit + Yahoo Finance** 的本地股票分析应用。

支持：

- 🇺🇸 美股（如 `AAPL`、`NVDA`）
- 🇭🇰 港股（如 `0700.HK`、`9988.HK`）
- 🇨🇳 A股（如 `600519` 茅台、`000001` 平安银行；也可用 `600519.SS` / `000001.SZ`）

## 功能

| 模块             | 说明                                                                                |
| ---------------- | ----------------------------------------------------------------------------------- |
| **行情看板**     | 最新价、涨跌、市值、PE、K线 + 成交量 + 多空摘要                                     |
| **多空分析**     | 综合判断 **看多 / 看空 / 中性**（得分仪表盘 + 分项依据）                            |
| **入场与目标价** | 入场评级、买卖区、止损、短中期目标、**仓位计算**、**交易计划卡**、**财报/除息提醒** |
| **期权价差**     | 仅 **Vertical**：先判看多/看空，再推荐最佳 Bull/Bear Put/Call 垂直价差              |
| **综合分析**     | 评分卡 + 风险 / 支撑阻力 / 趋势 / 基本面 / 相对强弱 / 量价                          |
| **技术分析**     | SMA、布林带、RSI、MACD                                                              |
| **多股对比**     | 多只股票归一化走势对比                                                              |
| **自选股**       | 本地 watchlist，含多空标签与得分排序                                                |

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

## 部署到 Streamlit Community Cloud

1. 把本项目推到 **GitHub** 公开或私有仓库
2. 打开 [share.streamlit.io](https://share.streamlit.io) 登录
3. **New app** → 选仓库
4. **Main file path** 填：`app.py`
5. Python 版本建议 **3.11**
6. Deploy 后用手机浏览器打开 `https://xxx.streamlit.app`

可选：在 Cloud 的 Advanced settings 里确认 `requirements.txt` 被识别。

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
├── app.py              # Streamlit 主界面
├── stock_service.py    # 行情数据（yfinance）
├── indicators.py       # 技术指标
├── charts.py           # Plotly 图表
├── requirements.txt
├── watchlist.json      # 运行后自动生成的自选列表
└── README.md
```

## 注意

- 数据来自 Yahoo Finance，可能有延迟或个别 A 股字段不全。
- **仅供学习与研究，不构成任何投资建议。**
- 若网络访问 Yahoo 受限，请检查代理 / 防火墙设置。
