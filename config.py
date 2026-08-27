"""短炒工作台的可編輯預設設定。"""

HK_UNIVERSE = {
    "0700.HK": "騰訊", "9988.HK": "阿里巴巴", "3690.HK": "美團",
    "1810.HK": "小米", "9618.HK": "京東", "9992.HK": "泡泡瑪特",
    "0388.HK": "港交所", "0005.HK": "匯豐", "1299.HK": "友邦",
    "2800.HK": "盈富基金", "2828.HK": "恒生中國企業", "3033.HK": "南方恒生科技",
}
US_UNIVERSE = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
    "META": "Meta", "TSLA": "Tesla", "GOOGL": "Alphabet", "AMD": "AMD",
    "AVGO": "Broadcom", "QQQ": "Nasdaq 100 ETF", "SPY": "S&P 500 ETF",
    "LITE": "Lumentum",
}
DEFAULT_UNIVERSE = {**HK_UNIVERSE, **US_UNIVERSE}
MODEL_DIR = "models"
MODEL_FILE = f"{MODEL_DIR}/lgbm_fwd5_clf.joblib"
FEATURE_FILE = f"{MODEL_DIR}/feature_list.json"
METRICS_FILE = f"{MODEL_DIR}/metrics.json"