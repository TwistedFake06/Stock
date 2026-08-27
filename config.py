"""短炒工作台的可編輯預設設定。"""

HK_UNIVERSE = {
    "0700.HK": "騰訊", "9988.HK": "阿里巴巴", "3690.HK": "美團",
    "1810.HK": "小米", "9618.HK": "京東", "9992.HK": "泡泡瑪特",
    "0388.HK": "港交所", "0005.HK": "匯豐", "1299.HK": "友邦",
    "2800.HK": "盈富基金", "2828.HK": "恒生中國企業", "3033.HK": "南方恒生科技",
    "2318.HK": "中國平安", "0941.HK": "中國移動", "1211.HK": "比亞迪股份",
    "1398.HK": "工商銀行", "0939.HK": "建設銀行", "3988.HK": "中國銀行",
    "0001.HK": "長和", "0016.HK": "新鴻基地產", "0002.HK": "中電控股",
    "0003.HK": "香港中華煤氣", "0011.HK": "恒生銀行", "2628.HK": "中國人壽",
}
US_UNIVERSE = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
    "META": "Meta", "TSLA": "Tesla", "GOOGL": "Alphabet", "AMD": "AMD",
    "AVGO": "Broadcom", "QQQ": "Nasdaq 100 ETF", "SPY": "S&P 500 ETF",
    "LITE": "Lumentum",
    "QCOM": "Qualcomm", "ORCL": "Oracle", "MU": "Micron", "SNDK": "SanDisk",
    "VRT": "Vertiv", "SMCI": "Super Micro Computer", "NFLX": "Netflix",
    "IONQ": "IonQ", "RGTI": "Rigetti Computing", "ONDS": "Ondas Holdings",
    "PLTR": "Palantir", "HOOD": "Robinhood", "INTC": "Intel", "MRVL": "Marvell",
    "APP": "AppLovin", "MSTR": "Strategy", "COIN": "Coinbase",
}
DEFAULT_UNIVERSE = {**HK_UNIVERSE, **US_UNIVERSE}
MODEL_DIR = "models"
MODEL_FILE = f"{MODEL_DIR}/lgbm_fwd5_clf.joblib"
FEATURE_FILE = f"{MODEL_DIR}/feature_list.json"
METRICS_FILE = f"{MODEL_DIR}/metrics.json"