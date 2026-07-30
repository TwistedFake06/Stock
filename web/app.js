const state = {
  chart: null,
  volChart: null,
};

const ui = {
  form: document.getElementById("searchForm"),
  input: document.getElementById("symbolInput"),
  err: document.getElementById("errorText"),
  status: document.getElementById("statusText"),
  company: document.getElementById("companyText"),
  vPrice: document.getElementById("vPrice"),
  vChg: document.getElementById("vChg"),
  vHigh: document.getElementById("vHigh"),
  vLow: document.getElementById("vLow"),
  vRsi: document.getElementById("vRsi"),
  vMacd: document.getElementById("vMacd"),
  vSma20: document.getElementById("vSma20"),
  vSma50: document.getElementById("vSma50"),
  biasName: document.getElementById("biasName"),
  biasScore: document.getElementById("biasScore"),
  planBox: document.getElementById("planBox"),
};

function normalizeSymbol(raw) {
  const s = (raw || "").trim().toUpperCase();
  if (!s) return "";
  if (/^\d{6}$/.test(s)) {
    return s[0] === "6" ? `${s}.SS` : `${s}.SZ`;
  }
  return s;
}

function toStooqSymbol(normalized) {
  if (normalized.endsWith(".HK"))
    return normalized.replace(".HK", ".HK").toLowerCase();
  if (normalized.endsWith(".SS") || normalized.endsWith(".SZ"))
    return normalized.slice(0, 6).toLowerCase() + ".cn";
  if (normalized.includes(".")) return normalized.toLowerCase();
  return normalized.toLowerCase() + ".us";
}

function fmt(v, d = 2) {
  if (!Number.isFinite(v)) return "—";
  return v.toLocaleString(undefined, {
    maximumFractionDigits: d,
    minimumFractionDigits: d,
  });
}

function sma(series, n) {
  const out = Array(series.length).fill(null);
  let sum = 0;
  for (let i = 0; i < series.length; i += 1) {
    sum += series[i];
    if (i >= n) sum -= series[i - n];
    if (i >= n - 1) out[i] = sum / n;
  }
  return out;
}

function ema(series, n) {
  const out = Array(series.length).fill(null);
  const k = 2 / (n + 1);
  let prev = series[0];
  out[0] = prev;
  for (let i = 1; i < series.length; i += 1) {
    prev = series[i] * k + prev * (1 - k);
    out[i] = prev;
  }
  return out;
}

function rsi(series, n = 14) {
  if (series.length < n + 1) return Array(series.length).fill(null);
  const out = Array(series.length).fill(null);
  let gain = 0;
  let loss = 0;
  for (let i = 1; i <= n; i += 1) {
    const diff = series[i] - series[i - 1];
    if (diff >= 0) gain += diff;
    else loss += -diff;
  }
  gain /= n;
  loss /= n;
  out[n] = loss === 0 ? 100 : 100 - 100 / (1 + gain / loss);

  for (let i = n + 1; i < series.length; i += 1) {
    const diff = series[i] - series[i - 1];
    const g = diff > 0 ? diff : 0;
    const l = diff < 0 ? -diff : 0;
    gain = (gain * (n - 1) + g) / n;
    loss = (loss * (n - 1) + l) / n;
    out[i] = loss === 0 ? 100 : 100 - 100 / (1 + gain / loss);
  }
  return out;
}

function macd(series) {
  const e12 = ema(series, 12);
  const e26 = ema(series, 26);
  const line = series.map((_, i) =>
    e12[i] != null && e26[i] != null ? e12[i] - e26[i] : null,
  );
  const signal = ema(
    line.map((x) => x ?? 0),
    9,
  );
  const hist = line.map((x, i) => (x != null ? x - signal[i] : null));
  return { line, signal, hist };
}

function atr(rows, n = 14) {
  if (rows.length < n + 1) return null;
  const tr = [];
  for (let i = 1; i < rows.length; i += 1) {
    const h = rows[i].high;
    const l = rows[i].low;
    const pc = rows[i - 1].close;
    tr.push(Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc)));
  }
  let val = tr.slice(0, n).reduce((a, b) => a + b, 0) / n;
  for (let i = n; i < tr.length; i += 1) {
    val = (val * (n - 1) + tr[i]) / n;
  }
  return val;
}

function scoreBias(rows, closes, sma20, sma50, rsi14, macdHist) {
  const i = closes.length - 1;
  const close = closes[i];
  let score = 0;

  if (close > sma20[i]) score += 18;
  else score -= 18;

  if (sma20[i] > sma50[i]) score += 14;
  else score -= 14;

  if (macdHist[i] > 0) score += 16;
  else score -= 16;

  if (rsi14[i] >= 50 && rsi14[i] <= 70) score += 10;
  else if (rsi14[i] < 35) score -= 10;

  const mom5 = i >= 5 ? (close - closes[i - 5]) / closes[i - 5] : 0;
  if (mom5 > 0.02) score += 12;
  else if (mom5 < -0.02) score -= 12;

  const window = closes.slice(Math.max(0, i - 20), i);
  const hi20 = window.length ? Math.max(...window) : close;
  const lo20 = window.length ? Math.min(...window) : close;
  if (close > hi20) score += 12;
  if (close < lo20) score -= 12;

  score = Math.max(-100, Math.min(100, Math.round(score)));

  let bias = "Neutral";
  if (score >= 45) bias = "Strong Bullish";
  else if (score >= 18) bias = "Bullish";
  else if (score <= -45) bias = "Strong Bearish";
  else if (score <= -18) bias = "Bearish";

  return { score, bias };
}

function buildPlan(last, bias, score, atr14) {
  if (!Number.isFinite(last.close)) return "No plan data.";
  const atrV = Number.isFinite(atr14) ? atr14 : last.close * 0.02;

  if (score >= 0) {
    const buyLo = last.close - 0.6 * atrV;
    const buyHi = last.close - 0.2 * atrV;
    const stop = last.close - 1.4 * atrV;
    const t1 = last.close + 0.8 * atrV;
    const t2 = last.close + 1.8 * atrV;
    return `Bias: ${bias}\nBuy zone: ${fmt(buyLo)} - ${fmt(buyHi)}\nStop: ${fmt(stop)}\nTargets: T1 ${fmt(t1)}, T2 ${fmt(t2)}`;
  }

  const sellLo = last.close + 0.2 * atrV;
  const sellHi = last.close + 0.6 * atrV;
  const stop = last.close + 1.4 * atrV;
  const t1 = last.close - 0.8 * atrV;
  const t2 = last.close - 1.8 * atrV;
  return `Bias: ${bias}\nShort zone: ${fmt(sellLo)} - ${fmt(sellHi)}\nStop: ${fmt(stop)}\nTargets: T1 ${fmt(t1)}, T2 ${fmt(t2)}`;
}

function parseCsv(text) {
  const normalized = extractCsvBlock(text);
  const lines = normalized.trim().split(/\r?\n/);
  if (lines.length < 3) throw new Error("Not enough rows from provider.");
  const out = [];
  for (let i = 1; i < lines.length; i += 1) {
    const parts = lines[i].split(",");
    if (parts.length < 6) continue;
    const [date, open, high, low, close, volume] = parts;
    const row = {
      date,
      open: Number(open),
      high: Number(high),
      low: Number(low),
      close: Number(close),
      volume: Number(volume),
    };
    if (Number.isFinite(row.close)) out.push(row);
  }
  if (!out.length) throw new Error("Failed to parse CSV rows.");
  return out;
}

function extractCsvBlock(text) {
  if (!text) return "";
  const rows = text.split(/\r?\n/);
  const header = "Date,Open,High,Low,Close,Volume";
  const start = rows.findIndex((r) => r.trim() === header);
  if (start >= 0) return rows.slice(start).join("\n");
  return text;
}

async function fetchText(url, timeoutMs = 12000) {
  const ctl = new AbortController();
  const timer = window.setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      cache: "no-store",
      signal: ctl.signal,
      mode: "cors",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.text();
  } finally {
    window.clearTimeout(timer);
  }
}

function candidateUrls(stooqUrl) {
  const encoded = encodeURIComponent(stooqUrl);
  return [
    { name: "Stooq direct", url: stooqUrl },
    {
      name: "AllOrigins raw",
      url: `https://api.allorigins.win/raw?url=${encoded}`,
    },
    {
      name: "AllOrigins get",
      url: `https://api.allorigins.win/get?url=${encoded}`,
    },
    { name: "corsproxy.io", url: `https://corsproxy.io/?${encoded}` },
    {
      name: "r.jina.ai",
      url: `https://r.jina.ai/http://stooq.com/q/d/l/?s=${stooqUrl.split("s=")[1]}`,
    },
  ];
}

function parseAllOriginsGet(text) {
  try {
    const payload = JSON.parse(text);
    if (payload && typeof payload.contents === "string") {
      return payload.contents;
    }
  } catch (_err) {
    // Fall through when response is not JSON.
  }
  return text;
}

async function fetchHistory(symbol) {
  const stooq = toStooqSymbol(symbol);
  const stooqUrl = `https://stooq.com/q/d/l/?s=${stooq}&i=d`;
  const attempts = [];
  for (const source of candidateUrls(stooqUrl)) {
    try {
      let raw = await fetchText(source.url);
      if (source.name === "AllOrigins get") {
        raw = parseAllOriginsGet(raw);
      }
      const rows = parseCsv(raw).slice(-260);
      return { rows, source: source.name };
    } catch (err) {
      attempts.push(`${source.name}: ${err.message || "failed"}`);
    }
  }

  throw new Error(`All providers failed. ${attempts.join(" | ")}`);
}

function destroyCharts() {
  if (state.chart) state.chart.destroy();
  if (state.volChart) state.volChart.destroy();
}

function renderCharts(rows, sma20, sma50) {
  destroyCharts();
  const labels = rows.map((r) => r.date);
  const close = rows.map((r) => r.close);
  const volume = rows.map((r) => r.volume);

  const c1 = document.getElementById("priceChart").getContext("2d");
  state.chart = new Chart(c1, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Close",
          data: close,
          borderColor: "#0a7c86",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.15,
        },
        {
          label: "SMA20",
          data: sma20,
          borderColor: "#ee6c4d",
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.15,
        },
        {
          label: "SMA50",
          data: sma50,
          borderColor: "#7f56d9",
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.15,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { boxWidth: 10 } } },
      scales: {
        x: { display: false },
        y: { ticks: { maxTicksLimit: 6 } },
      },
    },
  });

  const c2 = document.getElementById("volChart").getContext("2d");
  state.volChart = new Chart(c2, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "Volume", data: volume, backgroundColor: "#95b8a6" }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { display: false }, y: { ticks: { maxTicksLimit: 5 } } },
    },
  });
}

function updateMetrics(rows, source) {
  const closes = rows.map((r) => r.close);
  const sma20 = sma(closes, 20);
  const sma50 = sma(closes, 50);
  const rsi14 = rsi(closes, 14);
  const m = macd(closes);
  const bias = scoreBias(rows, closes, sma20, sma50, rsi14, m.hist);
  const last = rows[rows.length - 1];
  const prev = rows[rows.length - 2] || last;
  const chg = last.close - prev.close;
  const chgPct = prev.close ? (chg / prev.close) * 100 : 0;
  const atr14 = atr(rows, 14);

  ui.status.textContent = `Loaded ${rows.length} daily bars`;
  ui.company.textContent = `${source} · last date ${last.date}`;
  ui.vPrice.textContent = fmt(last.close);
  ui.vChg.textContent = `${chg >= 0 ? "+" : ""}${fmt(chg)} (${chgPct >= 0 ? "+" : ""}${fmt(chgPct)}%)`;
  ui.vChg.className = `value ${chg >= 0 ? "up" : "down"}`;
  ui.vHigh.textContent = fmt(last.high);
  ui.vLow.textContent = fmt(last.low);
  ui.vRsi.textContent = fmt(rsi14[rsi14.length - 1]);
  ui.vMacd.textContent = fmt(m.hist[m.hist.length - 1], 4);
  ui.vSma20.textContent = fmt(sma20[sma20.length - 1]);
  ui.vSma50.textContent = fmt(sma50[sma50.length - 1]);

  ui.biasName.textContent = bias.bias;
  ui.biasScore.textContent = `Score ${bias.score}`;
  ui.planBox.textContent = buildPlan(last, bias.bias, bias.score, atr14);

  renderCharts(rows, sma20, sma50);
}

async function runQuery(raw) {
  ui.err.textContent = "";
  ui.status.textContent = "Loading market data...";
  const symbol = normalizeSymbol(raw);
  if (!symbol) {
    ui.status.textContent = "Enter ticker first.";
    return;
  }
  try {
    const { rows, source } = await fetchHistory(symbol);
    updateMetrics(rows, source);
  } catch (e) {
    ui.status.textContent = "Failed to fetch data.";
    ui.err.textContent = `Error: ${e.message}`;
  }
}

ui.form.addEventListener("submit", (e) => {
  e.preventDefault();
  runQuery(ui.input.value);
});

document.querySelectorAll("[data-symbol]").forEach((el) => {
  el.addEventListener("click", () => {
    ui.input.value = el.getAttribute("data-symbol") || "AAPL";
    runQuery(ui.input.value);
  });
});

runQuery("AAPL");
