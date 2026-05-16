# Stock Analyzer

A Streamlit dashboard that pulls live market data, computes ~24 technical indicators, and trains an **XGBoost** next-day price predictor for a curated list of stocks and ETFs.

Built for personal research and learning — **not financial advice**.

---

## Features

- **Market Overview** — daily / 5-day / 1-month % change for every tracked ticker, sortable, color-coded
- **Stock Analysis** — candlestick chart with Bollinger Bands, SMA 20/50/200, volume, RSI, MACD
- **Price Forecast** — next-day close prediction from an XGBoost regressor trained on 24 features
- **Prediction Accuracy** — MAE / MAPE / RMSE / directional accuracy on a held-out test set, plus predicted-vs-actual chart
- **Train Models** — train predictors for any subset of tickers from the UI
- **Disk caching** — raw OHLCV is cached to `data/raw/` so reruns don't re-hit Yahoo Finance
- **Fundamentals** — sector, market cap, P/E, P/B, beta, dividend yield

The default ticker list (33 names) is in [config.py](config.py) and covers broad-market ETFs, mega-cap tech, fintech, and a few thematic picks. Edit `TICKERS` to track your own.

---

## Quick start (local)

### Prerequisites
- **Python 3.10+** (uses `X | None` union syntax)
- ~500 MB disk for cached data and trained models
- Internet access (for Yahoo Finance)

### Install

```bash
git clone https://github.com/<your-username>/Stock_Analyzer.git
cd Stock_Analyzer

python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r Stock_Analyzer/requirements.txt
```

### Run

From the **repo root** (the folder containing the `Stock_Analyzer/` package):

```bash
streamlit run Stock_Analyzer/dashboard/app.py
```

Streamlit will open at <http://localhost:8501>.

> First load downloads ~33 tickers from Yahoo Finance (about 30 seconds). Subsequent runs read from `data/raw/` and are instant.

### Train models

Open the **⚙️ Train Models** page in the sidebar, pick tickers + training period, click **Train**. Models are saved as `models/<TICKER>_predictor.pkl`. The **Price Forecast** and **Prediction Accuracy** pages will light up for any ticker that has a saved model.

---

## Project layout

```
Stock_Analyzer/                  ← repo root (clone target)
├── README.md
└── Stock_Analyzer/              ← Python package
    ├── config.py                ← tickers + hyperparameters
    ├── requirements.txt
    ├── dashboard/
    │   └── app.py               ← Streamlit entry point
    ├── src/
    │   ├── fetcher.py           ← yfinance download + CSV cache
    │   ├── features.py          ← 24 technical indicators
    │   └── predictor.py         ← XGBoost regressor wrapper
    ├── data/
    │   ├── raw/                 ← cached OHLCV CSVs (gitignored)
    │   └── processed/
    └── models/                  ← saved .pkl predictors (gitignored)
```

---

## Configuration

All knobs live in [config.py](config.py):

| Setting | Default | Meaning |
| --- | --- | --- |
| `TICKERS` | 33 symbols | What the app tracks |
| `TRAINING_PERIOD` | `"5y"` | History pulled for training |
| `DEFAULT_PERIOD` | `"2y"` | Default chart range in the dashboard |
| `TEST_SIZE` | `0.2` | Time-based train/test split (last 20% = test) |
| `N_ESTIMATORS` | 500 | XGBoost trees |
| `MAX_DEPTH` | 5 | XGBoost tree depth |
| `LEARNING_RATE` | 0.05 | XGBoost learning rate |

---

## Deployment

### Streamlit Community Cloud (recommended, free)

This is the simplest path and is designed for exactly this use case.

1. Push the repo to GitHub.
2. Go to <https://streamlit.io/cloud> and sign in with GitHub.
3. Click **New app**, pick this repo, and set:
   - **Branch:** `main`
   - **Main file path:** `Stock_Analyzer/dashboard/app.py`
   - **Python version:** 3.10 or newer
4. Under **Advanced settings → Requirements file**, point to `Stock_Analyzer/requirements.txt`.
5. Deploy. First boot trains nothing — users can train from the **⚙️ Train Models** page, but model files won't persist across container restarts on the free tier. For a permanent demo, commit a few pre-trained `.pkl` files (remove `*.pkl` from `.gitignore` for those specific files).

### Hugging Face Spaces (also free, supports Streamlit)

1. Create a new Space, choose **Streamlit** as the SDK.
2. Push the code; Spaces auto-detects `requirements.txt` and `app.py`.
3. You'll likely need to add a top-level `app.py` that imports from the package, or set the SDK entry point.

### Why not GitHub Pages?

**GitHub Pages only hosts static files** (HTML / CSS / JS). Streamlit is a Python web server — it needs a Python runtime to render anything, so it cannot run on GitHub Pages directly. If a GitHub-Pages-style static deploy is a hard requirement, the options are:

- **Rewrite as static** — port the dashboard to a JS framework (React/Vue) and pre-compute predictions, OR
- Use [`stlite`](https://github.com/whitphx/stlite) (Streamlit compiled to WebAssembly via Pyodide), which *can* be hosted on GitHub Pages. Caveat: yfinance + xgboost are heavy and may not all work in the browser sandbox.

For a real, working hosted version of this app, **Streamlit Community Cloud is the right answer**.

---

## How the model works

For each ticker the predictor:

1. Pulls daily OHLCV via `yfinance`.
2. Builds 24 features from technical indicators: multi-window returns, SMA/EMA ratios, RSI(14), MACD + histogram, Bollinger width/%B, ATR ratio, stochastic K/D, volume ratio, OBV ratio, 20-day realized volatility, and lagged 1-day returns at lags 1/2/3/5.
3. **Target:** next-day close price (regression).
4. **Split:** time-based — the last 20% of rows is held out as a test set to avoid look-ahead leakage.
5. **Model:** `XGBRegressor` (n_estimators=500, max_depth=5, lr=0.05, subsample=0.8, colsample_bytree=0.8).
6. Reports MAE / RMSE / MAPE / directional accuracy on the held-out portion.

The full pipeline is in [src/predictor.py](src/predictor.py) and [src/features.py](src/features.py).

---

## Known limitations

- **Mutual funds** like `FZROX` and `FBGRX` are Fidelity-only and may have no Yahoo Finance data — the dashboard handles this gracefully and shows a warning.
- The price predictor targets the next-day close; it is **not** designed for intraday signals.
- Yahoo Finance throttles aggressive querying. The fetcher already sleeps 0.4s between tickers; if you hit rate limits, wait a few minutes.
- Free Streamlit Cloud containers are ephemeral — trained `.pkl` files don't persist unless committed to the repo.

---

## Disclaimer

This project is for **educational and research purposes only**. Nothing here is investment advice. Past performance does not predict future returns. The models are simple by design and will be wrong — sometimes spectacularly. Do your own research and don't risk money you can't afford to lose.

---

## License

MIT — do what you want, just don't blame me when the market does its thing.
