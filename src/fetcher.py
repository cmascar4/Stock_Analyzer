"""Download and cache OHLCV + fundamental data via yfinance."""
import time
import pandas as pd
import yfinance as yf
import config


def fetch_ticker(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download OHLCV history for one ticker. Returns empty DataFrame on failure."""
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, auto_adjust=True)
        if df.empty:
            print(f"  [WARN] No data for {ticker}")
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "Date"
        return df[["Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception as exc:
        print(f"  [ERROR] {ticker}: {exc}")
        return pd.DataFrame()


def fetch_all(
    tickers: list[str] | None = None,
    period: str = "5y",
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download data for all tickers with optional disk caching."""
    if tickers is None:
        tickers = config.TICKERS

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        cache_path = config.RAW_DATA_DIR / f"{ticker}.csv"
        if use_cache and cache_path.exists():
            df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
            data[ticker] = df
            print(f"  [CACHE] {ticker}: {len(df)} rows")
        else:
            print(f"  [FETCH] {ticker} ...")
            df = fetch_ticker(ticker, period)
            if not df.empty:
                df.to_csv(cache_path)
                data[ticker] = df
            time.sleep(0.4)  # polite rate-limiting

    return data


def fetch_fundamentals(ticker: str) -> dict:
    """Return key fundamental metrics from yfinance. Returns {} on failure."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "short_name": info.get("shortName", ticker),
            "sector": info.get("sector", "ETF / Fund"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "beta": info.get("beta"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"),
        }
    except Exception:
        return {}
