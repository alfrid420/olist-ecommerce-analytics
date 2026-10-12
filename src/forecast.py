"""Forecast monthly revenue for the next 3 months with Prophet.

Input : ``data/clean_orders.csv``
Output: ``outputs/forecast.png`` + predictions and a business interpretation
        printed to the console.

The helper functions are also imported by the Streamlit dashboard.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "clean_orders.csv"
PLOT_PATH = ROOT / "outputs" / "forecast.png"
FORECAST_MONTHS = 3
INTERVAL_WIDTH = 0.80  # 80% prediction interval


def build_monthly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate orders to monthly revenue in Prophet's ``ds``/``y`` format.

    Months with no orders (possible after dashboard filtering) are filled with
    zero so the series has no gaps.
    """
    month_start = df["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby(month_start)["payment_value"].sum()
    full_range = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_range, fill_value=0.0)
    return pd.DataFrame({"ds": monthly.index, "y": monthly.to_numpy()})


def _prophet_forecast(
    monthly: pd.DataFrame, periods: int, interval_width: float
) -> pd.DataFrame:
    """Fit Prophet and return history + future predictions."""
    from prophet import Prophet  # imported lazily: slow to load

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    # With only ~24 monthly points a full yearly Fourier seasonality would
    # overfit. Instead, model the November (Black Friday) spike as a "holiday"
    # so it does not distort the underlying growth trend.
    november = monthly.loc[monthly["ds"].dt.month == 11, "ds"]
    holidays = (
        pd.DataFrame({"holiday": "black_friday_month", "ds": november})
        if len(november)
        else None
    )
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=holidays,
        interval_width=interval_width,
        changepoint_prior_scale=0.1,
    )
    model.fit(monthly)
    future = model.make_future_dataframe(periods=periods, freq="MS")
    return model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def _linear_fallback(
    monthly: pd.DataFrame, periods: int, interval_width: float
) -> pd.DataFrame:
    """Simple linear-trend forecast used only if Prophet is not installed."""
    n = len(monthly)
    ordinary = monthly[monthly["ds"].dt.month != 11]  # ignore the holiday spike
    slope, intercept = np.polyfit(ordinary.index, ordinary["y"], deg=1)
    resid_sd = float(np.std(ordinary["y"] - (slope * ordinary.index + intercept)))
    z = 1.2816 if interval_width == 0.80 else 1.96  # 80% or 95% normal quantile
    idx = np.arange(n + periods)
    yhat = slope * idx + intercept
    ds = pd.date_range(monthly["ds"].iloc[0], periods=n + periods, freq="MS")
    return pd.DataFrame(
        {
            "ds": ds,
            "yhat": yhat,
            "yhat_lower": yhat - z * resid_sd,
            "yhat_upper": yhat + z * resid_sd,
        }
    )


def run_forecast(
    monthly: pd.DataFrame,
    periods: int = FORECAST_MONTHS,
    interval_width: float = INTERVAL_WIDTH,
) -> tuple[pd.DataFrame, str]:
    """Forecast ``periods`` months ahead.

    Returns the forecast frame (history + future, with an ``is_forecast``
    column) and the name of the method that produced it.
    """
    try:
        fc = _prophet_forecast(monthly, periods, interval_width)
        method = "Prophet"
    except Exception as exc:  # noqa: BLE001
        # Prophet missing (ImportError) or its Stan backend failing to load,
        # which happens on some Windows setups. Keep the pipeline working.
        fc = _linear_fallback(monthly, periods, interval_width)
        method = f"Linear trend (Prophet unavailable: {type(exc).__name__})"
    last_actual = monthly["ds"].max()
    fc["is_forecast"] = fc["ds"] > last_actual
    # Revenue cannot be negative.
    fc[["yhat", "yhat_lower", "yhat_upper"]] = fc[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].clip(lower=0)
    return fc, method


def plot_forecast(monthly: pd.DataFrame, fc: pd.DataFrame, path: Path = PLOT_PATH) -> Path:
    """Save a chart of actuals, fitted trend, forecast and interval."""
    path.parent.mkdir(parents=True, exist_ok=True)
    future = fc[fc["is_forecast"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly["ds"], monthly["y"] / 1e3, "o-", label="Actual revenue")
    ax.plot(fc["ds"], fc["yhat"] / 1e3, "--", color="tab:orange", label="Model fit")
    ax.plot(future["ds"], future["yhat"] / 1e3, "o", color="tab:red", label="Forecast")
    ax.fill_between(
        future["ds"],
        future["yhat_lower"] / 1e3,
        future["yhat_upper"] / 1e3,
        color="tab:red",
        alpha=0.25,
        label=f"{INTERVAL_WIDTH:.0%} interval",
    )
    ax.axvline(monthly["ds"].max(), color="grey", linestyle=":")
    ax.set_title(f"Monthly Revenue Forecast (next {len(future)} months)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (R$ thousands)")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def interpret(monthly: pd.DataFrame, fc: pd.DataFrame) -> str:
    """Write a short business interpretation from the numbers."""
    future = fc[fc["is_forecast"]]
    forecast_total = future["yhat"].sum()
    low, high = future["yhat_lower"].sum(), future["yhat_upper"].sum()
    recent_total = monthly["y"].tail(len(future)).sum()
    change = (forecast_total / recent_total - 1) * 100
    first, last = future["ds"].iloc[0], future["ds"].iloc[-1]
    if change < 0:
        context = (
            "A dip is expected: the last three actual months include the "
            "Black Friday and Christmas peak, which overstates the baseline."
        )
    else:
        context = (
            "The trend continues upward even after the holiday peak, "
            "which signals real underlying growth."
        )
    return (
        f"BUSINESS INTERPRETATION: Revenue for {first:%b %Y}-{last:%b %Y} is "
        f"forecast at about R${forecast_total:,.0f} (80% range "
        f"R${low:,.0f}-R${high:,.0f}), {change:+.0f}% versus the last "
        f"{len(future)} months. {context} Use the forecast as the planning "
        f"baseline for marketing budgets and the upper bound as a stretch "
        f"target. Growth is driven almost entirely by new customers (only "
        f"~3% ever reorder), so spend should be tied to acquisition cost and "
        f"paired with retention campaigns. This data is synthetic, so treat "
        f"the numbers as a demonstration of the method, not a real decision."
    )


def main() -> None:
    """Fit the model, save the plot, and print predictions."""
    if not CLEAN_PATH.exists():
        raise SystemExit(f"{CLEAN_PATH} not found. Run `python src/clean_data.py` first.")
    df = pd.read_csv(CLEAN_PATH, parse_dates=["order_purchase_timestamp"])
    monthly = build_monthly_series(df)
    fc, method = run_forecast(monthly)
    plot_path = plot_forecast(monthly, fc)

    print(f"Method: {method}")
    print(f"Plot saved: {plot_path}\n")
    print(f"Forecast for the next {FORECAST_MONTHS} months (80% interval):")
    table = fc[fc["is_forecast"]].copy()
    table["month"] = table["ds"].dt.strftime("%Y-%m")
    for col in ("yhat", "yhat_lower", "yhat_upper"):
        table[col] = table[col].map(lambda v: f"R${v:,.0f}")
    print(
        table[["month", "yhat", "yhat_lower", "yhat_upper"]]
        .rename(columns={"yhat": "forecast", "yhat_lower": "lower", "yhat_upper": "upper"})
        .to_string(index=False)
    )
    print("\n" + interpret(monthly, fc))


if __name__ == "__main__":
    main()
