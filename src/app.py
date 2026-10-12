"""Streamlit dashboard for the Olist e-commerce analytics project.

Run with:  streamlit run src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make ``forecast.py`` importable no matter where Streamlit is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from forecast import build_monthly_series, run_forecast  # noqa: E402

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "clean_orders.csv"
MIN_MONTHS_FOR_FORECAST = 12


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    """Read the cleaned orders (cached across reruns)."""
    return pd.read_csv(
        path,
        parse_dates=["order_purchase_timestamp", "order_delivered_customer_date"],
    )


@st.cache_data(show_spinner="Fitting forecast model...")
def cached_forecast(monthly: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Run the forecast once per distinct filtered series."""
    return run_forecast(monthly)


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render the sidebar widgets and return the filtered orders."""
    st.sidebar.header("Filters")
    min_d = df["order_purchase_timestamp"].min().date()
    max_d = df["order_purchase_timestamp"].max().date()
    date_range = st.sidebar.date_input(
        "Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d
    )
    categories = sorted(df["product_category"].unique())
    states = sorted(df["customer_state"].unique())
    sel_cats = st.sidebar.multiselect("Product category", categories, default=categories)
    sel_states = st.sidebar.multiselect("Customer state", states, default=states)

    # date_input returns a 1-tuple while the user is still picking the end date.
    if not isinstance(date_range, tuple) or len(date_range) != 2:
        st.info("Select both a start and an end date in the sidebar.")
        st.stop()
    start, end = (pd.Timestamp(d) for d in date_range)

    mask = (
        (df["order_purchase_timestamp"] >= start)
        & (df["order_purchase_timestamp"] < end + pd.Timedelta(days=1))
        & df["product_category"].isin(sel_cats)
        & df["customer_state"].isin(sel_states)
    )
    filtered = df[mask]
    if filtered.empty:
        st.warning("No orders match the current filters.")
        st.stop()
    return filtered


def kpi_row(df: pd.DataFrame) -> None:
    """Show the four headline KPI cards."""
    revenue = df["payment_value"].sum()
    orders = len(df)
    # Customer-level repeat rate: share of unique customers flagged as repeat.
    repeat_rate = df.groupby("customer_id")["is_repeat_customer"].first().mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"R$ {revenue:,.0f}")
    c2.metric("Total Orders", f"{orders:,}")
    c3.metric("Avg Order Value", f"R$ {revenue / orders:,.2f}")
    c4.metric("Repeat Customer Rate", f"{repeat_rate:.1%}")


def revenue_chart(monthly: pd.DataFrame) -> None:
    """Monthly revenue line chart."""
    fig = px.line(
        monthly, x="ds", y="y", markers=True,
        labels={"ds": "Month", "y": "Revenue (R$)"},
    )
    fig.update_layout(margin={"t": 10})
    st.plotly_chart(fig, use_container_width=True)


def category_chart(df: pd.DataFrame) -> None:
    """Top 10 categories (excluding the 'unknown' bucket) by revenue."""
    top = (
        df[df["product_category"] != "unknown"]
        .groupby("product_category", as_index=False)["payment_value"]
        .sum()
        .nlargest(10, "payment_value")
        .sort_values("payment_value")
    )
    fig = px.bar(
        top, x="payment_value", y="product_category", orientation="h",
        labels={"payment_value": "Revenue (R$)", "product_category": ""},
    )
    fig.update_layout(margin={"t": 10})
    st.plotly_chart(fig, use_container_width=True)


def forecast_chart(monthly: pd.DataFrame) -> None:
    """Actuals plus a 3-month forecast with its confidence band."""
    if len(monthly) < MIN_MONTHS_FOR_FORECAST:
        st.info(
            f"Need at least {MIN_MONTHS_FOR_FORECAST} months of data to forecast "
            f"(current selection has {len(monthly)})."
        )
        return
    fc, method = cached_forecast(monthly)
    future = fc[fc["is_forecast"]]
    fig = go.Figure()
    # Upper bound first, then lower bound filled up to it = shaded band.
    fig.add_trace(go.Scatter(x=future["ds"], y=future["yhat_upper"], mode="lines",
                             line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future["ds"], y=future["yhat_lower"], mode="lines",
                             line={"width": 0}, fill="tonexty",
                             fillcolor="rgba(214,39,40,0.2)", name="80% interval"))
    fig.add_trace(go.Scatter(x=monthly["ds"], y=monthly["y"], mode="lines+markers",
                             name="Actual", line={"color": "#1f77b4"}))
    fig.add_trace(go.Scatter(x=future["ds"], y=future["yhat"], mode="lines+markers",
                             name="Forecast", line={"dash": "dash", "color": "#d62728"}))
    fig.update_layout(margin={"t": 10}, yaxis_title="Revenue (R$)", xaxis_title="Month")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Method: {method}. Forecast uses the filtered monthly series, so partial "
        "months at the edges of the date range can distort it."
    )


def main() -> None:
    """Assemble the dashboard page."""
    st.set_page_config(page_title="Olist E-Commerce Analytics", page_icon="📦", layout="wide")
    st.title("📦 Olist E-Commerce Analytics & Forecasting")
    st.markdown(
        "Which product categories deserve inventory investment, and what will "
        "revenue look like over the next 3 months?"
    )

    if not DATA_PATH.exists():
        st.error(
            "`data/clean_orders.csv` not found. Run `python src/generate_data.py` "
            "then `python src/clean_data.py` first."
        )
        st.stop()

    filtered = sidebar_filters(load_data(str(DATA_PATH)))
    kpi_row(filtered)
    st.divider()

    monthly = build_monthly_series(filtered)
    left, right = st.columns(2)
    with left:
        st.subheader("Monthly revenue")
        revenue_chart(monthly)
    with right:
        st.subheader("Top 10 categories by revenue")
        category_chart(filtered)

    st.subheader("3-month revenue forecast")
    forecast_chart(monthly)

    st.divider()
    st.caption(
        "Data: synthetic dataset inspired by the Olist Brazilian E-Commerce "
        "Public Dataset (CC BY-NC-SA 4.0). No real customer data is used."
    )


if __name__ == "__main__":
    main()
