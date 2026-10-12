"""Exploratory data analysis: six charts saved to ``outputs/`` + 5 insights.

Input : ``data/clean_orders.csv``
Output: PNG charts in ``outputs/`` and key insights printed to the console.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "clean_orders.csv"
OUTPUT_DIR = ROOT / "outputs"

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = sns.color_palette("deep")


def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    """Load the cleaned orders file."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/clean_data.py` first."
        )
    return pd.read_csv(
        path,
        parse_dates=["order_purchase_timestamp", "order_delivered_customer_date"],
    )


def _save(fig: plt.Figure, name: str) -> Path:
    """Save ``fig`` into the outputs folder and close it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_monthly_revenue(df: pd.DataFrame) -> Path:
    """Line chart of revenue per month with the peak month annotated."""
    monthly = df.groupby("order_month")["payment_value"].sum() / 1e3
    x = np.arange(len(monthly))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, monthly.values, marker="o", color=PALETTE[0], linewidth=2)
    peak = int(np.argmax(monthly.values))
    ax.annotate(
        f"Peak: {monthly.index[peak]}",
        xy=(peak, monthly.values[peak]),
        xytext=(peak - 4, monthly.values[peak] * 0.97),
        arrowprops={"arrowstyle": "->"},
    )
    ax.set_xticks(x[::2])
    ax.set_xticklabels(monthly.index[::2], rotation=45)
    ax.set_title("Monthly Revenue Trend")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (R$ thousands)")
    return _save(fig, "monthly_revenue_trend.png")


def plot_top_categories(df: pd.DataFrame) -> Path:
    """Horizontal bar chart of the 10 highest-revenue categories."""
    known = df[df["product_category"] != "unknown"]
    top = (
        known.groupby("product_category")["payment_value"]
        .sum()
        .nlargest(10)
        .sort_values()
        / 1e3
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index, top.values, color=PALETTE[1])
    ax.set_title("Top 10 Product Categories by Revenue")
    ax.set_xlabel("Revenue (R$ thousands)")
    return _save(fig, "top10_categories.png")


def plot_revenue_by_state(df: pd.DataFrame) -> Path:
    """Horizontal bar chart of revenue for every customer state."""
    by_state = (
        df.groupby("customer_state")["payment_value"].sum().sort_values() / 1e3
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(by_state.index, by_state.values, color=PALETTE[2])
    ax.set_title("Revenue by Customer State")
    ax.set_xlabel("Revenue (R$ thousands)")
    return _save(fig, "revenue_by_state.png")


def plot_repeat_vs_onetime(df: pd.DataFrame) -> Path:
    """Pie chart at the *customer* level (not order level)."""
    flags = df.groupby("customer_id")["is_repeat_customer"].first()
    counts = flags.value_counts().reindex([False, True], fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        counts.values,
        labels=["One-time", "Repeat"],
        autopct="%1.1f%%",
        startangle=90,
        explode=(0, 0.1),
        colors=[PALETTE[0], PALETTE[3]],
    )
    ax.set_title("Repeat vs One-Time Customers")
    return _save(fig, "repeat_vs_onetime.png")


def plot_review_distribution(df: pd.DataFrame) -> Path:
    """Bar chart of review-score counts with percentage labels."""
    scores = df["review_score"].dropna().astype(int)
    counts = scores.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index.astype(str), counts.values, color=PALETTE[4])
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count / counts.sum():.0%}",
            ha="center",
            va="bottom",
        )
    ax.set_title("Review Score Distribution")
    ax.set_xlabel("Review score")
    ax.set_ylabel("Orders")
    return _save(fig, "review_distribution.png")


def plot_delivery_vs_review(df: pd.DataFrame) -> Path:
    """Scatter of delivery days vs review score (sampled, jittered)."""
    data = df.dropna(subset=["delivery_days", "review_score"])
    sample = data.sample(n=min(5_000, len(data)), random_state=42)
    rng = np.random.default_rng(42)
    # Scores are integers, so add small vertical jitter to reveal density.
    jitter = rng.normal(0, 0.08, len(sample))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        sample["delivery_days"],
        sample["review_score"] + jitter,
        alpha=0.2,
        s=12,
        color=PALETTE[0],
    )
    # Overlay the average delivery time for each score.
    means = data.groupby("review_score")["delivery_days"].mean()
    ax.plot(means.values, means.index, "o-", color="crimson", label="Mean per score")
    ax.set_title("Delivery Time vs Review Score")
    ax.set_xlabel("Delivery time (days)")
    ax.set_ylabel("Review score")
    ax.legend()
    return _save(fig, "delivery_vs_review.png")


def compute_insights(df: pd.DataFrame) -> list[str]:
    """Return five headline insights, computed from the data."""
    total = df["payment_value"].sum()

    # 1. Growth + seasonality
    yearly = df.groupby("order_year")["payment_value"].sum()
    growth = (yearly.loc[2018] / yearly.loc[2017] - 1) * 100
    monthly = df.groupby("order_month")["payment_value"].sum()
    peak_month = monthly.idxmax()
    prev_month = monthly.shift(1).loc[peak_month]
    uplift = (monthly.max() / prev_month - 1) * 100

    # 2. Category concentration (exclude the "unknown" bucket)
    cats = (
        df[df["product_category"] != "unknown"]
        .groupby("product_category")["payment_value"]
        .sum()
        .sort_values(ascending=False)
    )
    top3_share = cats.head(3).sum() / cats.sum() * 100
    top3_names = ", ".join(cats.head(3).index)

    # 3. Geography
    states = (
        df.groupby("customer_state")["payment_value"]
        .sum()
        .sort_values(ascending=False)
    )
    sp_share = states.loc["SP"] / total * 100
    top3_states = states.head(3).sum() / total * 100

    # 4. Retention (customer level) and revenue contribution
    flags = df.groupby("customer_id")["is_repeat_customer"].first()
    repeat_rate = flags.mean() * 100
    repeat_rev_share = (
        df.loc[df["is_repeat_customer"], "payment_value"].sum() / total * 100
    )

    # 5. Delivery vs reviews
    late_rate = df["is_late"].mean() * 100  # <NA> (undelivered) is skipped
    on_time = df.loc[df["is_late"] == 0, "review_score"].mean()
    late = df.loc[df["is_late"] == 1, "review_score"].mean()
    corr = df["delivery_days"].corr(df["review_score"])

    return [
        f"Revenue grew {growth:.0f}% from 2017 (R${yearly.loc[2017]:,.0f}) to "
        f"2018 (R${yearly.loc[2018]:,.0f}); {peak_month} was the peak month, "
        f"{uplift:.0f}% above the month before (Black Friday).",
        f"Revenue is concentrated: the top 3 categories ({top3_names}) "
        f"generate {top3_share:.0f}% of known-category revenue.",
        f"Sao Paulo (SP) alone drives {sp_share:.0f}% of revenue; the top 3 "
        f"states account for {top3_states:.0f}%.",
        f"Only {repeat_rate:.1f}% of customers ever order again, and they "
        f"account for just {repeat_rev_share:.1f}% of revenue -- growth is "
        f"almost entirely new-customer acquisition.",
        f"{late_rate:.1f}% of deliveries take longer than the late threshold; "
        f"late orders average {late:.2f} stars vs {on_time:.2f} for on-time "
        f"(delivery days vs score correlation r = {corr:.2f}).",
    ]


def main() -> None:
    """Create all charts and print the key insights."""
    df = load_clean()
    charts = [
        plot_monthly_revenue(df),
        plot_top_categories(df),
        plot_revenue_by_state(df),
        plot_repeat_vs_onetime(df),
        plot_review_distribution(df),
        plot_delivery_vs_review(df),
    ]
    print(f"Saved {len(charts)} charts to {OUTPUT_DIR}:")
    for path in charts:
        print(f"  - {path.name}")

    print("\nKEY INSIGHTS")
    print("-" * 60)
    for i, text in enumerate(compute_insights(df), start=1):
        print(f"{i}. {text}")


if __name__ == "__main__":
    main()
