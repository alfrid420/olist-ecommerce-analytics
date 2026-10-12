"""Clean the raw orders file and add analysis-ready columns.

Input : ``data/olist_orders_dataset.csv``
Output: ``data/clean_orders.csv``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INPUT_PATH = DATA_DIR / "olist_orders_dataset.csv"
OUTPUT_PATH = DATA_DIR / "clean_orders.csv"

# Orders in these states never generated revenue.
INVALID_STATUSES = ("canceled", "unavailable")
# The synthetic data has no "estimated delivery date", so "late" is defined
# as delivery taking longer than this many days from purchase.
LATE_THRESHOLD_DAYS = 20


def load_raw(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load the raw CSV and parse the two timestamp columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/generate_data.py` first."
        )
    return pd.read_csv(
        path,
        parse_dates=["order_purchase_timestamp", "order_delivered_customer_date"],
    )


def clean_orders(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Return the cleaned frame and a log of (step, rows affected)."""
    log: list[tuple[str, int]] = []
    df = raw.copy()

    def drop_step(name: str, keep: pd.Series) -> None:
        """Keep rows where ``keep`` is True and record how many were removed."""
        nonlocal df
        log.append((name, int((~keep).sum())))
        df = df[keep].copy()

    # 1. Timestamps: coerce bad values to NaT. Rows without a purchase date
    #    cannot be placed on a timeline.
    for col in ("order_purchase_timestamp", "order_delivered_customer_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    drop_step("Missing purchase timestamp", df["order_purchase_timestamp"].notna())

    # 2. Exact duplicate orders (same order_id) -- keep the first occurrence.
    drop_step("Duplicate order_id", ~df["order_id"].duplicated(keep="first"))

    # 3. Cancelled / unavailable orders generated no revenue.
    drop_step(
        "Cancelled or unavailable", ~df["order_status"].isin(INVALID_STATUSES)
    )

    # 4. Revenue is the core metric: rows with no payment value are unusable.
    drop_step(
        "Missing or non-positive payment_value",
        df["payment_value"].notna() & (df["payment_value"] > 0),
    )

    # 5. Standardise text fields. Unknown categories are kept as "unknown"
    #    so that their revenue still counts toward totals.
    df["customer_state"] = df["customer_state"].str.strip().str.upper()
    df["product_category"] = df["product_category"].str.strip().fillna("unknown")

    # 6. Delivery days. Negative values are impossible -> treat as missing.
    delta = df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    df["delivery_days"] = delta.dt.total_seconds() / 86_400
    df.loc[df["delivery_days"] < 0, "delivery_days"] = np.nan

    # 7. Delivered orders missing the delivery date: impute with the median
    #    delivery time of the customer's state (flagged for auditing).
    #    Orders still in transit legitimately have no delivery date -> NaN.
    delivered = df["order_status"].eq("delivered")
    complete = delivered & df["delivery_days"].notna()
    state_median = (
        df.loc[complete].groupby("customer_state")["delivery_days"].median()
    )
    overall_median = df.loc[complete, "delivery_days"].median()
    needs_fill = delivered & df["delivery_days"].isna()
    fill_values = (
        df.loc[needs_fill, "customer_state"].map(state_median).fillna(overall_median)
    )
    df["delivery_date_imputed"] = needs_fill
    df.loc[needs_fill, "delivery_days"] = fill_values
    imputed_ts = df.loc[needs_fill, "order_purchase_timestamp"] + pd.to_timedelta(
        fill_values, unit="D"
    )
    # Round to whole seconds and match the column's datetime unit, otherwise
    # newer pandas versions refuse a lossy cast on assignment.
    df.loc[needs_fill, "order_delivered_customer_date"] = imputed_ts.dt.round(
        "s"
    ).astype(df["order_delivered_customer_date"].dtype)
    df["delivery_days"] = df["delivery_days"].round(2)
    log.append(("Delivery dates imputed (rows kept)", int(needs_fill.sum())))

    # 8. Derived columns.
    ts = df["order_purchase_timestamp"]
    df["order_month"] = ts.dt.strftime("%Y-%m")
    df["order_year"] = ts.dt.year
    late = np.where(
        df["delivery_days"].notna(),
        df["delivery_days"] > LATE_THRESHOLD_DAYS,
        np.nan,
    )
    # Nullable integer: 1/0 for delivered orders, <NA> when not yet delivered.
    df["is_late"] = pd.Series(late, index=df.index).astype("Int8")

    # 9. Recompute the repeat flag from the *cleaned* data so that every
    #    downstream number (EDA, SQL, dashboard) agrees with each other.
    n_orders = df.groupby("customer_id")["order_id"].transform("count")
    df["is_repeat_customer"] = n_orders > 1

    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    return df, log


def main() -> None:
    """Run the cleaning pipeline and print a before/after summary."""
    raw = load_raw()
    clean, log = clean_orders(raw)
    clean.to_csv(OUTPUT_PATH, index=False)

    print("=" * 56)
    print("DATA CLEANING SUMMARY")
    print("=" * 56)
    print(f"Rows before : {len(raw):>9,}")
    for step, count in log:
        print(f"  {step:<42}{count:>9,}")
    print(f"Rows after  : {len(clean):>9,}  ({len(clean) / len(raw):.1%} kept)")
    print(f"Columns     : {raw.shape[1]} -> {clean.shape[1]}")
    print(f"Saved       : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
