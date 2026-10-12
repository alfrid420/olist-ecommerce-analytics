"""Generate a synthetic e-commerce orders dataset inspired by Olist.

The real Olist Brazilian E-Commerce dataset needs a Kaggle login, so this
script builds a statistically plausible stand-in with the same business
tension: strong revenue growth but very few repeat customers (~3%).

Output: ``data/olist_orders_dataset.csv`` (100,000 rows, seed 42).
All values are synthetic -- no real customer data is involved.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 100_000
DUPLICATE_SHARE = 0.005  # exact duplicate rows injected to make cleaning real
REPEAT_CUSTOMER_RATE = 0.03
FIRST_MONTH, LAST_MONTH = "2017-01-01", "2018-12-01"
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "olist_orders_dataset.csv"
)

# category -> (sales weight, median basket value in R$, log-sigma)
CATEGORIES = {
    "bed_bath_table": (0.100, 90, 0.7),
    "health_beauty": (0.090, 110, 0.7),
    "sports_leisure": (0.080, 100, 0.7),
    "furniture_decor": (0.070, 120, 0.8),
    "computers_accessories": (0.070, 130, 0.8),
    "housewares": (0.070, 90, 0.7),
    "watches_gifts": (0.060, 190, 0.7),
    "telephony": (0.050, 60, 0.7),
    "garden_tools": (0.050, 100, 0.7),
    "auto": (0.050, 120, 0.8),
    "toys": (0.050, 85, 0.7),
    "cool_stuff": (0.040, 140, 0.7),
    "perfumery": (0.040, 80, 0.6),
    "baby": (0.040, 95, 0.7),
    "electronics": (0.030, 75, 0.8),
    "stationery": (0.030, 55, 0.6),
    "fashion_bags_accessories": (0.030, 100, 0.7),
    "pet_shop": (0.025, 90, 0.6),
    "small_appliances": (0.020, 150, 0.8),
    "musical_instruments": (0.015, 210, 0.8),
}

# state -> (customer weight, typical delivery days). SP is the logistics hub.
STATES = {
    "SP": (0.420, 8), "RJ": (0.130, 11), "MG": (0.120, 11),
    "RS": (0.055, 14), "PR": (0.050, 12), "SC": (0.037, 13),
    "BA": (0.034, 18), "DF": (0.022, 13), "GO": (0.020, 14),
    "ES": (0.020, 13), "PE": (0.016, 20), "CE": (0.013, 21),
    "PA": (0.010, 23), "MT": (0.009, 17), "MA": (0.007, 22),
    "MS": (0.007, 16), "PB": (0.005, 20), "RN": (0.005, 20),
    "PI": (0.005, 21), "AL": (0.004, 21), "SE": (0.003, 21),
    "TO": (0.003, 20), "RO": (0.0025, 22), "AM": (0.0015, 27),
    "AC": (0.001, 27), "AP": (0.0007, 26), "RR": (0.0005, 28),
}

# Calendar-month demand multipliers (key 1 = January). November is the
# Black Friday spike, December the Christmas tail.
SEASONALITY = {
    1: 1.00, 2: 0.95, 3: 1.05, 4: 1.00, 5: 1.08, 6: 0.98,
    7: 1.00, 8: 1.02, 9: 0.90, 10: 0.95, 11: 1.60, 12: 1.12,
}

ORDER_STATUS = {
    "delivered": 0.968, "shipped": 0.012, "canceled": 0.010,
    "unavailable": 0.003, "processing": 0.004, "invoiced": 0.003,
}

# Relative order volume by hour of day (quiet overnight, busy evenings)
_HOUR_WEIGHTS = np.array(
    [1, 1, 1, 1, 1, 1.5, 2, 3, 4, 5, 6, 6,
     6, 6, 6, 6, 6, 6, 7, 7, 7, 6, 4, 2]
)
HOUR_P = _HOUR_WEIGHTS / _HOUR_WEIGHTS.sum()


def _normalise(values) -> np.ndarray:
    """Return ``values`` scaled to sum to 1 (probability vector)."""
    arr = np.asarray(list(values), dtype=float)
    return arr / arr.sum()


def _hex_ids(rng: np.random.Generator, n: int) -> np.ndarray:
    """Create ``n`` random 32-character hex identifiers (Olist-style)."""
    high = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64)
    low = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64)
    return np.array([f"{a:016x}{b:016x}" for a, b in zip(high, low)])


def _monthly_weights(months: pd.DatetimeIndex) -> np.ndarray:
    """Order-volume weight per month: linear growth trend x seasonality."""
    trend = 1 + 0.11 * np.arange(len(months))
    seasonal = np.array([SEASONALITY[m] for m in months.month])
    return _normalise(trend * seasonal)


def _assign_customers(rng: np.random.Generator, n_orders: int) -> np.ndarray:
    """Return a customer index per order so ~3% of customers order again.

    97% of customers place exactly one order. Of the 3% repeat customers,
    85% place two orders and 15% place three.
    """
    n_customers = int(n_orders / (1 + REPEAT_CUSTOMER_RATE * 1.15)) + 500
    while True:
        is_repeat = rng.random(n_customers) < REPEAT_CUSTOMER_RATE
        extra = 1 + (rng.random(n_customers) < 0.15)
        sizes = 1 + is_repeat * extra
        if sizes.sum() >= n_orders:
            break
        n_customers += 1000  # rare: draw a few more customers
    # Repeat each customer index by its order count, trim to exact size.
    return np.repeat(np.arange(n_customers), sizes)[:n_orders]


def generate_orders(seed: int = SEED, n_rows: int = N_ROWS) -> pd.DataFrame:
    """Build the synthetic orders table (including injected data issues)."""
    rng = np.random.default_rng(seed)
    n_dupes = int(n_rows * DUPLICATE_SHARE)
    n = n_rows - n_dupes

    # --- Purchase timestamps with growth + holiday seasonality -------------
    months = pd.date_range(FIRST_MONTH, LAST_MONTH, freq="MS")
    month_idx = rng.choice(len(months), size=n, p=_monthly_weights(months))
    days_in_month = months.days_in_month.to_numpy()[month_idx]
    day = (rng.random(n) * days_in_month).astype(int)
    hour = rng.choice(24, size=n, p=HOUR_P)
    seconds = day * 86_400 + hour * 3_600 + rng.integers(0, 3_600, size=n)
    purchase = (
        months.to_numpy().astype("datetime64[s]")[month_idx]
        + seconds.astype("timedelta64[s]")
    )
    month_num = months.month.to_numpy()[month_idx]

    # --- Customers: persistent id + state; repeat buyers get >1 order ------
    cust_idx = _assign_customers(rng, n)
    n_cust = int(cust_idx.max()) + 1
    cust_ids = _hex_ids(rng, n_cust)
    cust_state = rng.choice(
        list(STATES), size=n_cust, p=_normalise(v[0] for v in STATES.values())
    )
    is_repeat = (np.bincount(cust_idx, minlength=n_cust) > 1)[cust_idx]
    state = cust_state[cust_idx]

    # --- Category, price, status -------------------------------------------
    category = rng.choice(
        list(CATEGORIES),
        size=n,
        p=_normalise(v[0] for v in CATEGORIES.values()),
    )
    median = np.array([CATEGORIES[c][1] for c in category], dtype=float)
    sigma = np.array([CATEGORIES[c][2] for c in category], dtype=float)
    basket = median * np.exp(rng.normal(0, 1, n) * sigma)
    freight = rng.gamma(2.0, 6.0, n)
    payment = np.round(np.clip(basket + freight, 5, None), 2)
    status = rng.choice(
        list(ORDER_STATUS), size=n, p=_normalise(ORDER_STATUS.values())
    )

    # --- Delivery time: state-driven base + occasional delays --------------
    base_days = np.array([STATES[s][1] for s in state], dtype=float)
    delivery_days = base_days * rng.lognormal(0, 0.30, n)
    delay_prob = np.where(np.isin(month_num, (11, 12)), 0.15, 0.07)  # peak load
    delayed = rng.random(n) < delay_prob
    delivery_days = delivery_days + delayed * rng.integers(5, 25, size=n)
    delivery_seconds = np.round(delivery_days * 86_400).astype("int64")
    delivered_ts = purchase + delivery_seconds.astype("timedelta64[s]")

    # --- Review score: worse when delivery is slow --------------------------
    latent = 4.55 - 0.07 * (delivery_days - 8) + rng.normal(0, 0.95, n)
    review = pd.Series(np.clip(np.rint(latent), 1, 5)).astype("Int8")

    delivered = status == "delivered"
    # Only delivered orders have a delivery date and a review.
    delivered_ts = pd.Series(delivered_ts).where(delivered)

    # --- Inject realistic missing values (~5% overall) ----------------------
    no_delivery_date = delivered & (rng.random(n) < 0.02)
    delivered_ts = delivered_ts.mask(no_delivery_date)
    review = review.mask(~delivered | (rng.random(n) < 0.05))
    category = pd.Series(category).mask(rng.random(n) < 0.03)
    payment = pd.Series(payment).mask(rng.random(n) < 0.005)

    df = pd.DataFrame(
        {
            "order_id": _hex_ids(rng, n),
            "customer_id": cust_ids[cust_idx],
            "order_status": status,
            "order_purchase_timestamp": purchase,
            "order_delivered_customer_date": delivered_ts,
            "payment_value": payment,
            "product_category": category,
            "customer_state": state,
            "review_score": review,
            "is_repeat_customer": is_repeat,
        }
    )

    # --- Inject exact duplicate rows ----------------------------------------
    dupes = df.sample(n=n_dupes, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)
    df = df.sort_values("order_purchase_timestamp", kind="stable")
    return df.reset_index(drop=True)


def main() -> None:
    """Generate the dataset, save it, and print a short summary."""
    df = generate_orders()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d %H:%M:%S")

    unique = df.drop_duplicates("order_id")
    cust = unique.groupby("customer_id")["is_repeat_customer"].first()
    print(f"Saved {len(df):,} rows -> {OUTPUT_PATH}")
    print(
        f"Date range : {df['order_purchase_timestamp'].min():%Y-%m-%d} to "
        f"{df['order_purchase_timestamp'].max():%Y-%m-%d}"
    )
    print(f"Customers  : {len(cust):,} ({cust.mean():.1%} repeat)")
    print("Missing values (% of rows):")
    print((df.isna().mean() * 100).round(2)[lambda s: s > 0].to_string())


if __name__ == "__main__":
    main()
