-- =====================================================================
-- Olist E-Commerce Analytics: 12 business queries (SQLite dialect)
-- Table: orders (loaded from data/clean_orders.csv by src/load_sql.py)
-- Run all of them with:  python src/load_sql.py --run-queries
-- Note: every query ends with a semicolon, and comments avoid semicolons
-- so the runner can split this file safely.
-- =====================================================================


-- Q1: Monthly revenue for the last 12 months of data
-- Business question: what does the recent revenue run-rate look like?
-- The window is relative to the newest month in the table (not today's
-- date) so the query still works on a static historical dataset.
SELECT order_month,
       ROUND(SUM(payment_value), 2) AS revenue,
       COUNT(*)                     AS orders
FROM orders
WHERE order_month IN (
    SELECT DISTINCT order_month
    FROM orders
    ORDER BY order_month DESC
    LIMIT 12
)
GROUP BY order_month
ORDER BY order_month;


-- Q2: Top 10 categories by revenue
-- Business question: where should inventory investment go first?
-- The "unknown" bucket (missing category) is excluded from the ranking.
SELECT product_category,
       ROUND(SUM(payment_value), 2) AS revenue,
       COUNT(*)                     AS orders,
       ROUND(AVG(payment_value), 2) AS avg_order_value
FROM orders
WHERE product_category <> 'unknown'
GROUP BY product_category
ORDER BY revenue DESC
LIMIT 10;


-- Q3: Average order value (AOV) by state
-- Business question: do some regions buy bigger baskets?
-- States with fewer than 100 orders are dropped because their averages
-- are too noisy to act on.
SELECT customer_state,
       COUNT(*)                     AS orders,
       ROUND(AVG(payment_value), 2) AS avg_order_value
FROM orders
GROUP BY customer_state
HAVING COUNT(*) >= 100
ORDER BY avg_order_value DESC;


-- Q4: Customer repeat purchase rate
-- Business question: how many customers ever come back? (the retention gap)
-- Step 1 counts orders per customer, step 2 summarises that distribution.
WITH customer_orders AS (
    SELECT customer_id, COUNT(*) AS n_orders
    FROM orders
    GROUP BY customer_id
)
SELECT COUNT(*)                                            AS total_customers,
       SUM(CASE WHEN n_orders > 1 THEN 1 ELSE 0 END)       AS repeat_customers,
       ROUND(100.0 * SUM(CASE WHEN n_orders > 1 THEN 1 ELSE 0 END)
             / COUNT(*), 2)                                AS repeat_rate_pct
FROM customer_orders;


-- Q5: Month-over-month revenue growth %
-- Business question: how fast is revenue accelerating or slowing?
-- LAG() looks at the previous month's revenue (window function, needs
-- SQLite 3.25 or newer). The first month has no predecessor -> NULL.
WITH monthly AS (
    SELECT order_month, SUM(payment_value) AS revenue
    FROM orders
    GROUP BY order_month
)
SELECT order_month,
       ROUND(revenue, 2) AS revenue,
       ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY order_month))
             / LAG(revenue) OVER (ORDER BY order_month), 2) AS mom_growth_pct
FROM monthly
ORDER BY order_month;


-- Q6: Top 5 states by order count
-- Business question: where is demand concentrated?
SELECT customer_state,
       COUNT(*)                                           AS orders,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM orders), 2)
                                                          AS pct_of_orders
FROM orders
GROUP BY customer_state
ORDER BY orders DESC
LIMIT 5;


-- Q7: Average delivery time by state
-- Business question: which regions get the slowest service?
-- Only orders with a delivery time are averaged (in-transit orders are NULL).
SELECT customer_state,
       COUNT(delivery_days)             AS delivered_orders,
       ROUND(AVG(delivery_days), 1)     AS avg_delivery_days
FROM orders
WHERE delivery_days IS NOT NULL
GROUP BY customer_state
ORDER BY avg_delivery_days DESC;


-- Q8: Review score vs delivery time
-- Business question: does slow delivery hurt customer satisfaction?
-- Average review per delivery-time bucket, plus the squared Pearson
-- correlation (r_squared) computed from raw sums. SQLite has no CORR() or
-- guaranteed SQRT(), so r_squared is used and its sign is reported in
-- the "direction" column. (The signed r is printed by src/eda.py.)
WITH base AS (
    SELECT delivery_days AS x, review_score AS y
    FROM orders
    WHERE delivery_days IS NOT NULL AND review_score IS NOT NULL
),
sums AS (
    SELECT COUNT(*) * 1.0 AS n,
           SUM(x) AS sx, SUM(y) AS sy,
           SUM(x * y) AS sxy, SUM(x * x) AS sxx, SUM(y * y) AS syy
    FROM base
),
corr AS (
    SELECT (n * sxy - sx * sy) * (n * sxy - sx * sy)
           / ((n * sxx - sx * sx) * (n * syy - sy * sy)) AS r_squared,
           CASE WHEN n * sxy - sx * sy < 0 THEN 'negative' ELSE 'positive' END
               AS direction
    FROM sums
),
buckets AS (
    SELECT CASE WHEN x <= 7  THEN '1) 0-7 days'
                WHEN x <= 14 THEN '2) 8-14 days'
                WHEN x <= 21 THEN '3) 15-21 days'
                ELSE              '4) 22+ days' END AS delivery_bucket,
           COUNT(*)           AS orders,
           ROUND(AVG(y), 2)   AS avg_review_score
    FROM base
    GROUP BY delivery_bucket
)
SELECT b.delivery_bucket, b.orders, b.avg_review_score,
       ROUND(c.r_squared, 4) AS overall_r_squared,
       c.direction           AS correlation_direction
FROM buckets AS b
CROSS JOIN corr AS c
ORDER BY b.delivery_bucket;


-- Q9: Revenue contribution by category (% of total) with cumulative share
-- Business question: how concentrated is revenue? (Pareto view)
WITH cat AS (
    SELECT product_category, SUM(payment_value) AS revenue
    FROM orders
    GROUP BY product_category
)
SELECT product_category,
       ROUND(revenue, 2) AS revenue,
       ROUND(100.0 * revenue / SUM(revenue) OVER (), 2) AS pct_of_total,
       ROUND(100.0 * SUM(revenue) OVER (ORDER BY revenue DESC)
             / SUM(revenue) OVER (), 2) AS cumulative_pct
FROM cat
ORDER BY revenue DESC;


-- Q10: Loyal customers with 3 or more orders
-- Business question: who are the best candidates for a loyalty programme?
-- total_loyal_customers repeats on every row via a window function.
SELECT customer_id,
       COUNT(*)                         AS n_orders,
       ROUND(SUM(payment_value), 2)     AS lifetime_value,
       MIN(order_month)                 AS first_order_month,
       MAX(order_month)                 AS last_order_month,
       COUNT(*) OVER ()                 AS total_loyal_customers
FROM orders
GROUP BY customer_id
HAVING COUNT(*) >= 3
ORDER BY n_orders DESC, lifetime_value DESC
LIMIT 20;


-- Q11: Late delivery rate by category
-- Business question: which categories have the worst fulfilment problems?
-- is_late is NULL for undelivered orders so AVG ignores them.
-- Categories with fewer than 200 delivered orders are dropped.
SELECT product_category,
       COUNT(is_late)                   AS delivered_orders,
       ROUND(100.0 * AVG(is_late), 2)   AS late_rate_pct
FROM orders
WHERE product_category <> 'unknown'
GROUP BY product_category
HAVING COUNT(is_late) >= 200
ORDER BY late_rate_pct DESC;


-- Q12: Best performing month by revenue
-- Business question: when is the peak, and how big is it?
SELECT order_month,
       ROUND(SUM(payment_value), 2) AS revenue,
       COUNT(*)                     AS orders
FROM orders
GROUP BY order_month
ORDER BY revenue DESC
LIMIT 1;
