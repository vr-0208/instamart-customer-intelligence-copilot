
-- Q1: Show the first 30 customers
SELECT *
FROM customers
ORDER BY customer_id
LIMIT 30;

-- Q2: Calculate completed-order revenue by city and product category
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS total_revenue
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
JOIN products AS p
    ON o.product_id = p.product_id
WHERE LOWER(o.status) = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- Q3: Find products priced above the average catalogue price
SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;


-- Q4: Find customers who have never placed a completed order
SELECT
    c.customer_id,
    c.city,
    c.age,
    c.gender
FROM customers AS c
WHERE NOT EXISTS (
    SELECT 1
    FROM orders AS o
    WHERE o.customer_id = c.customer_id
      AND LOWER(o.status) = 'completed'
)
ORDER BY c.customer_id;


-- Q5: Find customers whose completed-order spending is above average
WITH customer_spending AS (
    SELECT
        c.customer_id,
        c.city,
        ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM customers AS c
    JOIN orders AS o
        ON c.customer_id = o.customer_id
    WHERE LOWER(o.status) = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT
    customer_id,
    city,
    total_spend
FROM customer_spending
WHERE total_spend > (
    SELECT AVG(total_spend)
    FROM customer_spending
)
ORDER BY total_spend DESC;


-- Q6: Rank customers by completed-order spending within each city
WITH customer_spending AS (
    SELECT
        c.customer_id,
        c.city,
        ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM customers AS c
    JOIN orders AS o
        ON c.customer_id = o.customer_id
    WHERE LOWER(o.status) = 'completed'
    GROUP BY c.customer_id, c.city
),
ranked_customers AS (
    SELECT
        customer_id,
        city,
        total_spend,
        RANK() OVER (
            PARTITION BY city
            ORDER BY total_spend DESC
        ) AS spending_rank,
        ROW_NUMBER() OVER (
            PARTITION BY city
            ORDER BY total_spend DESC, customer_id
        ) AS row_num
    FROM customer_spending
)
SELECT
    customer_id,
    city,
    total_spend,
    spending_rank,
    row_num
FROM ranked_customers
ORDER BY city, spending_rank, customer_id;


-- Q7: Calculate daily completed-order revenue and cumulative revenue
WITH daily_revenue AS (
    SELECT
        DATE(order_ts) AS order_date,
        ROUND(SUM(order_amount), 2) AS daily_revenue
    FROM orders
    WHERE LOWER(status) = 'completed'
    GROUP BY DATE(order_ts)
)
SELECT
    order_date,
    daily_revenue,
    ROUND(
        SUM(daily_revenue) OVER (
            ORDER BY order_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        2
    ) AS cumulative_revenue
FROM daily_revenue
ORDER BY order_date;
