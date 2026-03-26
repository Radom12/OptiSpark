# Databricks notebook source

# COMMAND ----------
# MAGIC %md
# MAGIC # ⚡ OptiSpark v0.2.0 — Skewed Join Demo
# MAGIC This notebook creates a skewed DataFrame, shows the problem, and uses OptiSpark to fix it.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Cell 1: Install dependencies

# COMMAND ----------

# %pip install git+https://github.com/Radom12/OptiSpark.git --force-reinstall --no-cache-dir --quiet
# dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Cell 2: Create the Skewed Data

# COMMAND ----------

from pyspark.sql import functions as F

# ── Left table: 10M rows, 90% have the SAME join key ──
df_orders = (
    spark.range(0, 10_000_000, numPartitions=200)
    .withColumn(
        "customer_id",
        F.when(F.col("id") < 9_000_000, F.lit("CUST_HOT_KEY"))  # 90% skew!
         .otherwise(F.concat(F.lit("CUST_"), F.col("id").cast("string")))
    )
    .withColumn("order_amount", F.round(F.rand() * 500, 2))
    .withColumn("order_date", F.date_add(F.lit("2024-01-01"), (F.rand() * 365).cast("int")))
)

# ── Right table: 1M customers ──
df_customers = (
    spark.range(0, 1_000_000, numPartitions=50)
    .withColumn(
        "customer_id",
        F.when(F.col("id") < 500_000, F.lit("CUST_HOT_KEY"))  # matching hot key
         .otherwise(F.concat(F.lit("CUST_"), F.col("id").cast("string")))
    )
    .withColumn("customer_name", F.concat(F.lit("Customer_"), F.col("id").cast("string")))
    .withColumn("region", F.element_at(F.array(F.lit("US"), F.lit("EU"), F.lit("APAC")), (F.rand() * 3).cast("int") + 1))
)

print(f"Orders: {df_orders.count():,} rows")
print(f"Customers: {df_customers.count():,} rows")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Cell 3: The Problematic Skewed Join (DON'T fix this — this is the "before")

# COMMAND ----------

# This join will be EXTREMELY slow because 9M rows all have "CUST_HOT_KEY"
# One executor gets hammered while others sit idle
df_joined = df_orders.join(df_customers, "customer_id")

# Trigger execution and observe the skew in the Spark UI
# df_joined.count()  # <-- uncomment to actually run (will be slow!)

# Show the skew — most rows have the same key
print("── Data Distribution (top 10 keys) ──")
df_orders.groupBy("customer_id").count().orderBy(F.desc("count")).show(10, truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Cell 4: Use OptiSpark to analyze and chat about the fix

# COMMAND ----------

import os
from optispark import OptiSpark

# Set your Gemini API key (or use Databricks secrets)
# os.environ["GEMINI_API_KEY"] = dbutils.secrets.get(scope="your-scope", key="gemini-api-key")
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY_HERE"  # <-- replace this

agent = OptiSpark(api_key=os.environ["GEMINI_API_KEY"])

# Just pass the problematic DataFrame — the agent introspects it automatically!
agent.chat(df=df_joined)

# Inside the REPL, try asking:
#   "What bottlenecks do you see?"
#   "Fix the skew in my join"
#   "Show me an optimized version of this query"
#   /plan    — to see the Catalyst execution plan
#   /schema  — to see the DataFrame schema
#   /context — to see all extracted metadata

# COMMAND ----------
# MAGIC %md
# MAGIC ## Cell 5: The Manual Fix (what OptiSpark would suggest)
# MAGIC This is the salted join technique that resolves the hot-key skew.

# COMMAND ----------

SALT_BUCKETS = 20  # Number of salt partitions

# ── Step 1: Salt the skewed (large) table ──
df_orders_salted = (
    df_orders
    .withColumn("salt", (F.rand() * SALT_BUCKETS).cast("int"))
    .withColumn("salted_key", F.concat(F.col("customer_id"), F.lit("_"), F.col("salt")))
)

# ── Step 2: Replicate the small table across all salt buckets ──
df_customers_replicated = (
    df_customers
    .crossJoin(spark.range(0, SALT_BUCKETS).withColumnRenamed("id", "salt"))
    .withColumn("salted_key", F.concat(F.col("customer_id"), F.lit("_"), F.col("salt")))
)

# ── Step 3: Join on the salted key (now evenly distributed!) ──
df_fixed = (
    df_orders_salted
    .join(df_customers_replicated, "salted_key")
    .drop("salt", "salted_key")  # Clean up temporary columns
)

# ── Step 4: Verify the fix ──
print(f"Fixed join result: {df_fixed.count():,} rows")

# Compare partition distribution — should be much more even now
print("\n── Partition distribution (fixed) ──")
df_fixed.withColumn("part_id", F.spark_partition_id()).groupBy("part_id").count().describe().show()
