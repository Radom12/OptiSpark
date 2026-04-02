# Databricks notebook source

# COMMAND ----------
# MAGIC %md
# MAGIC # ⚡ OptiSpark v0.4.0 — The "Nightmare" Bot Skew & Exploding Join PoC
# MAGIC This notebook creates a highly complex, multi-dimensional bottleneck designed to completely
# MAGIC crash standard PySpark jobs or force gigabytes of disk spill.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Cell 1: Install OptiSpark from GitHub

# COMMAND ----------

# %pip install git+https://github.com/Radom12/OptiSpark.git --force-reinstall --no-cache-dir --quiet
# dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Cell 2: Generate the "Data Engineering Nightmare" Dataset

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window

# ── 1. The Ad Clickstream (10M rows) ──
# 15% of traffic is "BOT" traffic with a NULL user_id
# 80% of clicks belong to ONE massive campaign: 'BLACK_FRIDAY'
df_clicks = (
    spark.range(0, 10_000_000, numPartitions=200)
    .withColumn(
        "user_id",
        F.when(F.col("id") % 100 < 15, F.lit(None).cast("string"))  # 15% Nulls (Bot traffic)
         .otherwise(F.concat(F.lit("USER_"), (F.col("id") % 500_000).cast("string"))) # 500k unique users
    )
    .withColumn(
        "campaign_id",
        F.when(F.col("id") < 8_000_000, F.lit("BLACK_FRIDAY")) # 80% of data in one partition
         .otherwise(F.concat(F.lit("CAMP_"), (F.col("id") % 100).cast("string")))
    )
    .withColumn("click_timestamp", F.current_timestamp() - F.expr("INTERVAL 1 DAY * rand() * 30"))
)

# ── 2. The Transactions (1M rows) ──
# Same users, but power users have dozens of purchases
df_purchases = (
    spark.range(0, 1_000_000, numPartitions=50)
    .withColumn(
        "user_id",
        F.when(F.col("id") % 100 < 5, F.lit(None).cast("string"))  # 5% Nulls in purchases
         .otherwise(F.concat(F.lit("USER_"), (F.col("id") % 100_000).cast("string"))) # 100k buyers
    )
    .withColumn("purchase_amount", F.round(F.rand() * 1000, 2))
    .withColumn("purchase_timestamp", F.current_timestamp() - F.expr("INTERVAL 1 DAY * rand() * 10"))
)

print(f"Clickstream: {df_clicks.count():,} rows")
print(f"Purchases: {df_purchases.count():,} rows")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Cell 3: The Fatal Pipeline (Do NOT run this in production without fixing!)
# MAGIC 
# MAGIC ### Bottlenecks created:
# MAGIC 1. **Null Skew**: Joining on `user_id` throws all Nulls to ONE executor core.
# MAGIC 2. **Exploding Join**: Power users (and bots) match across millions of clicks/purchases, amplifying rows.
# MAGIC 3. **Spill/OOM**: The Window partition over `campaign_id` captures 8M 'BLACK_FRIDAY' records in one partition.

# COMMAND ----------

# Attempt to find the "latest click before purchase" for ROI calculation
bad_pipeline_df = (
    df_clicks.join(df_purchases, "user_id", "inner")
    
    # Explodes because users clicked 100 times and bought 10 times = 1000 combinations per user
    
    # Now try to rank them within campaigns (will spill massively because of BLACK_FRIDAY)
    .withColumn(
        "rn", 
        F.row_number().over(
            Window.partitionBy("campaign_id").orderBy(F.desc("purchase_amount"))
        )
    )
    .filter(F.col("rn") == 1)
)

# Uncomment to watch your cluster burn 🔥
# bad_pipeline_df.count()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Cell 4: Launch OptiSpark (v0.4.0) to auto-diagnose and benchmark!

# COMMAND ----------

import os
from dotenv import load_dotenv
from optispark import OptiSpark

# Load environment variables (e.g. GEMINI_API_KEY from .env file)
load_dotenv()

agent = OptiSpark(api_key=os.environ.get("GEMINI_API_KEY"))

# Pass the fatal pipeline to the agent
agent.chat(df=bad_pipeline_df)

# Inside the chat, try:
#   1. "Fix these bottlenecks"
#   2. When it suggests code, type 'y' to safely auto-execute the fix.
#   3. Type /benchmark to run a 1% sampled Dry-Run comparing your crashy code to the AI fix!
