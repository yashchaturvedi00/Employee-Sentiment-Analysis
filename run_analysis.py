#!/usr/bin/env python
"""
Employee Sentiment Analysis and Engagement Evaluation
Complete Analysis Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
sns.set_style("whitegrid")

print("=" * 80)
print("EMPLOYEE SENTIMENT ANALYSIS - COMPLETE PIPELINE")
print("=" * 80)

# ============================================================================
# LOAD AND PREPROCESS DATA
# ============================================================================

print("\n1. LOADING AND PREPROCESSING DATA")
print("-" * 80)

df = pd.read_excel("test.xlsx")

print(f"Dataset Overview:")
print(f"  - Shape: {df.shape}")
print(f"  - Columns: {df.columns.tolist()}")

# Rename columns
df = df.rename(columns={
    "from": "employee_id",
    "body": "message_text",
    "date": "timestamp",
    "Subject": "subject"
})

# Preprocess
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["message_text"] = df["message_text"].astype(str).fillna("")

print("✓ Data loaded and preprocessed successfully")

# ============================================================================
# TASK 1: SENTIMENT LABELING
# ============================================================================

print("\n2. SENTIMENT LABELING")
print("-" * 80)

analyzer = SentimentIntensityAnalyzer()

def label_sentiment(text):
    if not text.strip():
        return "Neutral"
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment_label"] = df["message_text"].apply(label_sentiment)
df["sentiment_score"] = df["sentiment_label"].map({"Positive": 1, "Neutral": 0, "Negative": -1})
df["msg_len"] = df["message_text"].str.len()
df["month"] = df["timestamp"].dt.to_period("M").astype(str)

print(f"Sentiment Distribution:")
print(df["sentiment_label"].value_counts())
print("✓ Sentiment labeling completed")

# ============================================================================
# TASK 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n3. EXPLORATORY DATA ANALYSIS")
print("-" * 80)

print(f"Total Records: {len(df)}")
print(f"Total Unique Employees: {df['employee_id'].nunique()}")
print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Create EDA visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(data=df, x="sentiment_label", ax=axes[0, 0], order=["Positive", "Neutral", "Negative"])
axes[0, 0].set_title("Distribution of Sentiment Labels", fontsize=12, fontweight="bold")

sentiment_dist = df["sentiment_label"].value_counts()
axes[0, 1].pie(sentiment_dist.values, labels=sentiment_dist.index, autopct="%1.1f%%", startangle=90)
axes[0, 1].set_title("Sentiment Distribution", fontsize=12, fontweight="bold")

monthly_sentiment = df.groupby("month")["sentiment_score"].mean()
axes[1, 0].plot(range(len(monthly_sentiment)), monthly_sentiment.values, marker="o", linewidth=2, markersize=6)
axes[1, 0].set_xticks(range(len(monthly_sentiment)))
axes[1, 0].set_xticklabels(monthly_sentiment.index, rotation=45, ha="right")
axes[1, 0].set_title("Average Monthly Sentiment Score", fontsize=12, fontweight="bold")
axes[1, 0].grid(True, alpha=0.3)

df.boxplot(column="msg_len", by="sentiment_label", ax=axes[1, 1])
axes[1, 1].set_title("Message Length Distribution by Sentiment", fontsize=12, fontweight="bold")
plt.suptitle("")

plt.tight_layout()
plt.savefig("eda_analysis.png", dpi=100, bbox_inches='tight')
print("✓ EDA visualizations saved to 'eda_analysis.png'")

# ============================================================================
# TASK 3: EMPLOYEE SCORE CALCULATION
# ============================================================================

print("\n4. EMPLOYEE SCORE CALCULATION")
print("-" * 80)

monthly_scores = (
    df.groupby(["employee_id", "month"])["sentiment_score"]
      .sum()
      .reset_index(name="monthly_sentiment_score")
)

monthly_counts = (
    df.groupby(["employee_id", "month"])["sentiment_label"]
      .value_counts()
      .unstack(fill_value=0)
      .reset_index()
)

if all(col in monthly_counts.columns for col in ["Positive", "Neutral", "Negative"]):
    monthly_scores = monthly_scores.merge(
        monthly_counts[["employee_id", "month", "Positive", "Neutral", "Negative"]],
        on=["employee_id", "month"],
        how="left"
    )

print(f"✓ Calculated sentiment scores for {monthly_scores['employee_id'].nunique()} employees")
print(f"✓ Across {monthly_scores['month'].nunique()} months")
print(f"\nSummary Statistics:")
print(monthly_scores["monthly_sentiment_score"].describe().round(2))

# ============================================================================
# TASK 4: EMPLOYEE RANKING
# ============================================================================

print("\n5. EMPLOYEE RANKING")
print("-" * 80)

def top3(group, ascending):
    return group.sort_values(
        ["monthly_sentiment_score", "employee_id"],
        ascending=[ascending, True]
    ).head(3)

rank_pos = monthly_scores.groupby("month", group_keys=False).apply(
    lambda g: top3(g, ascending=False)
).reset_index(drop=True)

rank_neg = monthly_scores.groupby("month", group_keys=False).apply(
    lambda g: top3(g, ascending=True)
).reset_index(drop=True)

print("\nTOP 3 POSITIVE EMPLOYEES (Last 3 Months):")
for month in sorted(rank_pos["month"].unique())[-3:]:
    month_data = rank_pos[rank_pos["month"] == month].reset_index(drop=True)
    print(f"\n{month}:")
    for idx, row in month_data.iterrows():
        print(f"  {idx+1}. {row['employee_id']}: Score = {row['monthly_sentiment_score']:+.0f}")

print("\n\nTOP 3 NEGATIVE EMPLOYEES (Last 3 Months):")
for month in sorted(rank_neg["month"].unique())[-3:]:
    month_data = rank_neg[rank_neg["month"] == month].reset_index(drop=True)
    print(f"\n{month}:")
    for idx, row in month_data.iterrows():
        print(f"  {idx+1}. {row['employee_id']}: Score = {row['monthly_sentiment_score']:+.0f}")

# Create ranking visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

latest_month = sorted(rank_pos["month"].unique())[-1]
pos_latest = rank_pos[rank_pos["month"] == latest_month].reset_index(drop=True)
neg_latest = rank_neg[rank_neg["month"] == latest_month].reset_index(drop=True)

if len(pos_latest) > 0:
    labels_pos = [f"#{i+1} {email.split('@')[0]}" for i, email in enumerate(pos_latest["employee_id"])]
    axes[0].barh(labels_pos, pos_latest["monthly_sentiment_score"], color=["green", "lightgreen", "lightgray"][:len(pos_latest)])
    axes[0].set_title(f"Top 3 Positive Employees ({latest_month})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Sentiment Score")

if len(neg_latest) > 0:
    labels_neg = [f"#{i+1} {email.split('@')[0]}" for i, email in enumerate(neg_latest["employee_id"])]
    axes[1].barh(labels_neg, neg_latest["monthly_sentiment_score"], color=["red", "lightcoral", "lightgray"][:len(neg_latest)])
    axes[1].set_title(f"Top 3 Negative Employees ({latest_month})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Sentiment Score")

plt.tight_layout()
plt.savefig("employee_ranking.png", dpi=100, bbox_inches='tight')
print("\n✓ Ranking visualizations saved to 'employee_ranking.png'")

# ============================================================================
# TASK 5: FLIGHT RISK IDENTIFICATION
# ============================================================================

print("\n6. FLIGHT RISK IDENTIFICATION")
print("-" * 80)

neg = df[df["sentiment_label"] == "Negative"].copy()
neg = neg.sort_values(["employee_id", "timestamp"])

def rolling_30d(group):
    group = group.set_index("timestamp").sort_index()
    group["neg_30d_count"] = group["sentiment_label"].rolling("30D").count()
    return group.reset_index()

neg_roll = neg.groupby("employee_id", group_keys=False).apply(rolling_30d)

flight_risk = (
    neg_roll.groupby("employee_id")["neg_30d_count"].max().reset_index()
    .rename(columns={"neg_30d_count": "max_30d_negative_count"})
)

flight_risk_threshold = 4
flight_risk["is_flight_risk"] = flight_risk["max_30d_negative_count"] >= flight_risk_threshold
flight_risk_ids = flight_risk[flight_risk["is_flight_risk"]]["employee_id"].tolist()

print(f"Flight Risk Threshold: ≥{flight_risk_threshold} negative messages in 30 days")
print(f"Number of Flight Risk Employees: {len(flight_risk_ids)}")

if flight_risk_ids:
    print(f"\nFlight Risk Employees:")
    risk_details = flight_risk[flight_risk["is_flight_risk"]].sort_values("max_30d_negative_count", ascending=False)
    for idx, row in risk_details.iterrows():
        print(f"  {row['employee_id']}: {row['max_30d_negative_count']:.0f} negative messages in 30-day window")

# Create flight risk visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].hist(flight_risk["max_30d_negative_count"].dropna(), bins=15, color="skyblue", edgecolor="black")
axes[0].axvline(x=flight_risk_threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold ({flight_risk_threshold})")
axes[0].set_xlabel("Max Negative Messages in 30-Day Window")
axes[0].set_ylabel("Number of Employees")
axes[0].set_title("Distribution of 30-Day Negative Message Counts", fontsize=12, fontweight="bold")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

risk_status = flight_risk["is_flight_risk"].value_counts()
labels = [f"Flight Risk\n({risk_status.get(True, 0)})", f"Normal\n({risk_status.get(False, 0)})"]
colors = ["red", "green"]
axes[1].pie(
    [risk_status.get(True, 0), risk_status.get(False, 0)],
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=90
)
axes[1].set_title("Employee Flight Risk Status", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("flight_risk_analysis.png", dpi=100, bbox_inches='tight')
print("✓ Flight risk visualizations saved to 'flight_risk_analysis.png'")

# ============================================================================
# TASK 6: PREDICTIVE MODELING
# ============================================================================

print("\n7. PREDICTIVE MODELING - LINEAR REGRESSION")
print("-" * 80)

features = df.groupby(["employee_id", "month"]).agg(
    message_count = ("message_text", "count"),
    avg_msg_len   = ("msg_len", "mean"),
    std_msg_len   = ("msg_len", "std"),
    pos_count     = ("sentiment_label", lambda x: (x=="Positive").sum()),
    neg_count     = ("sentiment_label", lambda x: (x=="Negative").sum()),
).reset_index()

features["pos_ratio"] = features["pos_count"] / features["message_count"]
features["neg_ratio"] = features["neg_count"] / features["message_count"]

data_model = features.merge(monthly_scores, on=["employee_id", "month"], how="left")

X = data_model[["message_count", "avg_msg_len", "std_msg_len", "pos_ratio", "neg_ratio"]].fillna(0)
y = data_model["monthly_sentiment_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)

print(f"\nFeature Engineering Summary:")
print(f"  - Total samples: {len(data_model)}")
print(f"  - Features used: message_count, avg_msg_len, std_msg_len, pos_ratio, neg_ratio")
print(f"  - Target variable: monthly_sentiment_score")

print(f"\nModel Performance Metrics:")
print(f"\nTraining Set:")
print(f"  - R² Score:  {r2_train:.4f}")
print(f"  - RMSE:      {rmse_train:.4f}")

print(f"\nTesting Set:")
print(f"  - R² Score:  {r2_test:.4f}")
print(f"  - RMSE:      {rmse_test:.4f}")
print(f"  - MAE:       {mae_test:.4f}")

coeffs = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": lr.coef_,
    "Abs_Coefficient": np.abs(lr.coef_)
}).sort_values("Abs_Coefficient", ascending=False)

print(f"\nFeature Coefficients & Interpretation:")
print(f"Intercept: {lr.intercept_:.4f}\n")
print(coeffs.to_string(index=False))

print("\nInterpretation:")
for idx, row in coeffs.iterrows():
    direction = "increases" if row["Coefficient"] > 0 else "decreases"
    print(f"  • {row['Feature']:15s}: {direction:10s} sentiment score by {row['Coefficient']:.4f} per unit increase")

# Create predictive model visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(y_test, y_test_pred, alpha=0.6, color="blue")
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
axes[0, 0].set_xlabel("Actual Sentiment Score")
axes[0, 0].set_ylabel("Predicted Sentiment Score")
axes[0, 0].set_title(f"Actual vs Predicted (Test Set, R²={r2_test:.4f})", fontsize=12, fontweight="bold")
axes[0, 0].grid(True, alpha=0.3)

residuals = y_test - y_test_pred
axes[0, 1].scatter(y_test_pred, residuals, alpha=0.6, color="green")
axes[0, 1].axhline(y=0, color="r", linestyle="--", lw=2)
axes[0, 1].set_xlabel("Predicted Sentiment Score")
axes[0, 1].set_ylabel("Residuals")
axes[0, 1].set_title("Residuals Plot", fontsize=12, fontweight="bold")
axes[0, 1].grid(True, alpha=0.3)

colors = ["green" if x > 0 else "red" for x in coeffs["Coefficient"]]
axes[1, 0].barh(coeffs["Feature"], coeffs["Coefficient"], color=colors)
axes[1, 0].set_xlabel("Coefficient Value")
axes[1, 0].set_title("Feature Coefficients", fontsize=12, fontweight="bold")
axes[1, 0].axvline(x=0, color="black", linestyle="-", linewidth=0.8)
axes[1, 0].grid(True, alpha=0.3, axis="x")

axes[1, 1].hist(residuals, bins=20, color="purple", edgecolor="black", alpha=0.7)
axes[1, 1].set_xlabel("Residuals")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_title("Distribution of Residuals", fontsize=12, fontweight="bold")
axes[1, 1].axvline(x=residuals.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {residuals.mean():.2f}")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("predictive_model.png", dpi=100, bbox_inches='tight')
print("\n✓ Predictive model visualizations saved to 'predictive_model.png'")

# ============================================================================
# SUMMARY & CONCLUSIONS
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print("\nKey Findings:")
print(f"  • Dataset contains {len(df)} employee messages from {df['employee_id'].nunique()} employees")
print(f"  • Sentiment Distribution: {df['sentiment_label'].value_counts()['Positive']} Positive, "
      f"{df['sentiment_label'].value_counts().get('Neutral', 0)} Neutral, "
      f"{df['sentiment_label'].value_counts().get('Negative', 0)} Negative")
print(f"  • Predictive Model explains {r2_test:.1%} of variance in sentiment scores")
print(f"  • {len(flight_risk_ids)} employees identified as flight risk")

print("\nGenerated Files:")
print("  ✓ eda_analysis.png - Exploratory Data Analysis visualizations")
print("  ✓ employee_ranking.png - Top/Bottom employee rankings")
print("  ✓ flight_risk_analysis.png - Flight risk identification visualizations")
print("  ✓ predictive_model.png - Predictive model performance visualizations")

print("\n✓ All analysis tasks completed successfully!")
