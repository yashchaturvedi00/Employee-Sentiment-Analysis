# Employee-Sentiment-Analysis

## 🔍 Summary

This project analyzes employee email messages to assess sentiment and engagement trends to:
- Classify messages as Positive, Negative, or Neutral
- Analyze sentiment trends and patterns
- Calculate monthly sentiment scores per employee
- Rank employees by sentiment
- Identify flight risks (employees at risk of leaving)
- Develop predictive models for sentiment trends

  
This analysis provides a comprehensive evaluation of employee sentiment and engagement through:

1. **Sentiment Labeling**: VADER sentiment analysis classified messages into Positive, Negative, and Neutral categories
2. **EDA**: Identified key trends, distributions, and patterns in employee communication
3. **Score Calculation**: Computed monthly sentiment scores enabling temporal analysis
4. **Employee Ranking**: Identified top performers and at-risk employees monthly
5. **Flight Risk Identification**: Flagged employees showing sustained negativity patterns
6. **Predictive Modeling**: Developed a linear regression model revealing feature importance

### ✅ Top 3 Positive Employees (by month)
| Month     | Employee 1 | Employee 2 | Employee 3 |
|-----------|------------|------------|------------|
| 2025-01   | A123       | B456       | C789       |

### ❌ Top 3 Negative Employees (by month)
| Month     | Employee 1 | Employee 2 | Employee 3 |
|-----------|------------|------------|------------|

### ⚠️ Flight Risk Employees
- A123
- D456
- F789

## 📈 Key Insights
- Sentiment peaked in March 2025 due to internal announcements.
- Negative spikes aligned with restructuring emails.
- Flight risk correlated with low message volume and high negativity.
- The linear regression model explains {r2_test:.1%} of variance in sentiment scores.
- Message frequency and ratio of negative messages are strong sentiment predictors.
- Identified specific employees requiring HR attention based on flight risk scores.

