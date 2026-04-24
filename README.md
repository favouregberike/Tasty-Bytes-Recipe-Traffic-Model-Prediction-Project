# Tasty Bytes — Recipe Traffic Prediction Project

## Overview

This project builds a proof-of-concept predictive model for Tasty Bytes, a recipe content platform. The goal is to classify recipes as likely to generate high or low site traffic, enabling data-driven content and editorial decisions.

The analysis covers the full data science pipeline: data validation, exploratory analysis, statistical testing, model development, evaluation, and business recommendations.

---

## Business Objective

Tasty Bytes wants to predict which recipes will drive high traffic to their site and do so with at least **80% accuracy**. Achieving this would allow the team to:

- Prioritize high-potential recipes in content planning
- Reduce reliance on intuition in editorial decisions
- Improve return on content development investment

---

## Dataset

**File:** `recipe_site_traffic_2212.csv`

**Rows:** 947 recipes (895 after cleaning)

### Column Reference

| Column | Type | Description |
|---|---|---|
| `recipe` | Integer | Unique recipe identifier |
| `calories` | Float | Total caloric content |
| `carbohydrate` | Float | Carbohydrate content (grams) |
| `sugar` | Float | Sugar content (grams) |
| `protein` | Float | Protein content (grams) |
| `category` | Category | Recipe type (e.g., Breakfast, Dessert, Chicken) |
| `servings` | Integer | Number of servings the recipe yields |
| `high_traffic` | Boolean | Target variable: `True` = high traffic, `False` = low traffic |

---

## Project Structure

```
├── recipe_site_traffic_2212.csv   # Raw data
├── notebook.ipynb                 # Full analysis notebook
├── introduction.md                # Project introduction and problem framing
└── README.md                      # This file
```

---

## Methods

### Data Cleaning
- Removed 52 rows with missing nutritional values
- Standardized `servings` column (e.g., "4 as a snack" → 4)
- Converted `high_traffic` to Boolean (`"High"` → `True`, null → `False`)
- Cast `category` to `category` dtype and `servings` to `Int64`

### Exploratory Analysis
- Descriptive statistics for all numerical columns
- Distribution plots (histograms with KDE) for calories, carbohydrates, sugar, protein
- Bar charts for category, traffic label, and serving size distribution
- Scatterplot matrix of nutritional features
- Multi-variable plots comparing nutritional content by traffic status (boxplots, violin plots, pairplots)

### Statistical Testing
- **ANOVA** tests: calories, carbohydrates, and sugar showed statistically significant differences between high- and low-traffic recipes; protein did not
- **Chi-squared test**: confirmed a highly significant association between recipe category and traffic status (p ≈ 0)
- **Correlation matrix**: no strong multicollinearity between nutritional features

### Models Developed

| Model | Accuracy |
|---|---|
| Logistic Regression (baseline) | 64.8% |
| Logistic Regression + GridSearchCV | 65.6% |
| Logistic Regression + RandomizedSearchCV | 65.4% |
| LDA (baseline) | 62.6% |
| **LDA + GridSearchCV** | **68.3%** ✅ Best |
| LDA + RandomizedSearchCV | 59.9% |

### Best Model
**Linear Discriminant Analysis with GridSearchCV**
- Shrinkage: `auto`
- Solver: `lsqr`
- Cross-validated accuracy: **68.3%**

---

## Key Findings

- **Recipe category** is the strongest predictor of high traffic. Potato, Vegetable, and Pork categories consistently correlate with higher engagement; Beverages and Breakfast correlate negatively.
- **Calories and carbohydrates** are the most informative nutritional predictors, each showing statistically significant variation between traffic groups.
- **Protein** does not significantly differ between high- and low-traffic recipes and contributes less to model performance.
- **No model reached the 80% accuracy target**, suggesting that additional features, non-linear models, or larger training data may be needed.

---

## Recommendations

1. **Invest in high-performing categories**; Focus content development on Potato, Vegetable, and Pork recipes, which show the strongest association with high traffic.

2. **Deprioritize or reposition low-traffic categories**; Beverages and Breakfast recipes may require a different distribution or engagement strategy rather than organic promotion.

3. **Explore advanced models**; Gradient Boosting, Random Forest, or Neural Networks may capture non-linear relationships that LDA and Logistic Regression cannot, and could close the gap to the 80% target.

4. **Enrich the dataset** ; Incorporate user engagement signals (click-through rate, time on page, saves), seasonality data, and recipe complexity metrics to improve predictive power.

5. **Implement continuous retraining**; User preferences shift over time; the model should be periodically updated with new data to remain accurate.

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn missingno
```

**Python version:** 3.8+

---

## Usage

Open and run `notebook.ipynb` from top to bottom. All cleaning, analysis, modeling, and evaluation steps are documented inline with written commentary.
