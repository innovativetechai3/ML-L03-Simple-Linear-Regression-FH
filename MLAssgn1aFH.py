"""
==============================
Assignment No. 1(a) [ML L03 - Simple Linear Regression]
==============================
Submitted By : FH
Date         : 22 June 2026
Dataset      : SalaryFHL3.csv (Kaggle)

ML Pipeline:
- Data loading
- EDA (Exploratory Data Analysis)
- Train-test split
- Model training (Linear Regression)
- Synthetic inference generation
- Model evaluation (R² score)
"""

# Import all required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# ******************* EDA *******************
# Load dataset & perform base structure of EDA
df=pd.read_csv("SalaryFHL3.csv")
print(df.head())
print(df.tail())
print(df.shape)
df.info()
print(df.describe())

print("Linear Regression using Scikit-Learn (Train-Test Split + Synthetic Inference)")

# Features and target
X = df[['YearsExperience']]
Y = df['Salary']

# Train-test split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Train Linear Regression model
lr = linear_model.LinearRegression()
lr.fit(X_train, Y_train)
intercept=lr.intercept_
coefficient=lr.coef_

# Generate synthetic (unseen) YearsExperience values
exp_value=[]
random_years = []
existing_values = set(df["YearsExperience"].round(2))

while len(random_years) < 10:
    value = round(
        np.random.uniform(
            df["YearsExperience"].min(),
            df["YearsExperience"].max()
        ),
        2
    )
    # Filtering duplicates
    if value not in existing_values and value not in random_years:
        random_years.append(value)
        # Manual prediction
        exp_value.append((coefficient*value)+intercept)

# Convert synthetic data into sklearn-compatible DataFrame
X_synthetic = pd.DataFrame(random_years, columns=['YearsExperience'])

# Predict for all values
predictions = lr.predict(X_synthetic)

# Results table
result_df = pd.DataFrame({
    "YearsExperience": random_years,
    "Predicted_Salary": predictions,
    "Expected_Salary": exp_value
})

print(result_df)
print(f"intercept={intercept} \ncoefficient={coefficient}")

# Model evaluation
print("R² score (test data):", lr.score(X_test, Y_test))

# ******************* Visualization of data *******************

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

# (1) Regression Plot
axes[0].scatter(df.YearsExperience, df.Salary, label="Actual Data")

sorted_df = df.sort_values("YearsExperience")

axes[0].plot(
    sorted_df["YearsExperience"],
    lr.predict(sorted_df[["YearsExperience"]]),
    color="red",
    label="Regression Line"
)

axes[0].set_xlabel("YearsExperience")
axes[0].set_ylabel("Salary")
axes[0].set_title("Regression Fit")
axes[0].legend()

# (2) Distribution Plot
sns.histplot(df["Salary"], kde=True, bins=50, ax=axes[1])
axes[1].set_title("Salary Distribution")

#  (3) Actual vs Predicted
axes[2].scatter(Y_test, lr.predict(X_test))
axes[2].set_xlabel("Actual Salary")
axes[2].set_ylabel("Predicted Salary")
axes[2].set_title("Actual vs Predicted")

plt.tight_layout()
plt.subplots_adjust(
    left=0.06,
    right=0.98,
    wspace=0.35,
    top=0.90
)

plt.savefig("regression_plotFH.png", dpi=300, bbox_inches="tight")
plt.show()