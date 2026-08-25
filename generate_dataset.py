"""
Customer Segmentation - Dataset Generator
Generates a realistic synthetic customer dataset
"""

import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

N = 300

customer_ids = [f"CUST_{str(i).zfill(4)}" for i in range(1, N + 1)]

ages = np.concatenate([
    np.random.normal(25, 4, 60),   # Young adults
    np.random.normal(35, 5, 90),   # Middle-aged
    np.random.normal(50, 7, 90),   # Mature
    np.random.normal(65, 6, 60),   # Seniors
])
ages = np.clip(ages, 18, 80).astype(int)
np.random.shuffle(ages)

genders = np.random.choice(['Male', 'Female'], size=N, p=[0.48, 0.52])

# Annual Income in thousands (INR lakhs equivalent)
incomes = np.concatenate([
    np.random.normal(25, 5, 75),   # Low income
    np.random.normal(55, 8, 90),   # Middle income
    np.random.normal(85, 10, 90),  # Upper-middle income
    np.random.normal(120, 15, 45), # High income
])
incomes = np.clip(incomes, 10, 200).astype(int)
np.random.shuffle(incomes)

# Spending Score (1-100) - correlated somewhat with income
spending_scores = []
for inc in incomes:
    if inc < 40:
        score = np.random.normal(30, 12)
    elif inc < 70:
        score = np.random.normal(50, 15)
    elif inc < 100:
        score = np.random.normal(65, 12)
    else:
        score = np.random.normal(80, 10)
    spending_scores.append(score)
spending_scores = np.clip(spending_scores, 1, 100).astype(int)

# Purchase Frequency (per year)
purchase_freq = []
for ss in spending_scores:
    freq = ss / 10 + np.random.normal(0, 2)
    purchase_freq.append(max(1, int(freq)))

# Membership Years
membership_years = np.random.choice(range(0, 15), size=N,
                                     p=[0.1, 0.12, 0.1, 0.1, 0.09, 0.08,
                                        0.07, 0.07, 0.06, 0.05, 0.04,
                                        0.04, 0.03, 0.03, 0.02])

df = pd.DataFrame({
    'Customer_ID': customer_ids,
    'Age': ages,
    'Gender': genders,
    'Annual_Income_k': incomes,
    'Spending_Score': spending_scores,
    'Purchase_Frequency': purchase_freq,
    'Membership_Years': membership_years
})

df.to_csv('customer_data.csv', index=False)
print(f"✅ Dataset generated: {N} customers saved to 'customer_data.csv'")
print(df.head(10).to_string())
print(f"\nShape: {df.shape}")
print(f"\nBasic Statistics:")
print(df.describe())
