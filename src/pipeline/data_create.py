"""
Module: data_create.py
Author: Prachi Tavse, Tanisha Priya, Harshita Jangde
Date: 2025-09-08
Description:
    Generates synthetic Nova Partner dataset for machine learning experiments.
    The dataset includes demographics, earnings, trips, ratings, cancellations,
    payments, engagement metrics, and loans. The final dataset is saved as CSV.
"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 3000
zones = ["Urban North", "Urban South", "Rural East", "Rural West"]
job_types = ["Delivery Bike", "Delivery Car", "Delivery Van"]
genders = ["Male", "Female"]

data = []

for i in range(n_samples):
    # Demographics
    age = np.random.randint(20, 60)
    gender = np.random.choice(genders, p=[0.6, 0.4])
    zone = np.random.choice(zones, p=[0.4, 0.3, 0.15, 0.15])
    job_type = np.random.choice(job_types, p=[0.5, 0.4, 0.1])
    
    # Earnings & trips (structured correlation)
    base_earning = np.random.normal(12000, 3000)
    if zone.startswith("Urban"):
        base_earning *= 1.2
    if job_type == "Delivery Car":
        base_earning *= 1.1
    if job_type == "Delivery Van":
        base_earning *= 1.3
    
    earnings_weekly = max(5000, base_earning + np.random.normal(0, 2000))
    trip_count_monthly = int(earnings_weekly / 200 + np.random.normal(0, 10))
    
    # Ratings & cancellations
    cancellation_rate = np.clip(np.random.beta(2, 20), 0, 0.5)
    avg_rating = round(np.clip(4.0 + np.random.normal(0.5 - cancellation_rate, 0.2), 3, 5), 2)
    
    # Payments & reliability
    rent_paid_last_12 = np.random.randint(8, 13)
    utility_paid_last_12 = np.random.randint(9, 13)
    
    # Engagement
    app_logins = int(np.random.normal(20, 5))
    complaints = np.random.binomial(1, 0.1 if avg_rating > 4.5 else 0.3)
    
    # Loans
    outstanding_loans = max(0, np.random.normal(earnings_weekly * 0.2, 2000))
    if earnings_weekly < 8000:
        outstanding_loans *= 1.5
    
    data.append([
        f"GIG{i}",
        earnings_weekly,
        trip_count_monthly,
        avg_rating,
        cancellation_rate,
        zone,
        job_type,
        age,
        gender,
        rent_paid_last_12,
        utility_paid_last_12,
        app_logins,
        complaints,
        outstanding_loans
    ])

# Build dataframe
df = pd.DataFrame(data, columns=[
    "partner_id", "earnings_weekly", "trip_count_monthly", 
    "avg_rating_last_90_days", "cancellation_rate", "zone", 
    "job_type", "age", "gender", "rent_paid_last_12_months", 
    "utility_paid_last_12", "app_logins_last_30_days", 
    "complaints_last_6_months", "outstanding_loans"
])

# Save to CSV
df.to_csv("../../Data/Nova_Partner_Data.csv", index=False)

print("Dataset with 3000 rows saved as Nova_Partner_Data.csv")
print(df.head())