import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_bank_transactions_india(num_records=50000):  # Default 50k rows
    users = [f"User_{i}" for i in range(1, 1001)]  # 1000 users
    banks = ["HDFC Bank", "ICICI Bank", "SBI Bank", "Axis Bank", "Kotak Bank"]
    transaction_modes = ["IMPS", "NEFT", "RTGS", "UPI", "DEBIT_CARD", "CREDIT_CARD"]
    transaction_types = ["ONLINE", "CARD_SWIPE", "ATM_WITHDRAW", "UPI", "BANK_TRANSFER"]
    merchant_categories = ["ELECTRONICS", "GROCERY", "FASHION", "TRAVEL", "FOOD", "HEALTH"]
    cities = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata", "Ahmedabad"]
    device_types = ["Android", "iPhone", "Web", "ATM"]

    country = "India"  # Only India

    data = []

    for i in range(num_records):
        user = random.choice(users)
        bank_name = random.choice(banks)
        account_number = random.randint(1000000000, 9999999999)

        trans_type = random.choice(transaction_types)
        trans_mode = random.choice(transaction_modes)
        merchant = random.choice(merchant_categories)
        city = random.choice(cities)
        device = random.choice(device_types)

        # Transaction amount
        amount = round(np.random.exponential(scale=3000), 2)
        if amount > 100000:
            amount = round(random.uniform(30000, 120000), 2)

        # Balance
        balance_before = round(random.uniform(5000, 200000), 2)
        balance_after = round(balance_before - amount, 2)

        # Fraud probability (only domestic rules)
        fraud_probability = 0.01
        if amount > 50000:
            fraud_probability += 0.15
        if device in ["Web", "ATM"]:
            fraud_probability += 0.05

        # Suspicious score for ML
        suspicious_score = round(fraud_probability * random.uniform(0.5, 1.5), 3)

        # Fraud label
        is_fraud = 1 if random.random() < fraud_probability else 0

        # Timestamp
        timestamp = datetime.now() - timedelta(days=random.randint(0, 365))

        # Transaction ID
        transaction_id = f"TXN{1000000 + i}"

        # Fake IP and GPS
        ip_address = ".".join(str(random.randint(1, 255)) for _ in range(4))
        latitude = round(random.uniform(10.0, 28.0), 6)
        longitude = round(random.uniform(70.0, 90.0), 6)

        data.append([
            transaction_id, user, bank_name, account_number,
            amount, balance_before, balance_after,
            trans_type, trans_mode, merchant,
            country, city, 0, device,  # is_international = 0
            ip_address, latitude, longitude,
            timestamp, suspicious_score, is_fraud
        ])

    df = pd.DataFrame(data, columns=[
        "transaction_id", "user_id", "bank_name", "account_number",
        "amount", "balance_before", "balance_after",
        "transaction_type", "transaction_mode", "merchant_category",
        "country", "city", "is_international", "device_type",
        "ip_address", "latitude", "longitude",
        "timestamp", "suspicious_score", "is_fraud"
    ])

    return df

# Generate dataset
df = generate_bank_transactions_india(50000)  # 50k rows
df.to_csv("synthetic_bank_transactions_india.csv", index=False)
print("CSV file created: synthetic_bank_transactions_india.csv")
print(df.head())
