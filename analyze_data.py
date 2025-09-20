import pandas as pd

# 'patient_data.csv' ko apni file ke naam se badlein
df = pd.read_csv('patient_data.csv')

# Baaki code jaisa hai, waise hi rehne dein
print("--- Pehli 5 Rows ---")
print(df.head())

print("\n--- Data ki Jaankari ---")
print(df.info())

print("\n--- Rows aur Columns ---")
print(f"Dataset mein {df.shape[0]} rows aur {df.shape[1]} columns hain.")