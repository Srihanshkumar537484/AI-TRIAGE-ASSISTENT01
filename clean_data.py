import pandas as pd

# 1. Dataset ko load karein
# 'your_dataset.csv' ko apni file ke naam se badlein
df = pd.read_csv('patient_data.csv')

# 2. Khaali jagah (NaN) ko ek khali string (blank space) se bharein
df.fillna('', inplace=True)

# 3. Saare symptom columns ko combine karke ek naya 'symptoms' column banayein
# Hum 'Symptom_1' se 'Symptom_17' tak ke columns ko jodenge
df['symptoms'] = df.iloc[:, 1:].apply(lambda row: ' '.join(row), axis=1)

# 4. Ab naye 'symptoms' column ko saaf karein
# Faltu ke spaces hatayein aur sabhi ko chote aksharon (lowercase) mein badlein
df['symptoms'] = df['symptoms'].str.lower().str.strip()

# 5. Zaroori columns ko chunein aur baaki ko hata dein
# Ab hum sirf 'Disease' aur naye 'symptoms' column ko rakhenge
final_df = df[['Disease', 'symptoms']]

# 6. Safai ke baad data ka ek chota sa hissa dekhein
print("--- Data Safai ke Baad ---")
print(final_df.head())

# Safai ke baad data ki jaankari
print("\n--- Safai ke Baad Data ki Jaankari ---")
print(final_df.info())

# 7. Saaf kiye hue data ko ek nayi CSV file mein save karein
final_df.to_csv('cleaned_triage_data.csv', index=False)
print("\n'cleaned_triage_data.csv' file safalta-poorvak ban gayi hai!")