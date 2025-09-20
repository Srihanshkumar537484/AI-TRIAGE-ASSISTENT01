import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Safai kiya hua data load karein
# 'your_dataset.csv' ko apni file ke naam se badlein
df = pd.read_csv('cleaned_triage_data.csv')

# 2. Apne data ko features (X) aur target (y) mein alag karein
X = df['symptoms']
y = df['Disease']

# 3. Data ko training aur testing sets mein baantein
# Hum data ka 80% training ke liye aur 20% testing ke liye istemal karenge
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Text ko numbers mein badalna
# Model ko text samajh nahi aata, isliye hum use numbers mein badlenge
# Hum TF-IDF Vectorizer ka istemal karenge
vectorizer = TfidfVectorizer()

# Training data par TF-IDF fit karein aur badle hue data ko transform karein
X_train_vectorized = vectorizer.fit_transform(X_train)

# Testing data ko transform karein
# Note: yahan hum sirf transform karenge, fit nahi.
X_test_vectorized = vectorizer.transform(X_test)

# Ab hum data ke shapes print karke dekhenge
print(f"Training data shape (X_train): {X_train_vectorized.shape}")
print(f"Testing data shape (X_test): {X_test_vectorized.shape}")
print(f"Training labels shape (y_train): {y_train.shape}")
print(f"Testing labels shape (y_test): {y_test.shape}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 5. Model ko chunna
model = LogisticRegression()

# 6. Model ko training data par train karein
# Yahan par model seekhega ki kis symptom ke liye kya bimari hai
model.fit(X_train_vectorized, y_train)

# 7. Model ko test karna
# Ab hum model se testing data par prediction karwayenge
predictions = model.predict(X_test_vectorized)

# 8. Model ki performance dekhna
# Yahan hum dekhenge ki hamare model ne kitni sahi predictions ki hain
print("\n--- Model ki Performance ---")

# Accuracy score
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

import joblib

# Model aur vectorizer ko save karein
joblib.dump(model, 'triage_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

print("\nModel aur Vectorizer safalta-poorvak save kar liye gaye hain.")
print("Ab aapko 'triage_model.joblib' aur 'tfidf_vectorizer.joblib' files dikhengi.")