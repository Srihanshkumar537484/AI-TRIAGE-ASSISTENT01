# Import the pandas library
import pandas as pd

# Load the dataset from the CSV file
# Make sure the filename matches the one in your folder
df = pd.read_csv('dataset.csv')

# Print the first few rows to see what the data looks like
print("Original DataFrame:")
print(df.head())

# Print the column names to understand the data structure
print("\nColumn names:")
print(df.columns)