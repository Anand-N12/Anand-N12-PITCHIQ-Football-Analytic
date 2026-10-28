
import pandas as pd
import numpy as np

print("=" * 50)
print("PITCHIQ - DATA CLEANING PROCESS")
print("=" * 50)

# Load dataset
data = pd.read_csv("football_player_position_synthetic.csv")

# 1. Display dataset information
print("\n1. Dataset Information")
data.info()

# 2. Check dataset shape
print("\n2. Dataset Shape")
print(data.shape)

# 3. Check missing values
print("\n3. Missing Values")
print(data.isnull().sum())

# 4. Count total missing values
print("\n4. Total Missing Values")
print(data.isnull().sum().sum())

# 5. Check duplicate records
print("\n5. Duplicate Records")
print(data.duplicated().sum())

# 6. Remove duplicate records
data = data.drop_duplicates().copy()
print("\n6. Duplicates Removed")
print("Remaining rows:", len(data))

# 7. Handle missing numerical values
numeric_columns = data.select_dtypes(
    include="number"
).columns

data[numeric_columns] = data[numeric_columns].fillna(
    data[numeric_columns].median()
)
print("\n7. Numerical Missing Values Handled")

# 8. Handle missing categorical values
categorical_columns = data.select_dtypes(
    include=["object", "category", "string"]
).columns

for column in categorical_columns:
    if data[column].isnull().any():
        mode_values = data[column].mode()
        if not mode_values.empty:
            data[column] = data[column].fillna(
                mode_values.iloc[0]
            )

print("\n8. Categorical Missing Values Handled")

# 9. Check data types
print("\n9. Column Data Types")
print(data.dtypes)

# 10. Validate player positions
valid_positions = [
    "Defender",
    "Forward",
    "Goalkeeper",
    "Midfielder"
]

data["class"] = data["class"].astype("string").str.strip()

data = data[data["class"].isin(valid_positions)].copy()

print("\n10. Valid Player Positions")
print(data["class"].value_counts())

# 11. Check negative numerical values
numeric_columns = data.select_dtypes(
    include="number"
).columns

print("\n11. Negative Values Check")
print((data[numeric_columns] < 0).sum())

# 12. Save the cleaned dataset
data.to_csv(
    "football_player_position_cleaned.csv",
    index=False
)

print("\n12. Cleaned Dataset Saved")
print("Final shape:", data.shape)
print("File: football_player_position_cleaned.csv")

print("\nDATA CLEANING COMPLETED SUCCESSFULLY!")
