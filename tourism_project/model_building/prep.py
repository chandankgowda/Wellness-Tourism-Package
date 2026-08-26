
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the tourism dataset
df = pd.read_csv(
    "tourism_project/data/tourism.csv"
)

# Drop CustomerID because it is an identifier
df.drop(
    columns=["CustomerID"],
    inplace=True
)

# ---------------------------------------------------------
# Normalize Gender
# ---------------------------------------------------------

df["Gender"] = (
    df["Gender"]
    .astype("string")
    .str.strip()
    .str.title()
)

# Correct known inconsistent Gender value
df["Gender"] = df["Gender"].replace({
    "Fe Male": "Female"
})

# ---------------------------------------------------------
# Normalize Occupation
# ---------------------------------------------------------

df["Occupation"] = (
    df["Occupation"]
    .astype("string")
    .str.strip()
    .str.title()
)

# ---------------------------------------------------------
# Normalize MaritalStatus
# ---------------------------------------------------------

df["MaritalStatus"] = (
    df["MaritalStatus"]
    .astype("string")
    .str.strip()
    .str.title()
)

# Combine Single and Unmarried
df["MaritalStatus"] = df["MaritalStatus"].replace({
    "Single": "Not Married",
    "Unmarried": "Not Married"
})

# ---------------------------------------------------------
# Create AgeGroup
# ---------------------------------------------------------

df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[
        0,
        25,
        35,
        45,
        55,
        65,
        float("inf")
    ],
    labels=[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56-65",
        "65+"
    ],
    include_lowest=True
)

# Convert category type to a normal string column
# This ensures the values are preserved correctly in CSV files.
df["AgeGroup"] = df["AgeGroup"].astype("string")

# ---------------------------------------------------------
# Define features and target
# ---------------------------------------------------------

X = df.drop(
    columns=["ProdTaken"]
)

y = df["ProdTaken"]

# ---------------------------------------------------------
# Train-test split
# ---------------------------------------------------------
# stratify=y preserves the ProdTaken class proportion.

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------------
# Save prepared datasets
# ---------------------------------------------------------

Xtrain.to_csv(
    "Xtrain.csv",
    index=False
)

Xtest.to_csv(
    "Xtest.csv",
    index=False
)

ytrain.to_csv(
    "ytrain.csv",
    index=False
)

ytest.to_csv(
    "ytest.csv",
    index=False
)

# ---------------------------------------------------------
# Validation output
# ---------------------------------------------------------

print(
    "Data prepared: train/test splits written."
)

print(
    "Training shape:",
    Xtrain.shape
)

print(
    "Testing shape:",
    Xtest.shape
)

print(
    "\nPrepared feature columns:"
)

print(
    Xtrain.columns.tolist()
)

print(
    "\nAgeGroup values:"
)

print(
    sorted(
        Xtrain["AgeGroup"]
        .dropna()
        .unique()
        .tolist()
    )
)

print(
    "\nMaritalStatus values:"
)

print(
    sorted(
        Xtrain["MaritalStatus"]
        .dropna()
        .unique()
        .tolist()
    )
)

# Required safety validation
assert "AgeGroup" in Xtrain.columns
assert "AgeGroup" in Xtest.columns

print(
    "\nAgeGroup validation completed successfully."
)
