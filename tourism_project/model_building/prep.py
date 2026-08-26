
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("tourism_project/data/tourism.csv")

# Drop CustomerID since it is only an identifier
df.drop(columns=["CustomerID"], inplace=True)

# NOTE:
# Categorical columns are intentionally left as raw strings.
# The training pipeline will use OneHotEncoder.
# Streamlit will also send raw categorical values.
# Encoding them here would make training and serving use
# different representations and can cause prediction issues.

X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# Stratified split preserves purchase / non-purchase ratio
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save train/test splits
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)

ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print(f"Training rows: {Xtrain.shape[0]}")
print(f"Testing rows : {Xtest.shape[0]}")
print(f"Features     : {Xtrain.shape[1]}")
