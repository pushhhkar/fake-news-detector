import pandas as pd

# Load fake and real news
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = "FAKE"
true["label"] = "REAL"

# Combine both
df = pd.concat([fake, true], axis=0)

# Save as train.csv
df.to_csv("train.csv", index=False)

print("train.csv created successfully!")