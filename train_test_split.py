import pandas as pd
from sklearn.model_selection import train_test_split

# ======================
# LOAD DATASET
# ======================
df = pd.read_csv(
    "ai_student_impact_dataset (1).csv"
)

print("\nDataset Shape:")
print(df.shape)

# ======================
# INPUT AND OUTPUT
# ======================
target = "Burnout_Risk_Level"

X = df.drop(
    columns=[
        "Student_ID",
        target
    ]
)

y = df[target]

print("\nFeatures:")
print(X.columns)

print("\nTarget Classes:")
print(y.unique())

# ======================
# TRAIN TEST SPLIT
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================
# DISPLAY SHAPES
# ======================
print("\nTrain Shape")
print(X_train.shape)
print(y_train.shape)

print("\nTest Shape")
print(X_test.shape)
print(y_test.shape)

# ======================
# CLASS DISTRIBUTION
# ======================
print("\nOriginal Distribution")
print(y.value_counts())

print("\nTrain Distribution")
print(y_train.value_counts())

print("\nTest Distribution")
print(y_test.value_counts())

# ======================
# SPLIT PERCENTAGE
# ======================
print("\nTraining %")
print(
    len(X_train)/len(df)*100
)

print("\nTesting %")
print(
    len(X_test)/len(df)*100
)