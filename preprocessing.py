# 3_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("ai_student_impact_dataset (1).csv")

# =========================
# SEPARATE INPUT AND OUTPUT
# =========================
X = df.drop(columns=['Burnout_Risk_Level'])
y = df['Burnout_Risk_Level']

# =========================
# TRAIN-TEST SPLIT
# =========================cd

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# FIND NUMERICAL COLUMNS
# =========================
num_cols = list(
    X_train.select_dtypes(include=['int64', 'float64']).columns
)

print("Numerical Columns:")
print(num_cols)

# =========================
# FIND CATEGORICAL COLUMNS
# =========================
cat_cols = list(
    X_train.select_dtypes(include=['object']).columns
)

print("\nCategorical Columns:")
print(cat_cols)

# =========================
# NUMERICAL PIPELINE..preprocessing pipeline
# =========================
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# =========================
# CATEGORICAL PIPELINE
# =========================
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# =========================
# COLUMN TRANSFORMER,,combine
# =========================
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# =========================
# APPLY PREPROCESSING
# =========================
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# =========================
# DISPLAY SHAPES
# =========================
print("\nShape after preprocessing:")
print("X_train_processed:", X_train_processed.shape)
print("X_test_processed :", X_test_processed.shape)

print("\nPreprocessing completed successfully!")