import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("ai_student_impact_dataset (1).csv")

# =========================
# BASIC INFORMATION
# =========================
print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Last 5 rows ---")
print(df.tail())

print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# =========================
# NUMERICAL COLUMNS
# =========================
numerical_columns = df.select_dtypes(include=['int64', 'float64'])

print("\nNumerical Columns:")
print(numerical_columns.columns)

# =========================
# COLUMN ANALYSIS
# =========================
for col in df.columns:

    print("\n========================")
    print("Column:", col)

    # Numerical Columns
    if df[col].dtype in ['int64', 'float64']:

        mean = df[col].mean()
        std = df[col].std()
        minimum = df[col].min()
        maximum = df[col].max()

        q1 = df[col].quantile(0.25)
        median = df[col].quantile(0.50)
        q3 = df[col].quantile(0.75)

        skew = df[col].skew()

        if abs(skew) < 0.5:
            distribution = "Normal Distribution"
        elif skew > 1:
            distribution = "Highly Right Skewed"
        elif skew > 0.5:
            distribution = "Right Skewed"
        elif skew < -1:
            distribution = "Highly Left Skewed"
        else:
            distribution = "Left Skewed"

        print("Distribution:", distribution)
        print("Mean:", mean)
        print("Standard Deviation:", std)
        print("Minimum:", minimum)
        print("Maximum:", maximum)
        print("25%:", q1)
        print("Median:", median)
        print("75%:", q3)

        # Histogram
        plt.figure(figsize=(6,4))
        plt.hist(df[col], bins=20, edgecolor='black')
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

    # Categorical Columns
    else:

        print("Categorical Column")

        print(df[col].value_counts())

        plt.figure(figsize=(6,4))
        df[col].value_counts().plot(kind='bar')
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


# ===================================================
# CORRELATION ANALYSIS
# ===================================================
print("\n========================")
print("CORRELATION ANALYSIS")
print("========================")

# Select numerical columns only
numerical_df = df.select_dtypes(include=['int64', 'float64'])

# Correlation matrix
corr_matrix = numerical_df.corr()

print("\nCorrelation Matrix")
print(corr_matrix)

# ===================================================
# HEATMAP
# ===================================================
plt.figure(figsize=(12,8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.show()

# ===================================================
# HIGHLY CORRELATED FEATURES
# ===================================================
print("\nHighly Correlated Features")

threshold = 0.7

for i in range(len(corr_matrix.columns)):
    for j in range(i):

        correlation_value = corr_matrix.iloc[i, j]

        if abs(correlation_value) > threshold:

            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]

            print(
                f"{col1} and {col2} --> Correlation = {correlation_value:.2f}"
            )

# ===================================================
# CORRELATION WITH TARGET VARIABLE
# ===================================================
print("\n========================")
print("CORRELATION WITH TARGET")
print("========================")

# Create copy
df_corr = df.copy()

# Convert target labels into numbers
df_corr["Burnout_Risk_Level"] = df_corr["Burnout_Risk_Level"].map({
    "Low": 0,
    "Medium": 1,
    "High": 2
})

# Select numerical columns
numerical_target = df_corr.select_dtypes(include=['int64', 'float64'])

# Correlation matrix
corr_target = numerical_target.corr()

# Correlation with target variable
target_corr = corr_target["Burnout_Risk_Level"].sort_values(
    ascending=False
)

print("\nCorrelation with Burnout_Risk_Level")
print(target_corr)

# ===================================================
# BAR PLOT FOR TARGET CORRELATION
# ===================================================
plt.figure(figsize=(10,6))

target_corr.drop("Burnout_Risk_Level").plot(
    kind='bar',
    color='skyblue'
)

plt.title("Correlation with Burnout Risk Level")
plt.xlabel("Features")
plt.ylabel("Correlation Coefficient")
plt.xticks(rotation=45)
plt.grid()

plt.show()