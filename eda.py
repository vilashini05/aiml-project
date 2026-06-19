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

print("\n--- Info ---")
print(df.info())

print("\n--- Statistical Summary ---")
print(df.describe())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# Numerical columns preview
numerical_columns = df.select_dtypes(include=['int64', 'float64'])
print("\nNumerical Columns Preview:")
print(numerical_columns.head())

# =========================
# FULL COLUMN ANALYSIS
# =========================
for col in df.columns:

    print("\n======================")
    print("Column:", col)

    # ---------------------
    # NUMERICAL COLUMNS
    # ---------------------
    if df[col].dtype in ['int64', 'float64']:

        mean = df[col].mean()
        std = df[col].std()
        minimum = df[col].min()
        maximum = df[col].max()

        q1 = df[col].quantile(0.25)
        q2 = df[col].quantile(0.50)
        q3 = df[col].quantile(0.75)

        skew = df[col].skew()

        # Distribution type
        if abs(skew) < 0.5:
            dist = "Normal Distribution"
        elif skew > 1:
            dist = "Highly Right Skewed"
        elif skew > 0.5:
            dist = "Right Skewed"
        elif skew < -1:
            dist = "Highly Left Skewed"
        else:
            dist = "Left Skewed"

        print("Distribution:", dist)
        print("Mean:", mean)
        print("Std Dev:", std)
        print("Min:", minimum)
        print("Max:", maximum)
        print("25%:", q1)
        print("50% (Median):", q2)
        print("75%:", q3)

        # Histogram
        plt.figure()
        df[col].hist(bins=20, edgecolor='black')
        plt.title(col + " - " + dist)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

    # ---------------------
    # CATEGORICAL COLUMNS
    # ---------------------
    else:

        print("Distribution: Categorical")

        # Bar chart
        plt.figure()
        df[col].value_counts().plot(kind='bar')
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()