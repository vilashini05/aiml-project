import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("ai_student_impact_dataset (1).csv")

# Separate input and output
X = df.drop(columns=['Burnout_Risk_Level'])
y = df['Burnout_Risk_Level']

# Create training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Display dimensions
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)