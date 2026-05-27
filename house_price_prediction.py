import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset from URL
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

print("Dataset Preview:")
print(data.head())

# Features and target
X = data.drop("medv", axis=1)
y = data["medv"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("R² Score:", round(r2, 2))

# Sample prediction
sample_house = X_test.iloc[0:1]
predicted_price = model.predict(sample_house)

print("\nPredicted House Price:", round(predicted_price[0], 2))
print("Actual House Price:", y_test.iloc[0])