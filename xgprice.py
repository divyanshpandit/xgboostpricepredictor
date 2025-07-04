import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1.__Load data__))
df = pd.read_csv("HDFC.csv")

# 2.__Preprocess__))
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

#__Convert numeric columns__))
numeric_cols = [
    "Prev Close", "Open", "High", "Low", "VWAP", "Volume", "Turnover", "Close"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 3.__Feature engineering__))
# Moving averages
df["MA5"] = df["Close"].rolling(window=5).mean()
df["MA10"] = df["Close"].rolling(window=10).mean()

#__RSI (14)__))
delta = df["Close"].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(14).mean()
avg_loss = pd.Series(loss).rolling(14).mean()
rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

#__Drop only rows where essential features are missing__))
required_columns = [
    "Prev Close", "Open", "High", "Low", "VWAP", "Volume", "Turnover", "MA5", "MA10", "RSI", "Close"
]
df = df.dropna(subset=required_columns)

#__Confirm data is valid__))
print(f"After cleaning, data shape: {df.shape}")

# 4.__Features and labels__))
features = [
    "Prev Close", "Open", "High", "Low", "VWAP", "Volume", "Turnover", "MA5", "MA10", "RSI"
]
X = df[features]
y = df["Close"]

# 5.__Train-test split__))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# 6.__XGBoost model__))
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)

# 7.__Predict__))
y_pred = model.predict(X_test)

# 8.__Evaluate__))
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.2f}")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test R2: {r2:.4f}")

# 9.__Plot predictions vs actual__))
plt.figure(figsize=(14, 7))
plt.plot(df["Date"].iloc[-len(y_test):], y_test, label="Actual", color="blue")
plt.plot(df["Date"].iloc[-len(y_test):], y_pred, label="Predicted", color="red")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("HDFC Price Prediction with XGBoost")
plt.legend()
plt.show()
