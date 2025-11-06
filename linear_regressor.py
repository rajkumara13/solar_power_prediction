import warnings
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Suppress warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# Load and clean dataset
df = pd.read_csv("niteditedfinal.csv")
df = df.fillna(0)

# Select features and targets
X = df.iloc[:, [0, 1, 2, 3, 4]].values  # Features: Year, Month, Day, Hour, Minute
Y_temp = df.iloc[:, [5]].values        # Target: Temperature
Y_ghi = df.iloc[:, [6]].values         # Target: GHI

# Split into training and testing sets
x_train, x_test, y_temp_train, y_temp_test = train_test_split(X, Y_temp, random_state=42)
x_train, x_test, y_ghi_train, y_ghi_test = train_test_split(X, Y_ghi, random_state=42)

# Train Linear Regression models
lr_temp = LinearRegression()
lr_ghi = LinearRegression()

lr_temp.fit(x_train, y_temp_train)
lr_ghi.fit(x_train, y_ghi_train)

# Prepare future input time (+15 minutes)
nextTime = datetime.datetime.now() + datetime.timedelta(minutes=15)
now = nextTime.strftime("%Y,%m,%d,%H,%M")
now = [int(i) for i in now.split(",")]

# Predict temperature and GHI
temp = lr_temp.predict([now])[0][0]
ghi = lr_ghi.predict([now])[0][0]

# Power calculation
f = 0.18 * 7.4322 * ghi
midd = 0.95 * (temp - 25)
power = f * midd

# Display results
print(f"Time (+15 min): {nextTime.strftime('%Y-%m-%d %H:%M')}")
print(f"Temperature: {temp:.2f} °C")
print(f"GHI: {ghi:.2f} W/m²")
print(f"Predicted Power Output: {power:.2f} W")
