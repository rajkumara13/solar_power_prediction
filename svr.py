import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint
import random
import datetime
import glob
from time import sleep
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def listToString(s): 
    str1 = "" 
    for ele in s: 
        str1 += ele  
    return str1

# Suppress warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# Load dataset
df = pd.read_csv("niteditedfinal.csv")
df = df.fillna(0)
nonzero_mean = df[df != 0].mean()

# Define features and targets
X = df.iloc[:, [0, 1, 2, 3, 4]].values
Y_temp = df.iloc[:, [5]].values
Y_ghi = df.iloc[:, [6]].values

# Scale data
sc_X = StandardScaler()
sc_Y_temp = StandardScaler()
sc_Y_ghi = StandardScaler()

X_scaled = sc_X.fit_transform(X)
Y_temp_scaled = sc_Y_temp.fit_transform(Y_temp)
Y_ghi_scaled = sc_Y_ghi.fit_transform(Y_ghi)

# Split data
x_train, x_test, y_temp_train, y_temp_test = train_test_split(X_scaled, Y_temp_scaled, random_state=42)
x_train, x_test, y_ghi_train, y_ghi_test = train_test_split(X_scaled, Y_ghi_scaled, random_state=42)

# Initialize SVR models
svr_temp = SVR(kernel='rbf')
svr_ghi = SVR(kernel='rbf')

# Train models
svr_temp.fit(x_train, y_temp_train.ravel())
svr_ghi.fit(x_train, y_ghi_train.ravel())

# Get current time + 15 minutes as prediction input
nextTime = datetime.datetime.now() + datetime.timedelta(minutes=15)
now = nextTime.strftime("%Y,%m,%d,%H,%M")
now = [int(i) for i in now.split(",")]

# Scale input
now_scaled = sc_X.transform([now])

# Predict
temp_scaled = svr_temp.predict(now_scaled)
ghi_scaled = svr_ghi.predict(now_scaled)

# Inverse transform predictions
temp = sc_Y_temp.inverse_transform(temp_scaled.reshape(-1, 1))[0][0]
ghi = sc_Y_ghi.inverse_transform(ghi_scaled.reshape(-1, 1))[0][0]

# Power calculation
f = 0.18 * 7.4322 * ghi
midd = 0.95 * (temp - 25)
power = f * midd

# Print results
print(f"Time (+15 min): {nextTime.strftime('%Y-%m-%d %H:%M')}")
print(f"Temperature: {temp:.2f} °C")
print(f"GHI: {ghi:.2f} W/m²")
print(f"Predicted Power Output: {power:.2f} W")

# Optional sleep if you loop
# sleep(1)
