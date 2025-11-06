import warnings
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import time

# Load dataset
df = pd.read_csv("niteditedfinal.csv")
df = df.fillna(0)

# Combine date and time columns (optional but recommended)
df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])

# Filter train data (Year 1 and Year 2)
train_df = df[df['Year'].isin([2017, 2018])]  # change years as needed
test_df = df[df['Year'].isin([2019])] 
# Prepare training data
X_train = train_df[['Year', 'Month', 'Day', 'Hour', 'Minute']].values
y_temp_train = train_df.iloc[:, 5].values  # Assuming temp at col 5
y_ghi_train = train_df.iloc[:, 6].values   # Assuming GHI at col 6
X_test=test_df[['Year', 'Month', 'Day', 'Hour', 'Minute']].values
y_temp_test = test_df.iloc[:, 5].values  # Assuming temp at col 5
y_ghi_test= test_df.iloc[:, 6].values 
# Train models
model_temp = RandomForestRegressor()
model_temp.fit(X_train, y_temp_train)

model_ghi = RandomForestRegressor()
model_ghi.fit(X_train, y_ghi_train)

# Predict every 15 minutes


    # Predict temp and ghi
temp = model_temp.predict(X_test)
ghi = model_ghi.predict(X_test)

    # Calculate Power: P = ηSI(1 - 0.05(T - 25))
eta = 0.18
area = 7.4322
temp_factor = (1 - 0.05 * (temp - 25))
power = eta * area * ghi * temp_factor
test_df['temp_pred']=temp
test_df['ghi_pred']=ghi
test_df['power_pred']=power
for i in range(100):
    print(f"Temperature: {temp[i]:.2f} °C")
    print(f"GHI: {ghi[i]:.2f} W/m²")
    print(f"Predicted Power: {power[i]:.2f} W")
mse_temp=mean_squared_error(y_temp_test,temp)
print("Mean squared mean for temperature:",mse_temp)
mae_temp=mean_absolute_error(y_temp_test,temp)
print("Mean absolute mean for temperature:",mae_temp)
r2_temp=r2_score(y_temp_test,temp)
print("R2 score for temperature:",r2_temp)
mse_ghi=mean_squared_error(y_ghi_test,ghi)
print("Mean squared mean for ghi:",mse_ghi)
mae_ghi=mean_absolute_error(y_ghi_test,ghi)
print("Mean absolute mean for ghi:",mae_ghi)
r2_ghi=r2_score(y_ghi_test,ghi)
print("R2 score for ghi:",r2_ghi)
