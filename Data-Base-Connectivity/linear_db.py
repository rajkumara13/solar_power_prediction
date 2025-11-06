import warnings
warnings.simplefilter("default")  
import pandas as pd
import numpy as np
import datetime
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import MySQLdb

db = MySQLdb.connect(host="localhost", user="root", passwd="", db="prediction_v2")
cur = db.cursor()

def list_to_float(lst):
    """Convert a single-element list to a float"""
    return float(''.join(map(str, lst)))

while True:
    try:
        # Load dataset
        df = pd.read_csv("niteditedfinal.csv")
        df = df.fillna(0)

        X = df.iloc[:, 0:5].values  # First 5 columns as features
        Y_temp = df.iloc[:, 5].values  # 6th column: temperature
        Y_ghi = df.iloc[:, 6].values  # 7th column: GHI

        # Train/test split
        x_train, x_test, y_temp_train, y_temp_test = train_test_split(X, Y_temp, random_state=42)
        _, _, y_ghi_train, y_ghi_test = train_test_split(X, Y_ghi, random_state=42)

        model_temp = LinearRegression()
        model_ghi = LinearRegression()

        model_temp.fit(x_train, y_temp_train)
        model_ghi.fit(x_train, y_ghi_train)

        future_time = datetime.datetime.now() + datetime.timedelta(minutes=15)
        timestamp = future_time.strftime("%Y-%m-%d %H:%M")
        time_features = future_time.strftime("%Y,%m,%d,%H,%M").split(",")
        time_features = list(map(float, time_features))  

        temp = model_temp.predict([time_features])
        ghi = model_ghi.predict([time_features])

        # Power Calculation
        panel_efficiency = 0.18
        panel_area = 7.4322
        temp_val = list_to_float(temp)
        ghi_val = list_to_float(ghi)

        irradiance_factor = panel_efficiency * panel_area * ghi_val
        temperature_correction = 1 - 0.05 * (temp_val - 25)
        power = irradiance_factor * temperature_correction

        # Insert into DB
        sql = ("""INSERT INTO power_pred_lr (time_updated, Temperature, GHI, power) 
                  VALUES (%s, %s, %s, %s)""", (timestamp, temp_val, ghi_val, power))

        try:
            print(f"Writing to DB at {timestamp} | Temp: {temp_val:.2f} | GHI: {ghi_val:.2f} | Power: {power:.2f}")
            cur.execute(*sql)
            db.commit()
        except Exception as db_err:
            db.rollback()
            print("Database write failed:", db_err)

        time.sleep(1)

    except Exception as e:
        print("Something went wrong:", e)
        time.sleep(1)
