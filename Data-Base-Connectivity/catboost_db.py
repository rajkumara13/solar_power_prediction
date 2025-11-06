import warnings
import pandas as pd
import numpy as np
import datetime
import time
import MySQLdb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

db = MySQLdb.connect(host="localhost", user="root", passwd="", db="prediction_v2")
cur = db.cursor()

while True:
    df = pd.read_csv("niteditedfinal.csv")
    df = df.fillna(0)
    df = df[df['GHI'] > 0]

    X = df.iloc[:, [0, 1, 2, 3, 4]].values
    Y_temp = df.iloc[:, 5].values
    Y_ghi = df.iloc[:, 6].values

    x_train, _, y_temp_train, _ = train_test_split(X, Y_temp, random_state=42)
    x_train, _, y_ghi_train, _ = train_test_split(X, Y_ghi, random_state=42)

    model_temp = CatBoostRegressor(verbose=0)
    model_ghi = CatBoostRegressor(verbose=0)

    model_temp.fit(x_train, y_temp_train)
    model_ghi.fit(x_train, y_ghi_train)

    latest_input = df.iloc[-1, [0, 1, 2, 3, 4]].values.reshape(1, -1)

    temp = float(model_temp.predict(latest_input)[0])
    ghi = float(model_ghi.predict(latest_input)[0])

    temp = max(temp, 0)
    ghi = max(ghi, 0)
    delta_temp = max(temp - 25, 0)

    power = 0.18 * 7.4322 * ghi * 0.95 * delta_temp

    time_str = (datetime.datetime.now() + datetime.timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M")

    # Output to screen
    print(f"Time: {time_str} | Temperature: {round(temp,2)} | GHI: {round(ghi,2)} | Power: {round(power,2)}")

    sql = """INSERT INTO power_pred_catboost (time_updated, Temperature, GHI, power) VALUES (%s, %s, %s, %s)"""
    values = (time_str, temp, ghi, power)

    try:
        cur.execute(sql, values)
        db.commit()
    except:
        db.rollback()

    time.sleep(1)
