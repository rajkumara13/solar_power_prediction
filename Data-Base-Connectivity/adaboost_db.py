import pandas as pd
import numpy as np
import datetime
import time
import MySQLdb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# MySQL connection
db = MySQLdb.connect(host="localhost", user="root", passwd="", db="prediction_v2")
cur = db.cursor()

while True:
    df = pd.read_csv("niteditedfinal.csv")
    df = df.fillna(0)

    X = df.iloc[:, [0, 1, 2, 3, 4]].astype(np.float32).values
    Y_temp = df.iloc[:, 5].astype(np.float32).values.ravel()
    Y_ghi = df.iloc[:, 6].astype(np.float32).values.ravel()

    x_train, _, y_temp_train, _ = train_test_split(X, Y_temp, random_state=42)
    _, _, y_ghi_train, _ = train_test_split(X, Y_ghi, random_state=42)

    model_temp = AdaBoostRegressor(estimator=DecisionTreeRegressor(), n_estimators=100, random_state=42)
    model_ghi = AdaBoostRegressor(estimator=DecisionTreeRegressor(), n_estimators=100, random_state=42)

    model_temp.fit(x_train, y_temp_train)
    model_ghi.fit(x_train, y_ghi_train)

    now_dt = datetime.datetime.now() + datetime.timedelta(minutes=15)
    time_str = now_dt.strftime("%Y-%m-%d %H:%M")
    now_input = np.array(now_dt.strftime("%Y,%m,%d,%H,%M").split(","), dtype=np.float32).reshape(1, -1)

    temp = float(model_temp.predict(now_input)[0])
    ghi = float(model_ghi.predict(now_input)[0])

    irradiance = 0.18 * 7.4322 * ghi
    midd = 0.95 * (temp - 25)
    power = float(irradiance * midd)

    print("Time:", time_str)
    print("Temperature:", temp)
    print("GHI:", ghi)
    print("Power:", power)

    sql = (
        """INSERT INTO power_pred_adaboost (time_updated,Temperature,GHI,power) VALUES (%s,%s,%s,%s)""",
        (time_str, temp, ghi, power),
    )

    try:
        cur.execute(*sql)
        db.commit()
    except Exception as e:
        db.rollback()
        print("Database error:", e)

    time.sleep(1)
