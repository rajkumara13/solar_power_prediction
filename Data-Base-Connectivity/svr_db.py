
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pprint
import time
from time import sleep
import random
import datetime
import glob
import MySQLdb
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

db = MySQLdb.connect(host="localhost", user="root", passwd="", db="prediction_v2")
cur = db.cursor()

def listToString(s): 
    str1 = "" 
    for ele in s: 
        str1 += ele  
    return str1

while True:
    df = pd.read_csv("niteditedfinal.csv")
    df = df.fillna(0)

    cols = [0, 1, 2, 3, 4]
    X = df[df.columns[cols]].values

    cols = [5]
    Y_temp = df[df.columns[cols]].values.ravel()

    cols = [6]
    Y_ghi = df[df.columns[cols]].values.ravel()

    x_train, x_test, y_temp_train, y_temp_test = train_test_split(X, Y_temp, random_state=42)
    x_train, x_test, y_ghi_train, y_ghi_test = train_test_split(X, Y_ghi, random_state=42)

    svr1 = SVR()
    svr2 = SVR()

    svr1.fit(x_train, y_temp_train)
    svr2.fit(x_train, y_ghi_train)

    time_now = datetime.datetime.now() + datetime.timedelta(minutes=15)
    time_str = time_now.strftime("%Y-%m-%d %H:%M")

    now = time_now.strftime("%Y,%m,%d,%H,%M")
    now = list(map(int, now.split(",")))

    temp = svr1.predict([now])
    ghi = svr2.predict([now])

    f = 0.18 * 7.4322 * ghi
    insi = temp - 25
    midd = 0.95 * insi
    power = f * midd
    power = power.tolist()
    power = ''.join(map(str, power))
    power = float(power)

    temp = temp.tolist()
    temp = ' '.join(map(str, temp))
    temp = float(temp)

    ghi = ghi.tolist()
    ghi = ' '.join(map(str, ghi))
    ghi = float(ghi)

    print("Time:", time_str)
    print("Temperature:", temp)
    print("GHI:", ghi)
    print("Power:", power)

    sql = ("""INSERT INTO power_pred_svr (time_updated,Temperature,GHI,power) VALUES (%s,%s,%s,%s)""", (time_str, temp, ghi, power))

    try:  
        cur.execute(*sql)  
        db.commit()  
    except:  
        db.rollback()  

    time.sleep(1)
