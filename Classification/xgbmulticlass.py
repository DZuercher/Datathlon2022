def main():
    import numpy as np
    import pandas as pd

    df=pd.read_csv("raw_data/trainset_full.csv", low_memory=False)

    import datetime as dt
    df["measured_at"]=pd.to_datetime(df.measured_at)
    df["week"]=np.int64(df.measured_at.dt.isocalendar().week)
    df["month"]=df.measured_at.dt.month
    df["hourofday"]=df.measured_at.dt.hour
    df["isnight"]=(df.hourofday >= 18) | (df.hourofday <=5)
    df["highweek"]= (df.week == 14) | (df.week ==13) | (df.week == 5) | (df.week == 7) | (df.week == 37) | (df.week == 29) | (df.week==30) | (df.week == 31)
    df["isnoon"]=(df.hourofday >= 7) & (df.hourofday<=14)
    df["Error"]=df.error_category != "NO_ERROR"
    df["speed"]=(df.rotor_speed+df.generator_speed)
    df["direction"]=(df.nacelle_direction+df.wind_direction)


    xgb_attribs=['turbine_id', 'wind_speed', 'power','week',
       'temp_environment', 'temp_hydraulic_oil', 'temp_gear_bearing', 'cosphi',
       'blade_angle_avg', 'hydraulic_pressure', 'park_id', 'month', 'speed', 'direction','isnight', 'isnoon','highweek']

    from sklearn.preprocessing import LabelEncoder

    labelencoder=LabelEncoder()
    labelencoder.fit(df.error_category)
    df["EncodedErrors"]=labelencoder.transform(df.error_category)

    X_train_xgb=df[xgb_attribs]
    y_train_xgb=df.EncodedErrors

    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV

    xgb=XGBClassifier(objective="multi:softmax",use_label_encoder=False)
    parameters={"n_estimators":[50,80,90,100,120,150,200], "learning_rate":[0.1,0.2,0.5], "max_depth":[4,6,7,8,9,10], "gamma":[0,0.1,0.3],"alpha":[0,0.02,0.1,0.5]}

    xgb_cv=GridSearchCV(xgb, parameters, scoring="accuracy",cv=10)
    xgb_cv.fit(X_train_xgb,y_train_xgb)

    print("Best parameters:", xgb_cv.best_params_ , ", Best CV Accuracy:", xgb_cv.best_score_)

if __name__ == "__main__":
    main()