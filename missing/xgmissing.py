import copy
import frogress
import datetime as dt
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle

class SIB:
    def __init__(self, missing_attributes=['wind_speed', 'power', 'rotor_speed', 'generator_speed', 'temp_environment', 'temp_hydraulic_oil', 'temp_gear_bearing', 'blade_angle_avg'], iterations=3, debug=False):
        self.missing_attributes = np.asarray(missing_attributes)
        self.debug = debug
        self.iterations = iterations

    def train_xgb(self, df):
        # for each parc get the average of each attribute at each time

        # get all park ids
        self.park_ids = np.unique(df['park_id'])

        # preprocessing
        self.df_train = copy.deepcopy(df)
        self.df_train["measured_at"]=pd.to_datetime(self.df_train.measured_at)
        self.df_train["week"]=self.df_train.measured_at.dt.isocalendar().week.astype(int)
        self.df_train["month"]=self.df_train.measured_at.dt.month
        self.df_train["hourofday"]=self.df_train.measured_at.dt.hour
        self.df_train["isnight"]=(self.df_train.hourofday >= 18) | (self.df_train.hourofday <=5)
        self.df_train["isnoon"]=(self.df_train.hourofday >= 7) & (self.df_train.hourofday<=14)
        self.df_train["Error"]=self.df_train.error_category != "NO_ERROR"
        self.df_train["speed"]=(self.df_train.rotor_speed+self.df_train.generator_speed)
        self.df_train["direction"]=(self.df_train.nacelle_direction+self.df_train.wind_direction)


        if self.debug:
            self.df_train = self.df_train.iloc[:100]

    
        # build xgboost for each attribute
        self.xgbs = {}
        for at in self.missing_attributes:
            print(f"Building xgb for attribute {at}")
            xgb_cv=XGBRegressor(n_estimators=97 , learning_rate=0.1, max_depth=7,gamma=0.1, alpha=0.0)
            #params={"n_estimators":[32,64,90,100,150,200], "max_depth":[4,6,7,8,9],"learning_rate":[0.1,0.2,0.5],
            #        "reg_lambda":[0,0.1,0.01], "gamma":[0,0.1,0.3], "alpha":[0,0.02,0.1,0.5]}
            params={"n_estimators":[100], "learning_rate":[0.1], "max_depth":[6], "gamma":[0.1]}


            # xgb_cv=GridSearchCV(xgb, params, scoring='neg_mean_squared_error', cv=10)
            # XGBClassifier(objective="multi:softmax",use_label_encoder=False, n_estimators=97 , learning_rate=0.1, max_depth=7,gamma=0.1, alpha=0.0)


            # xgb_cv=GridSearchCV(xgb, params, scoring='neg_mean_squared_error', cv=10)
            y = self.df_train[at]
            x = self.df_train.drop(columns=[at, 'measured_at', 'error_category', "index", 'nacelle_direction', 'wind_direction', 'rotor_speed', 'generator_speed'])
            xgb_cv.fit(x, y)
            # print("Best parameters:", xgb_cv.best_params_ , ", Best CV MSE:", xgb_cv.best_score_)
            self.xgbs[at] = xgb_cv
            with open(f'final_missing_{at}.pkl', 'wb+') as f:
                pickle.dump(xgb_cv, f)

        del self.df_train

    def train_spartial(self, df_miss):

        df_miss_ = copy.deepcopy(df_miss)
        times = pd.to_datetime(df_miss_["measured_at"]).view(int)/ 10**9 / 60. # in minutes
        self.time_zero_point = np.min(times)
        self.times = times - self.time_zero_point
        df_miss_['date'] = self.times
        self.times = np.unique(np.asarray(self.times, dtype=int))

        # calculate the averages of the attributes for each park and time    
        averages = {}
        for park_id in self.park_ids:
            print(f"Building partial interpolator for park {park_id}")
            idx = np.where(df_miss_["park_id"] == park_id)[0]
            df_park = df_miss_.iloc[idx]
            averages[park_id] = np.zeros((len(self.times), len(self.missing_attributes)))
            for idt, time in frogress.bar(enumerate(self.times)):
                idx_time = np.where(df_park['date'] == time)[0]
                if len(idx_time) == 0:
                    averages[park_id][idt] = np.full((len(self.missing_attributes,)), np.nan)
                else:
                    averages[park_id][idt] = np.nanmean(df_park[self.missing_attributes].iloc[idx_time], axis=0)
                if self.debug:
                    if idt > 100:
                        break
        self.averages = averages 
        del df_miss_


    def predict(self, X):
        x = copy.deepcopy(X)
        x = x.to_frame().T
        x = x.astype({'turbine_id' : int, 'power' : float, 'temp_environment' : float, 'temp_hydraulic_oil' : float, 'temp_gear_bearing': float, 'cosphi': float, 'blade_angle_avg': float, 'hydraulic_pressure': float, 'park_id': int, 'rotor_speed': float, 'generator_speed' : float, 'nacelle_direction':float, 'wind_direction': float})

        # get missing attributes
        check_nans = x.isna()
        check_nans = np.asarray(check_nans).reshape(-1)
        missing_attributes = list(x.keys()[check_nans])
        time = pd.to_datetime(x["measured_at"]).view(int) / 10**9 / 60. # in minutes
        time = time - self.time_zero_point
        park = x['park_id']

        avg = self.averages[int(park)][np.where(np.asarray(self.times) == int(time))[0][0], :]
        # replace with initial guess as averages of windpark
        for at in missing_attributes:
            x[at] = avg[np.where(self.missing_attributes == at)[0][0]]

        # preprocessing
        temp=pd.to_datetime(x.measured_at)
        x["week"]=temp.dt.isocalendar().week.astype(int)
        x["month"]=temp.dt.month
        x["hourofday"]=temp.dt.hour
        x["isnight"]=(x.hourofday >= 18) | (x.hourofday <=5)
        x["isnoon"]=(x.hourofday >= 7) & (x.hourofday<=14)
        x["Error"]=x.error_category != "NO_ERROR"
        x["speed"]=(x.rotor_speed+x.generator_speed)
        x["direction"]=(x.nacelle_direction+x.wind_direction)

        # run xgbs
        for iter in range(self.iterations):
            for at in missing_attributes:
                x[at]= self.xgbs[at].predict(x.drop(columns=[at, 'measured_at', 'error_category', "index", 'nacelle_direction', 'wind_direction', 'rotor_speed', 'generator_speed']))
        x = x.drop(columns=["week", "month", 'hourofday', 'isnight', 'isnoon', 'Error', 'speed', 'direction'])
        return x

data_full=pd.read_csv("raw_data/trainset_full.csv")
sib = SIB()
sib.train_xgb(data_full)