def main(indices, ARGS):

    import pandas as pd
    import numpy as np
    from xgboost import XGBRegressor
    import pickle
    import copy
    import frogress

    missing_attributes = np.asarray(['wind_speed', 'power', 'rotor_speed', 'generator_speed', 'temp_environment', 'temp_hydraulic_oil', 'temp_gear_bearing', 'blade_angle_avg'])


    # SET TO FALSE
    debug = False
    iterations = 3


    # laod in data
    data=pd.read_csv("/cluster/scratch/dominikz/hiddenset.csv")

    times = pd.to_datetime(data["measured_at"]).view(int)/ 10**9 / 60. # in minutes
    time_zero_point = np.min(times)
    times = times - time_zero_point
    times = np.unique(np.asarray(times, dtype=int))
    print(f"Got {len(times)} different times")

    # load xgboosted trees
    xgbs = {}
    for at in missing_attributes:
        with open(f'/cluster/scratch/dominikz/final_missing_{at}.pkl', 'rb') as f:
            xgbs[at] = pickle.load(f)

    with open(f'/cluster/scratch/dominikz/averages.pkl', 'rb') as f:
        averages = pickle.load(f)


    n = 5000  #chunk row size
    list_df = [data[i:i+n] for i in range(0,data.shape[0],n)]
    for index in indices:
        data = list_df[index]

        print(f"Got {data.shape[0]} rows")
        for ix, x in frogress.bar(data.iterrows()):
            x = x.to_frame().T
            x = x.astype({'turbine_id' : int, 'power' : float, 'temp_environment' : float, 'temp_hydraulic_oil' : float, 'temp_gear_bearing': float, 'cosphi': float, 'blade_angle_avg': float, 'hydraulic_pressure': float, 'park_id': int, 'rotor_speed': float, 'generator_speed' : float, 'nacelle_direction':float, 'wind_speed': float, 'wind_direction': float})

            # get missing attributes
            check_nans = x.isna()
            check_nans = np.asarray(check_nans).reshape(-1)
            missing = list(x.keys()[check_nans])
            if len(missing) == 0:
                continue
            time = pd.to_datetime(x["measured_at"]).view(int) / 10**9 / 60. # in minutes
            time = time - time_zero_point
            time = np.asarray(time, dtype=int)[0]
            park = x['park_id']

            avg = averages[int(park)][np.where(np.asarray(times) == int(time))[0][0], :]
            # replace with initial guess as averages of windpark
            for at in missing:
                x[at] = avg[np.where(missing_attributes == at)[0]]

            # preprocessing
            temp=pd.to_datetime(x.measured_at)
            x["week"]=temp.dt.isocalendar().week.astype(int)
            x["month"]=temp.dt.month
            x["hourofday"]=temp.dt.hour
            x["isnight"]=(x.hourofday >= 18) | (x.hourofday <=5)
            x["isnoon"]=(x.hourofday >= 7) & (x.hourofday<=14)
            # x["Error"]=x.error_category != "NO_ERROR"
            x["speed"]=(x.rotor_speed+x.generator_speed)
            x["direction"]=(x.nacelle_direction+x.wind_direction)

            # run xgbs
            for iter in range(iterations):
                for at in missing:
                    x[at]= xgbs[at].predict(x.drop(columns=[at, 'measured_at', "index", 'nacelle_direction', 'wind_direction', 'rotor_speed', 'generator_speed']))

            for at in missing:
                data[at][ix] = x[at]

        data.to_csv(f"/cluster/scratch/dominikz/part_{index}.csv")
        yield index
