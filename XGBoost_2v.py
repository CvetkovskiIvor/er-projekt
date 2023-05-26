import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_visualisation(data: pd.DataFrame):
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


    correlation = data.corr().abs()
    plt.figure(figsize=(20, 10))
    sns.heatmap(correlation, annot=True)
    plt.show()


    plt.figure(figsize=(12, 3))

    plt.title('Trip Duration Distribution')
    plt.xlabel('Trip Duration, minutes')
    plt.ylabel('No of Trips made')
    plt.hist(data.trip_duration, bins=100)

    plt.figure(figsize=(12, 6))

    data_by_hour = data.groupby('pickup_hour').size().reset_index(name='count')
    sns.barplot(x='pickup_hour', y='count', data=data_by_hour)

    plt.title('Pick-ups Hour Distribution')
    plt.xlabel('Hour of Day, 0-23')
    plt.ylabel('No of Trips made')
    plt.xticks(range(0, 24))
    plt.show()

    plt.figure(figsize=(12, 6))

    data_by_day = data.groupby("pickup_weekday").size().reset_index(name='count')
    sns.barplot(x='pickup_weekday', y='count', data=data_by_day)
    plt.title('Pick-ups Weekday Distribution')
    plt.xlabel('Weekday')
    plt.ylabel('No of Trips made')
    plt.xticks(range(0, 7), dow_names, rotation='horizontal')
    plt.show()

    pc = data.groupby('passenger_count')['trip_duration'].mean()

    plt.subplots(1, 1, figsize=(17, 10))
    plt.ylim(ymin=0)
    plt.ylim(ymax=19)
    plt.title('Time per store_and_fwd_flag')
    plt.legend(loc=0)
    plt.ylabel('Time in Minutes')
    sns.barplot(x=pc.index, y=pc.values)
    plt.show()


def feature_engineer(data: pd.DataFrame):
    """
    Function to pre-process and engineer features of the train data
    """
    f = lambda x: 0 if x == 'N' else 1
    data["store_and_fwd_flag"] = data["store_and_fwd_flag"].apply(f)

    data["dropoff_datetime"] = pd.to_datetime(data["dropoff_datetime"], format='%Y-%m-%d %H:%M:%S')
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"], format='%Y-%m-%d %H:%M:%S')


    data["pickup_month"] = data["pickup_datetime"].dt.month
    data["pickup_day"] = data["pickup_datetime"].dt.day
    data["pickup_weekday"] = data["pickup_datetime"].dt.weekday
    data["pickup_hour"] = data["pickup_datetime"].dt.hour
    data["pickup_minute"] = data["pickup_datetime"].dt.minute

    data["latitude_difference"] = data["dropoff_latitude"] - data["pickup_latitude"]
    data["longitude_difference"] = data["dropoff_longitude"] - data["pickup_longitude"]

    data["trip_duration"] = data["trip_duration"].apply(lambda x: round(x / 60))

    m = np.mean(data['trip_duration'])
    s = np.std(data['trip_duration'])
    data.loc[data['trip_duration'] < m - 2 * s, 'trip_duration'] = m - 2 * s
    data.loc[data['trip_duration'] > m + 2 * s, 'trip_duration'] = m + 2 * s

    data["trip_distance"] = 0.621371 * 6371 * (abs(2 * np.arctan2(
        np.sqrt(
            np.square(np.sin((abs(data["latitude_difference"]) * np.pi / 180) / 2))
        ), np.sqrt(
            1 - (np.square(np.sin((abs(data["latitude_difference"]) * np.pi / 180) / 2)))
        )
    )) +
                                                abs(2 * np.arctan2(
                                                    np.sqrt(
                                                        np.square(
                                                            np.sin((abs(data["longitude_difference"]) * np.pi / 180) / 2))
                                                    ), np.sqrt(
                                                        1 - (np.square(
                                                            np.sin((abs(data["longitude_difference"]) * np.pi / 180) / 2)))
                                                    )
                                                )))


def xgb_model(X: pd.DataFrame, y: pd.Series) -> xgb.Booster:
    """
    Function to train an XGBoost machine learning model on the data
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2019)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=2019)

    params = {
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'learning_rate': 0.05,
        'max_depth': 14,
        'subsample': 0.9,
        'colsample_bytree': 0.7,
        'colsample_bylevel': 0.7,
        'silent': 1,
    }

    dtrain = xgb.DMatrix(X_train, np.log(y_train + 1))
    dval = xgb.DMatrix(X_val, np.log(y_val + 1))

    watchlist = [(dval, 'eval'), (dtrain, 'train')]

    num_rounds = 200

    model = xgb.train(params, dtrain, num_boost_round=num_rounds, evals=watchlist, verbose_eval=True)

    y_pred = np.exp(model.predict(xgb.DMatrix(X_test))) - 1

    mae = np.mean(abs(y_pred - y_test))

    feature_scores = model.get_fscore()
    summ = sum(feature_scores.values())
    feature_scores = {key: value / summ for key, value in feature_scores.items()}

    print(f'Mean Absolute Error: {mae}')
    print(f'Feature Importance: {feature_scores}')

    return model


if __name__ == '__main__':
    taxiDB = pd.read_csv("train.csv")
    feature_engineer(taxiDB)
    data_visualisation(taxiDB)
    X = taxiDB.drop(["trip_duration", "id", "vendor_id", "pickup_datetime", "dropoff_datetime"], axis=1)
    y = taxiDB["trip_duration"]
    xgb_model = xgb_model(X, y)
    filename = "xgb_model2.xgb"
    xgb_model.save_model(filename)
