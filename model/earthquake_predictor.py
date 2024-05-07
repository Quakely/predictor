import numpy as np
import pandas as pd
import os
import warnings
import datetime as dt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import xgboost as xgb
import warnings
from sqlalchemy import create_engine
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import timedelta
from joblib import load


class EarthquakePredictor:
    warnings.filterwarnings('ignore')
    warnings.simplefilter(action='ignore', category=FutureWarning)

    def append_to_database(self, predictions):
        engine = create_engine('sqlite:///Earthquakedata_predict.db')
        with engine.connect() as connection:
            column_order = ['date', 'place', 'latitude', 'longitude', 'nst', 'depth', 'depth_ewma_15', 'depth_ewma_7',
                            'mag_ewma_22', 'mag_ewma_15', 'mag_ewma_7', 'tremors_count_7d', 'energy', 'total_energy_7d',
                            'tremors_count_15d', 'total_energy_15d', 'tremors_count_22d', 'total_energy_22d',
                            'mag_outcome']
            predictions[column_order].to_sql('Earthquake_predict', connection, if_exists='append', index=False,
                                             method='multi', chunksize=1000)

    def predict(self):
        engine = create_engine('sqlite:///Earthquakedata.db')
        df_features = pd.read_sql_table('Earthquake_features', con=engine)
        engine = create_engine('sqlite:///Earthquakedata_predict.db')
        df_predict = pd.read_sql_table('Earthquake_predict', con=engine)

        features = [f for f in list(df_features) if f not in ['date', 'lon_box_mean',
                                                              'lat_box_mean', 'mag_outcome', 'mag', 'place',
                                                              'combo_box_mean', 'latitude',
                                                              'longitude']]

        X_train, X_test, y_train, y_test = train_test_split(df_features[features],
                                                            df_features['mag_outcome'], test_size=0.3, random_state=42)

        xgboost_model = xgb.Booster()
        xgboost_model.load_model('xgboost_model.model')

        dlive = xgb.DMatrix(df_predict[features])
        preds = xgboost_model.predict(dlive)

        live_set = df_predict[
            ['date', 'place', 'latitude', 'longitude', 'nst', 'depth', 'depth_ewma_15', 'depth_ewma_7', 'mag_ewma_22',
             'mag_ewma_15', 'mag_ewma_7', 'tremors_count_7d', 'energy', 'total_energy_7d', 'tremors_count_15d',
             'total_energy_15d', 'tremors_count_22d', 'total_energy_22d', 'mag_outcome']]
        live_set.loc[:, 'mag'] = preds
        live_set = live_set.groupby(['date', 'place'], as_index=False).mean()

        live_set['date'] = live_set['date'].str.slice(stop=10)
        print(live_set['date'])
        live_set['date'] = pd.to_datetime(live_set['date'], format='%Y-%m-%d')
        live_set['date'] = live_set['date'] + pd.to_timedelta(7, unit='d')

        days = list(set([d for d in live_set['date'].astype(str) if
                         d > (dt.datetime.today() - dt.timedelta(days=1)).strftime('%Y-%m-%d')]))
        days.sort()

        earthquake_data = []

        for i in range(min(7, len(days))):
            live_set_tmp = live_set[live_set['date'] == days[i]]
            for index, row in live_set_tmp.iterrows():
                earthquake_data.append([row['latitude'], row['longitude'], row['nst'], row['depth']])

        new_data_array = np.array(earthquake_data)

        print(new_data_array)
        # self.append_to_database(live_set)

        loaded_model = load('/home/elie/backends/predictor/random_forest_model.joblib')
        earthquake_data = []

        for i in range(min(7, len(days))):
            live_set_tmp = live_set[live_set['date'] == days[i]]
            for index, row in live_set_tmp.iterrows():
                prediction = loaded_model.predict([[row['latitude'], row['longitude'], row['nst'], row['depth']]])[0]
                earthquake_data.append(
                    [row['date'], row['latitude'], row['longitude'], row['nst'], row['depth'], prediction])

        new_data_array = np.array(earthquake_data)
        return new_data_array
