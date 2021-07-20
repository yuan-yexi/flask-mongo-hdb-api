import pandas as pd
import numpy as np
import json
import requests
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score

import xgboost as xgb

def create_dataframe(data):
    town = []
    flat_type = []
    flat_model = []
    floor_area_sqm = []
    street_name = []
    resale_price = []
    month = []
    remaining_lease = []
    lease_commence_date = []
    storey_range = []
    _id = []
    block = []
    for i in range(0, len(data)):
        town.append(data[i]['town'])
        flat_type.append(data[i]['flat_type'])
        flat_model.append(data[i]['flat_model'])
        floor_area_sqm.append(data[i]['floor_area_sqm'])
        street_name.append(data[i]['street_name'])
        resale_price.append(data[i]['resale_price'])
        month.append(data[i]['month'])
        remaining_lease.append(data[i]['remaining_lease'])
        lease_commence_date.append(data[i]['lease_commence_date'])
        storey_range.append(data[i]['storey_range'])
        _id.append(data[i]['_id'])
        block.append(data[i]['block'])
    
    dataframe = pd.DataFrame({
        '_id': _id,
        'town': town,
        'flat_type': flat_type,
        'flat_model': flat_model,
        'block': block,
        'street_name': street_name,
        'month': month,
        'remaining_lease': remaining_lease,
        'lease_commence_date': lease_commence_date,
        'storey_range': storey_range,
        'floor_area_sqm': floor_area_sqm,
        'resale_price': resale_price
    })

    return dataframe

def split_mean(x):
    split_list = x.split(' TO ')
    mean = (float(split_list[0])+float(split_list[1]))/2
    
    return mean

def map_flat_type(dataframe):
    flat_type_map = {
        'EXECUTIVE': 7,
        'MULTI-GENERATION': 6,
        '5 ROOM': 5,
        '4 ROOM': 4,
        '3 ROOM': 3,
        '2 ROOM': 2,
        '1 ROOM': 1
    }

    return dataframe['flat_type'].map(lambda x: flat_type_map[x])

def lease_commence(dataframe):
    # Clean lease remaining
    dataframe['lease_commence_date'] = dataframe['lease_commence_date'].astype('int64')
    return 99 - (int(datetime.today().year) - dataframe['lease_commence_date'])
    
def dummy_var_town(dataframe):
    return pd.get_dummies(data=dataframe, columns=['town'], drop_first=True)

if __name__ == "__main__":
    # Call API
    query_string='https://data.gov.sg/api/action/datastore_search?resource_id=42ff9cfe-abe5-4b54-beda-c88f9bb438ee&limit=200000'
    resp = requests.get(query_string) # Convert JSON into Python Object 
    data = json.loads(resp.content)
    response_data = data['result']['records']

    # Transform features
    price_data = create_dataframe(response_data)
    price_data['storey_mean'] = price_data['storey_range'].apply(lambda x: split_mean(x))
    price_data['flat_type_mapped'] = map_flat_type(price_data)
    price_data['lease_remain_years'] = lease_commence(price_data)
    price_data = dummy_var_town(price_data)
    price_data['resale_price'] = price_data['resale_price'].astype('float32')
    price_data['floor_area_sqm'] = price_data['floor_area_sqm'].astype('float32')
    price_data['price_per_sqft'] = ((price_data['resale_price'] / price_data['floor_area_sqm']) / 10.764).round(2)

    # Prep training data
    X = price_data[['floor_area_sqm', 'flat_type_mapped', 'storey_mean', 'lease_remain_years']]

    y = price_data['price_per_sqft']

    # Train, test, split
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

    model_xgb = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=3, learning_rate=0.01, missing=1)
    model_xgb.fit(X_train, y_train, 
          eval_set=[(X_train, y_train), (X_test, y_test)], 
          early_stopping_rounds=20)


    model_xgb.save_model('xgb_reg_pred.json')
    model_export = xgb.XGBRegressor()
    model_export.load_model('xgb_reg_pred.json')
    print(model_export)
