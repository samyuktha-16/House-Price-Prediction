import json
import pickle
import numpy as np


location=None
data_columns=None
model=None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    return round(model.predict([x])[0],2)
    

def load_saved_artifacts():
    print("loading the saved artifacts...start")
    global data_columns
    global location

    with open("server/artifacts/columns.json","r") as f:
        data_columns=json.load(f)['data_columns']
        location=data_columns[3:]
        global model
    with open("server/artifacts/bengaluru_house_prices.pickle","rb") as f:
        model=pickle.load(f)
    print("loading saved artifacts..done")

def get_location_names():
    return location 

def get_data_columns():
    return data_columns

if __name__=='__main__':
    load_saved_artifacts()
    print(get_location_names())
    #