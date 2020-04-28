import pandas as pd
import pickle

def get_time_series_data():
    '''
    map: country_state -> 
                        [
                            [confirmed]
                            [death]
                            [recovered]
                            # active = confirmed - death - recovered
                        ]
    '''
    ts = dict()

    # extract confirned time series
    confirmed_df = pd.read_csv('data/jhu/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

    for index, row in confirmed_df.iterrows():
        key = str(row['Country/Region'])+'_'+str(row['Province/State'])
        ts[key] = [[i for i in row.values][4:]] # e.g. China_Beijing, 3d array [confirmed, death, recovered]
    
    # extract death time series
    death_df = pd.read_csv('data/jhu/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

    for index, row in death_df.iterrows():
        key = str(row['Country/Region'])+'_'+str(row['Province/State'])
        try:
            ts[key].append([i for i in row.values][4:]) # e.g. China_Beijing, 3d array [confirmed, death, recovered]
        except:
            continue
    
    # extract recover time series
    recover_df = pd.read_csv('data/jhu/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

    for index, row in recover_df.iterrows():
        key = str(row['Country/Region'])+'_'+str(row['Province/State'])
        try:
            ts[key].append([i for i in row.values][4:]) # e.g. China_Beijing, 3d array [confirmed, death, recovered]
        except:
            continue
    
    return ts

if __name__ == '__main__':
    ts = get_time_series_data()
    with open('src/feature_engineering/time_series.pkl', 'wb') as f:
        pickle.dump(ts, f)

    # check
    ts = None
    with open('src/feature_engineering/time_series.pkl', 'rb') as f:
        ts = pickle.load(f)
    
    print(len(ts))
    droped = list()
    for key in ts:
        if len(ts[key]) < 3:
            droped.append(key)
    for key in droped:
        print(key)
        del ts[key]
    
    print(len(ts))