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
        confirmed = [i for i in row.values][4:]
        diff_confirmed = [confirmed[0]] + [confirmed[i] - confirmed[i-1] for i in range(1, len(confirmed))]
        for i in range(len(diff_confirmed)):
            if diff_confirmed[i] < 0:
                diff_confirmed[i] = 0
        ts[key] = [diff_confirmed] # e.g. China_Beijing, 3d array [confirmed, death, recovered]
    
    # extract death time series
    death_df = pd.read_csv('data/jhu/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

    for index, row in death_df.iterrows():
        key = str(row['Country/Region'])+'_'+str(row['Province/State'])
        death = [i for i in row.values][4:]
        diff_death = [death[0]] + [death[i] - death[i-1] for i in range(1, len(death))]
        for i in range(len(diff_death)):
            if diff_death[i] < 0:
                diff_death[i] = 0
        try:
            ts[key].append(diff_death) # e.g. China_Beijing, 3d array [confirmed, death, recovered]
        except:
            continue
    
    # extract recover time series
    recover_df = pd.read_csv('data/jhu/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

    for index, row in recover_df.iterrows():
        key = str(row['Country/Region'])+'_'+str(row['Province/State'])
        recover = [i for i in row.values][4:]
        diff_recover = [recover[0]] + [recover[i] - recover[i-1] for i in range(1, len(recover))]
        for i in range(len(recover)):
            if diff_recover[i] < 0:
                diff_recover[i] = 0
        try:
            ts[key].append(diff_recover) # e.g. China_Beijing, 3d array [confirmed, death, recovered]
        except:
            continue
    
    return ts

if __name__ == '__main__':
    ts = get_time_series_data()

    droped = list()
    for key in ts:
        if len(ts[key]) < 3:
            droped.append(key)
    for key in droped:
        print(key)
        del ts[key]
    
    print('Total droped', len(droped))

    # write to file
    with open('src/feature_engineering/time_series.pkl', 'wb') as f:
        pickle.dump(ts, f)

    # check
    ts = None
    with open('src/feature_engineering/time_series.pkl', 'rb') as f:
        ts = pickle.load(f)
    print('Num of regions', len(ts))