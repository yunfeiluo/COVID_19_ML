import pickle

if __name__ == '__main__':
    samples = dict() # map: country -> covariates (features), time series

    # get covariates
    features = None
    with open('src/feature_engineering/features.pkl', 'rb') as f:
        features = pickle.load(f)
    
    # get time series
    time_series = None
    with open('src/feature_engineering/time_series.pkl', 'rb') as f:
        time_series = pickle.load(f)
    
    # form samples
    for country in features:
        samples[country] = {'covariates': list(), 'time_series': list()}
        samples[country]['time_series'] = time_series[country]
        for feature in features[country]:
            samples[country]['covariates'].append(features[country][feature])    
    
    # write to pickle file
    with open('src/feature_engineering/samples.pkl', 'wb') as f:
        pickle.dump(samples, f)
    
    # check
    samples = None
    with open('src/feature_engineering/samples.pkl', 'rb') as f:
        samples = pickle.load(f)

    print(samples['China']['time_series'])