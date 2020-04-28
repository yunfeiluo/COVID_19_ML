import pandas as pd
import pickle

## helper functions #####################################################
def get_countries(df, col_name):
    countries = set([country for country in df[col_name]])
    return [country for country in countries]

def get_feature(df, col_name, countries, feature):
    feature_list = list()
    droped = list() # store the missing entry
    for country in countries:
        row = df.loc[df[col_name] == country]
        try:
            feature_list.append(row[feature].values[0])
        except:
            feature_list.append(-1)
            droped.append(country)
    return feature_list, droped

#########################################################################

# main function
def run():
    # open train file
    train = pd.read_csv('data/train.csv')
    # get country names
    features = dict()
    droped_countries = list()
    countries = sorted(get_countries(train, 'Country_Region'))

    # get GDP
    gdp_population = pd.read_csv('data/features/gdp_population.csv')
    total_GDP, droped = get_feature(gdp_population, 'country', countries, 'imfGDP')
    droped_countries += [i for i in droped]
    GDP_percapita, droped = get_feature(gdp_population, 'country', countries, 'gdpPerCapita')
    droped_countries += [i for i in droped]

    # get population density
    population, droped = get_feature(gdp_population, 'country', countries, 'pop')
    droped_countries += [i for i in droped]
    # update features
    for i in range(len(countries)):
        features[countries[i]] = dict()
        features[countries[i]]['total_GDP'] = total_GDP[i]
        features[countries[i]]['GDP_percapita'] = GDP_percapita[i] # normalized by /1000
        features[countries[i]]['pop_density'] = population[i]
        features[countries[i]]['life_expectancy'] = -1
    
    # land area data from another dataset
    land_area = pd.read_csv('data/features/land_area.csv', encoding="ISO-8859-1")
    land_area, droped = get_feature(land_area, 'Country Name', countries, '2018')
    droped_countries += [i for i in droped]
    
    for i in range(len(countries)):
        features[countries[i]]['pop_density'] /= land_area[i]
        if len(str(features[countries[i]]['pop_density']).split('.')) <= 1: # drop if nan
            droped_countries.append(countries[i])
    
    # life expectancy
    life_expectancy = pd.read_csv('data/features/life_expectancy.csv', encoding="ISO-8859-1")
    life_expectancy, droped = get_feature(life_expectancy, 'Country Name', countries, '2018')
    droped_countries += [i for i in droped]

    for i in range(len(countries)):
        features[countries[i]]['life_expectancy'] = life_expectancy[i]
        if len(str(features[countries[i]]['life_expectancy']).split('.')) <= 1: # drop if nan
            droped_countries.append(countries[i])
    
    droped_countries = [i for i in set(droped_countries)]
    for country in droped_countries:
        del features[country]
        countries.remove(country)
    
    print('Num of droped contries', len(droped_countries))
    # for country in droped_countries:
    #     print(country)

    # # generate features write to content
    content = '国家 | 总GDP | 人均GDP | 人口密度 | 人均寿命\n'
    for country in countries:
        content += '{} {} {} {} {}\n'.format(country, features[country]['total_GDP'], features[country]['GDP_percapita'], features[country]['pop_density'], features[country]['life_expectancy'])

    # write to file
    with open('src/feature_engineering/features.txt', 'w+') as f:
        print('write to file... ' + 'src/feature_engineering/features.txt')
        f.write(content)

    return features

# call main function
if __name__ == '__main__':
    features = run()

    # write to pickle file
    with open('src/feature_engineering/features.pkl', 'wb') as f:
        pickle.dump(features, f)
