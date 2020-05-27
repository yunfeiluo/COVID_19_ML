import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

'''
run 
    src.feature_engineering.form_time_series
to fetch the up-to-date data

Then run this file to generate the samples, labels, and covars. 
More details see comments for variable: train_data
'''
## read and split data #######################################################################################
data = None
with open('src/feature_engineering/time_series.pkl', 'rb') as f:
    data = pickle.load(f)

# # wash data
# covariates = None
# with open('src/feature_engineering/features.pkl', 'rb') as f:
#     covariates = pickle.load(f)
# countries = [i for i in covariates]
# droped = list()
# for region in data:
#     country = region.split('_')[0]
#     if country not in countries:
#         droped.append(region)
# for region in droped:
#     del data[region]

# window setting    
look_back = 14
train_len = -1

train_data = dict()
'''
@var train_data
map: region -> {
                    samples: [
                                        [[confirmed, death, recovered]]
                                    ], 
                    labels: [
                                    [confirmed, death, recovered]
                                ],
                    cavars: [covars],
                    scaler: MinMaxScaler obj
                }
'''

regions = list()

for region in data:
    regions.append(region)
    #train_data[region] = {"samples": list(), "labels": list(), "covars": covariates[region.split('_')[0]], "scaler": MinMaxScaler()}
    train_data[region] = {"samples": list(), "labels": list(), "scaler": MinMaxScaler()}
    data[region] = np.array(data[region]).T
    train_data[region]["scaler"] = train_data[region]["scaler"].fit(data[region])
    data[region] = train_data[region]["scaler"].transform(data[region]).tolist()

    train_len = len(data[region]) - 1 - look_back
    for j in range(train_len):
        train_data[region]["samples"].append(data[region][j:j+look_back])
        train_data[region]["labels"].append(data[region][j+look_back])


    # for i in range(3):
    #     train_x = data[region][i]
        
    #     # append train data
    #     train_len = len(train_x) - 1 - look_back
    #     for j in range(train_len):
    #         if i == 0:
    #             train_data[region]["samples"].append([train_x[j:j+look_back], list(), list()])
    #             train_data[region]["labels"].append([train_x[j+look_back], -1, -1])
    #         else:
    #             train_data[region]["samples"][j][i] = train_x[j:j+look_back]
    #             train_data[region]["labels"][j][i] = train_x[j+look_back]
        
    #     # reshape
    #     for k in range(len(train_data[region]["samples"])):
    #         train_data[region]["samples"][k] = np.array(train_data[region]["samples"][k]).T.tolist()

filename = 'data/train_data/newly_generated_data.pkl'
with open(filename, 'wb') as f:
    print("write to file", filename)
    pickle.dump(train_data, f)

# check
for key in train_data:
    print(key)
    train_data[key]["samples"] = np.array(train_data[key]["samples"])
    train_data[key]["labels"] = np.array(train_data[key]["labels"])
    #train_data[key]["covars"] = np.array(train_data[key]["covars"])

    print("train shape", train_data[key]["samples"].shape)
    print("label shape", train_data[key]["labels"].shape)
    #print("covar shape", train_data[key]["covars"].shape)
    print(' ')