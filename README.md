# COVID_19_ML 
- Machine learning Research Project, here is the link to report, https://github.com/yunfeiluo/yunfeiluo.github.io/blob/master/articles/projects/Covid-19%20Cases%20Forecasting,%20Machine%20Learning%20Project,%202020,%20Project%20Report.pdf  
- Cases data from JHU, more details about it see another repository, https://github.com/CSSEGISandData/COVID-19  

# Main tasks 
- Find useful features, e.g. GDP of region the country belongs to, population density, life expectancy, etc. for cases Forcasting 
- Build reasonable models for forcasting 

# Analysis log 
- split data into samples and labels in a ways that using the previous days (depend on the window size, 14 for current experiment) as features, and the upcoming day as label 
- Tried SVM with RBF kernel: perform not very well 
- Linear Ridge regression: Perform very well in the training set, but poor in the testing set 
- Linear Ridge regression as a pre-predict, then train shallow MLP with linear layer (concat with covariates, such as GDP, population density, life_expectancy, etc.): Perform not bad in training set, not bad in testing set
- Train shallow MLP Autoencoder with linear layer first, then use it as a tool for feature dimension reduction, then train the shallow MLP with linear layer (concat with covariates): perform about the same as the previous strategy

# planned works  
- For the autoencoder part, want to try PCA, or using LSTM or convolutional layer 
- For the forcaster part, want to try LSTM or GRU layer 
- Train only with data in China and South Korea, for forcasting the inflection point 
- build "personalized (country-lized)" model 
