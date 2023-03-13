import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score, KFold


def my_regression_results(model):
    
    score_test = model.score(X_test,y_test)
    print('Model r-squared score from test data: {:0.4f}'.format(score_test))

   
    y_pred = model.predict(X_test)
    y_pred2 = model.predict(X)
    %matplotlib inline
    import matplotlib.pyplot as plt
    
    plt.plot(y_test,y_pred,'k.')
    plt.xlabel('Test Values')
    plt.ylabel('Predicted Values');

   
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    print('Mean squared error on test data: {:0.7f}'.format(mse))
    print('Root mean squared error on test data: {:0.7f}'.format(rmse))
    


df = pd.read_csv('table.csv', sep=",")
target_column = ['target'] 
predictors = list(set(list(df.columns))-set(target_column))

print(df[predictors])
df.describe()
X = df[predictors].values
y = df[target_column].values
#Split data into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
df.head()

# create a baseline model object declaring defualt values for hyperparameters
model_lr = LinearRegression()

model_lr.fit(X_train,y_train) # this could be inside the function below too    
my_regression_results(model_lr)

rf_model = RandomForestRegressor(random_state=0)
#Fit model and compare results 
rf_model.fit(X_train,y_train)
my_regression_results(rf_model)

# define the parameter ranges to optimize
params = {
    "n_estimators": randint(50, 500),
    "max_features": [0.1, 0.5, 1.0, 2.0, 3.0, 4.0],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 20),
    "bootstrap": [True, False]
}

random_search = RandomizedSearchCV(
    rf_model,
    param_distributions=params,
    random_state=123,
    n_iter=25,
    cv=5,
    verbose=1,
    n_jobs=1,
    return_train_score=True)

random_search.fit(X_train, y_train)

random_search.best_params_

scores = cross_val_score(rf_model,X=X_train,y=y_train,cv=KFold(n_splits=5))
print (scores)

print(random_search.best_params_)
my_regression_results(random_search)
