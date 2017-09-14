import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression



def mean_prediction(y_train, X_test):
    '''
    Parameters:
    y_train = series or array like, true values for training data
    X_test = Dataframe for testing dataset

    Returns:
    results = Dataframe of testing salesID and predictions
    '''
    X_test['SalePrice'] = y_train.mean()
    results = X_test[['SalesID','SalePrice']]
    return results

def print_predictions(results, model_name):
    file_name = 'data/'+model_name+'_predictions.csv'
    results.to_csv(file_name, sep='\t', index=False)

def linear_model(X_train, y_train, X_test):
    '''
    Parameters:
    y_train = series or array like, true values for training data
    X_test = Dataframe for testing dataset

    Returns:
    results = Dataframe of testing salesID and predictions
    '''
    linear = LinearRegression()
    fit = linear.fit(X_train, y_train)
    y_pred = fit.predict(X_test)
    return y_pred

# clean up the data for Train and Test
def data_cleanup(X):
    X['cnst'] = 1
    X = X[['cnst','YearMade']].values  #model 3
    #put mean of year>1000 as value

    return X


if __name__=='__main__':
    print 'Reading in data...'
    df_train = pd.read_csv('data/Train.csv', low_memory=False)
    df_test = pd.read_csv('data/test.csv',low_memory=False)



    # Simple model 1 - mean
    df_train_ = pd.DataFrame.copy(df_train)
    df_test_ = pd.DataFrame.copy(df_test)
    y_train = df_train_.pop('SalePrice')
    X_train = df_train_
    X_test = df_test_
    print 'model1...'
    model1_predictions = mean_prediction(y_train, df_test)
    print 'savings results...'
    print_predictions(model1_predictions, 'model1')


    # Simple model 2 - linear with year
    # Preparing training data
    df_train_ = pd.DataFrame.copy(df_train)
    df_test_ = pd.DataFrame.copy(df_test)
    df_train_reduced = df_train_[df_train_['YearMade']>1000]
    df_train_reduced['cnst'] = 1
    y_train = df_train_reduced['SalePrice'].values
    X_train = df_train_reduced[['cnst','YearMade']].values
    # preparing test data
    df_test['cnst'] = 1
    X_test = df_test[['cnst','YearMade']].values
    # Predicting sales price
    print 'model2...'
    y_pred = linear_model(X_train, y_train, X_test)
    df_test['SalePrice'] = y_pred
    results = df_test[['SalesID','SalePrice']]
    print 'saving reults...'
    print_predictions(results, 'model2')


    # Simple model 3 - linear with year and Hydraulics
    # Preparing training data
    df_train_ = pd.DataFrame.copy(df_train)
    df_test_ = pd.DataFrame.copy(df_test)
    df_train_reduced = df_train_[df_train_['YearMade']>1000]
    #df_train_reduced['cnst'] = 1
    y_train = df_train_reduced['SalePrice'].values
    #X_train = df_train_reduced[['cnst','YearMade']].values

    # preparing test data
    #df_test['cnst'] = 1
    #X_test = df_test[['cnst','YearMade']].values

    X_train = data_cleanup(X_train)
    X_test = data_cleanup(X_test)
    # Predicting sales price
    print 'model3...'
    y_pred = linear_model(X_train, y_train, X_test)
    df_test['SalePrice'] = y_pred
    results = df_test[['SalesID','SalePrice']]
    print 'saving reults...'
    print_predictions(results, 'model2')
