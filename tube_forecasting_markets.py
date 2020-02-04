# -*- coding: utf-8 -*-
"""
Created on 21-01-2020

@author: Syamanthaka B
"""
## All required imports 
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Hides plot from displaying, only backend save happes
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
from pmdarima.arima import auto_arima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import datetime
from statsmodels.tsa.stattools import adfuller

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore") # Hides warnings for user
    
## Function y_exog
# Splits the data as Consumption - which is the Y, and Failure Rate which is the external regressor.
# Input is the original historic data frame for a given market
# Output is two data frames one each for Y and the external regressor
def y_exog(df):
    consumption_df = df[['Month-Year', 'Consumption']]
    consumption_df = consumption_df.set_index('Month-Year') # Set the month year as the index column
    
    fr_df = df[['Month-Year', 'Failure_rate']]
    fr_df = fr_df.set_index('Month-Year')
    
    return(consumption_df, fr_df)

## Function train_test
# Splits given data frame into training and test sets by a 70-30 rule.
# Percentage of split can be easily customized - notes inline
# Input is a data frame that needs to be split
# Output is train dataset, test data set and number of rows each of how much is 70 and 30 percents
def train_test(df):
    total_rows = len(df)
    # To customize the percentage, simply edit the number 0.7 below
    # For example, to change to 60-40, simply make below line - round(0.6 * total_rows)
    # If changed, recommened to change the variable names for ease of understanding
    seventy_pct = round(0.7 * total_rows) 
    thirty_pct = total_rows - seventy_pct
    
    train = df.iloc[:seventy_pct]
    test = df.iloc[seventy_pct:]
    
    return (train, test, seventy_pct, thirty_pct)

## Function auto_arima_model
# Generates an optimized stepwise model 
# Also checks data from stationarity using the ADF test and incorporates into model
# Input data frame, the name of column which is y and the external regressor data frame
# Returns a model object
def auto_arima_model(df, y_col, exog):
    result = adfuller(df[y_col]) # Taking the y value
    p_value = result[1]
    
    if p_value <= 0.05:
        stationary_bin = True
    else:
        stationary_bin = False
   
    stepwise_model = auto_arima(df, 
                           seasonal=False,
                           start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0,
                           d=1, D=1,
                           trace=False,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True,
                           exogenous=exog, stationary=True)
    return(stepwise_model)
    
## Function fit_predict
# Fits the model and predicts on test set
# Input is model object, train and test data sets for both y and external regressor, prediod of prediction and market name
# Output is the forecast along with confidence interval of 95%
# Also saves graphs of prediction under a folder called Results.
def fit_predict(model, con_train, con_test, fr_train, fr_test, predict_period, Market_name):
    model.fit(con_train, exogenous=fr_train, alpha = 0.5)
    fc, confint = model.predict(n_periods=predict_period, exogenous=fr_test, return_conf_int=True)
    
    fc_ind = pd.date_range(con_train.index[con_train.shape[0]-1], 
                           periods=predict_period, freq="M")
    # Forecast series
    fc_series = pd.Series(fc, index=fc_ind)
    # Upper and lower confidence bounds
    lower_series = pd.Series(confint[:, 0], index=fc_ind)
    upper_series = pd.Series(confint[:, 1], index=fc_ind)
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(con_train, color="blue")
    plt.plot(fc_series, color="red")
   
    plt.xlabel("date")
   
    plt.fill_between(lower_series.index, 
                     lower_series, 
                     upper_series, 
                     color="k", 
                     alpha=0.25)
    # If training/test data set is provided does the if part. Else predicts for entire data and gives upcoming forecast
    if isinstance(con_test, pd.DataFrame):
        plt.plot(con_test, color="green")
        plt.title("Training data set and prediction on test data set")
        plt.legend(("historic", "forecast", "past", "95% confidence interval"),  
               loc="upper left")
        plot_name = 'Results/train_test_preds_' + Market_name + '.png'
    else:
        plt.title("Historic data and prediction for next 18 months")
        plt.legend(("historic", "forecast", "95% confidence interval"))
        plot_name = 'Results/Predictions_' + Market_name + '.png'
    
    plt.savefig(plot_name)
    plt.clf() # Clear plot for freeing memory
    
    return(fc, confint)
    
## Function get_next_month
# Simple function that generates next month of a given date
def get_next_month(date):
    month = (date.month % 12) + 1
    year = date.year + (date.month + 1 > 12)
    return datetime.datetime(year, month, 1)

## Function create_pred_df
# Creates a df for predicted future dates using get_next_month function
def create_pred_df(pred_col_name, first_date, n_period):
    pred_df = pd.DataFrame(columns=[pred_col_name, 'Month-Year'])
    future_dates = []
    for i in range(1, n_period+1):
        new_month = get_next_month(first_date)
        first_date = new_month
        future_dates.append(new_month)
        
    pred_df['Month-Year'] = future_dates
 
    return(pred_df)

## Function exog_predict
# Function that makes a future prediction of external regressor to be used in the main prediction
# Uses holt winter model.
# Also saves graph of external regressor predictions.
def exog_predict(fr_df, Market_name):
    fr_df_new = create_pred_df('Failure_rate', fr_df.index[-1], 18)
    fr_df_new = fr_df_new.set_index('Month-Year')
   
    try:
        hw_model = ExponentialSmoothing(fr_df, seasonal="multiplicative", seasonal_periods=12, trend="multiplicative").fit(use_boxcox=False, damping_slope=1)
    except ValueError:
        hw_model = ExponentialSmoothing(fr_df, seasonal_periods=12, seasonal="additive").fit(use_boxcox=False)
    pred = hw_model.predict(start=fr_df_new.index[0], end=fr_df_new.index[-1])
   
    plt.figure(figsize=(8, 6))
    plt.plot(fr_df.index, fr_df, label='Exisiting Failure Rate')
    plt.plot(pred.index, pred, label='Holt-Winters predicted Failure Rate')
    plt.legend(loc='best')
    plt.title("HW preds for FR")
    plt_name = 'Results/hw_future_fr_preds_' + Market_name + '.png'
    plt.savefig(plt_name)
    plt.clf()
    
    pred_fr_df = pred.to_frame(name="Failure_rate")
    pred_fr_df.index.name = 'Month-Year'
    
    return (pred_fr_df)

## Function main
# This is the main function which calls all other functions as required and generates forecasts from start to end. 
def main(data, Markets):
    for each in Markets:
        ## Split by each market
        market_df = data.loc[data['Market'] == each, ['Month-Year', 'Consumption', 'Failure_rate']]
        
        ## Split the data set into consumption data and failure rate (Which is the external regressor)
        consumption_df, fr_df = y_exog(market_df)
        
        ## Train test split for consumption data and failure rate data at the rate of 70 and 30% respectively
        consumption_train, consumption_test, c70, c30 = train_test(consumption_df)
        
        fr_train, fr_test, f70, f30 = train_test(fr_df)
        
        ## Auto arima fit on training data with predict on test for verification. - Saves this image in folder
        stepwise_train = auto_arima_model(consumption_train, 'Consumption', fr_train)
        initial_fc, initial_ci = fit_predict(stepwise_train, consumption_train, consumption_test, fr_train, fr_test, c30, each)
        
        
        ## Holt Winters for future failure rate prediction
        pred_fr_df = exog_predict(fr_df, each)
        
        ## Auto arima on whole consumption data
        stepwise_full = auto_arima_model(consumption_df, 'Consumption', fr_df)

        forecast, conf_int = fit_predict(stepwise_full, consumption_df, '' , fr_df, pred_fr_df, 18, each)

        pred_consumption = create_pred_df('Consumption', consumption_df.index[-1], 18)
        
        pred_consumption['Consumption'] = forecast
        pred_consumption['Lower'] = conf_int[:,0]
        pred_consumption['Upper'] = conf_int[:,1]
        
        pred_consumption = pred_consumption[['Month-Year', 'Consumption', 'Lower', 'Upper']]
        pred_filename = "Results/prediction_" + each + ".csv"
        pred_consumption.to_csv(pred_filename, index=False)
         
    print("Completed")
     
## Define the markets and read the historic data file
Markets = ['LAT','APA','DAC','ISC','BNL','NAM','GRC','CEE','NOR','AFI','FRA','IIG','UKI','JPN','IBE','RCA','MET']
data = pd.read_excel('Data/historic_tube_data.xlsx') 

main(data, Markets)


