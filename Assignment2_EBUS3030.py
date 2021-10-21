#!/usr/bin/env python
# coding: utf-8

# In[50]:


from datetime import datetime, timedelta, date
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from __future__ import division


# In[51]:


import warnings
warnings.filterwarnings("ignore")


# In[52]:


from chart_studio import plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go


# In[53]:


import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split


# In[54]:


pyoff.init_notebook_mode()


# In[55]:


df_sales = pd.read_csv('output.csv')


# In[56]:


df_sales.dtypes


# In[57]:


df_sales.head(10)


# In[58]:


df_sales = df_sales[['Sale_Date', 'Row_Total']]
df_sales['Sale_Date'] = pd.to_datetime(df_sales['Sale_Date'])
df_sales.head(10)


# In[59]:


#df_sales = df_sales[['Sale_Date', 'Row_Total']]
df_sales = df_sales.set_index('Sale_Date')
df_sales = df_sales.Row_Total.resample('W').sum()
df_sales = pd.DataFrame(df_sales).reset_index()
df_sales.tail(10)


# In[60]:


#tried to add
#start = datetime(2021, 1, 10)
#future_dates = pd.DataFrame(pd.date_range(start, periods = 10, freq="W"), columns = ['Sale_Date'])
#df_sales = df_sales.append(future_dates).reset_index(drop=True)
#df_sales.tail(20)


# In[61]:


#represent month in date field as its first day
# df_sales['Sale_Date'] = df_sales['Sale_Date'].dt.year.astype('str') + '-W' + df_sales['Sale_Date'].dt.month.astype('str') + '-01'
#df_sales['Sale_Date'] = pd.to_datetime(df_sales['Sale_Date'])
#groupby date and sum the sales
#df_sales = df_sales.groupby('Sale_Date').Row_Total.sum().reset_index()
#df_sales.head()


# In[62]:


#plot monthly sales
plot_data = [
    go.Scatter(
        x=df_sales['Sale_Date'],
        y=df_sales['Row_Total'],
    )
]

plot_layout = go.Layout(
        title='Weekly Sales'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[63]:


#create a new dataframe to model the difference
df_diff = df_sales.copy()


# In[64]:


#add previous sales to the next row
df_diff['prev_sales'] = df_diff['Row_Total'].shift(1)


# In[65]:


df_diff.tail()


# In[66]:


#drop the null values and calculate the difference
df_diff = df_diff.dropna()


# In[67]:


df_diff['diff'] = (df_diff['Row_Total'] - df_diff['prev_sales'])


# In[68]:


df_diff.tail(10)


# In[69]:


#plot sales diff
plot_data = [
    go.Scatter(
        x=df_diff['Sale_Date'],
        y=df_diff['diff'],
    )
]

plot_layout = go.Layout(
        title='Weekly Sales Diff'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[70]:


#create new dataframe from transformation from time series to supervised
df_supervised = df_diff.drop(['prev_sales'],axis=1)


# In[71]:


#adding lags
for inc in range(1,13):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(inc)


# In[72]:


df_supervised.head(10)


# In[73]:


df_supervised.tail(6)


# In[74]:


#drop null values
df_supervised = df_supervised.dropna().reset_index(drop=True)


# In[75]:


# Import statsmodels.formula.api
import statsmodels.formula.api as smf 

# Define the regression formula
model = smf.ols(formula='diff ~ lag_1', data=df_supervised)

# Fit the regression
model_fit = model.fit()

# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)


# In[76]:


# Import statsmodels.formula.api
import statsmodels.formula.api as smf 

# Define the regression formula
model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6', data=df_supervised)

# Fit the regression
model_fit = model.fit()

# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)


# In[77]:


# Import statsmodels.formula.api
import statsmodels.formula.api as smf 

# Define the regression formula
model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + lag_10 + lag_11 + lag_12', data=df_supervised)

# Fit the regression
model_fit = model.fit()

# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj
print(regression_adj_rsq)


# In[78]:


#import MinMaxScaler and create a new dataframe for LSTM model
from sklearn.preprocessing import MinMaxScaler
df_model = df_supervised.drop(['Row_Total','Sale_Date'],axis=1)


# In[79]:


#split train and test set 25 train, 15 test
train_set, test_set = df_model[0:-15].values, df_model[-15:].values


# In[80]:


train_set.shape


# In[81]:


df_model.info()


# In[82]:


#apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)

# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)


# In[83]:


X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])


# In[84]:


X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print(X_train.shape[1])


# In[85]:


model = Sequential()
model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, shuffle=False)


# In[86]:


y_pred = model.predict(X_test,batch_size=1)


# In[87]:


y_pred


# In[88]:


y_test


# In[89]:


#reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])


# In[90]:


#rebuild test set for inverse transform
pred_test_set = []
for index in range(0,len(y_pred)):
    print(np.concatenate([y_pred[index],X_test[index]],axis=1))
    pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))


# In[91]:


pred_test_set[0]


# In[92]:


#reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])


# In[93]:


#inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)


# In[94]:


#create dataframe that shows the predicted sales
result_list = []
sales_dates = list(df_sales[-16:].Sale_Date)
act_sales = list(df_sales[-16:].Row_Total)
for index in range(0,len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
    result_dict['Sale_Date'] = sales_dates[index+1]
    result_list.append(result_dict)
df_result = pd.DataFrame(result_list)


# In[95]:


df_result.tail(6)
pred_test_set_inverted


# In[96]:


df_sales.head()


# In[98]:


#merge with actual sales dataframe
df_sales_pred = pd.merge(df_sales,df_result,on='Sale_Date',how='left')


# In[99]:


df_sales_pred


# In[100]:


#plot actual and predicted
plot_data = [
    go.Scatter(
        x=df_sales_pred['Sale_Date'],
        y=df_sales_pred['Row_Total'],
        name='actual'
    ),
        go.Scatter(
        x=df_sales_pred['Sale_Date'],
        y=df_sales_pred['pred_value'],
        name='predicted'
    )
    
]

plot_layout = go.Layout(
        title='Sales Prediction'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[178]:


import pandas as pd
from prophet import Prophet
df_sales = pd.read_csv('output.csv')
# filters for one location
df_sales = df_sales[df_sales['Office_Location'] == 'Broken Hill']
# 
df_sales = df_sales[['Sale_Date', 'Row_Total']]
df_sales['Sale_Date'] = pd.to_datetime(df_sales['Sale_Date'])
# sets index for resample
df_sales = df_sales.set_index('Sale_Date')
# resample to days
df_sales = df_sales.Row_Total.resample('D').sum()
df_sales = pd.DataFrame(df_sales).reset_index()

df_prof = df_sales.rename(columns={"Sale_Date": "ds", "Row_Total": "y"})
df_prof = df_prof[df_prof['ds']!='2021-01-03']


# In[179]:


m = Prophet()
m.fit(df_prof)


# In[180]:


future = m.make_future_dataframe(freq = "D", periods=88)
future


# In[181]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(13)


# In[182]:


fig1 = m.plot(forecast)


# In[183]:


fig2 = m.plot_components(forecast)


# In[176]:


from prophet.plot import plot_plotly, plot_components_plotly

plot_plotly(m, forecast)


# In[177]:


plot_components_plotly(m, forecast)

