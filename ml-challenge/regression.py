# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm

# %%
# load data
df_customer = pd.read_csv('../data/Case Study - Customer.csv', sep=';')
df_product = pd.read_csv('../data/Case Study - Product.csv', sep=';')
df_store = pd.read_csv('../data/Case Study - Store.csv', sep=';')
df_transaction = pd.read_csv('../data/Case Study - Transaction.csv', sep=';')

# %%
# convert Date to datetime
df_transaction['Date'] = pd.to_datetime(df_transaction['Date'], format='%d/%m/%Y')
# fill missing values
df_customer.isna().sum()
df_customer.fillna(method='ffill', inplace=True)

# %%
# merge df
merged_df = pd.merge(df_transaction, df_product, on='ProductID', how='left')
merged_df = pd.merge(merged_df, df_store, on='StoreID', how='left')
merged_df = pd.merge(merged_df, df_customer, on='CustomerID', how='left')
merged_df.head()

# %%
merged_df.info()

# %%
# create df for regression
reg_df = df_transaction.groupby('Date')['Qty'].sum().reset_index()
reg_df['Date'] = pd.to_datetime(reg_df['Date'], format='%d/%m/%Y')
reg_df.sort_values(by='Date', inplace=True)
reg_df.set_index('Date', inplace=True)


# %%
# plot qty sales in a year
reg_df.plot(figsize=(12,8), title='Daily Sales', xlabel='Date', ylabel='Total Qty', legend=False)

# %%
# split into training and testing data by 80/20
train = reg_df[:int(0.8*(len(reg_df)))]
test = reg_df[int(0.8*(len(reg_df))):]

# %%
# grid search for p, d, and q
auto_arima_model = pm.auto_arima(
    train['Qty'], 
    seasonal=False, 
    stepwise=False, 
    suppress_warnings=True, 
    trace = True
)
auto_arima_model.summary()

# %%
# import sarimax
p, d, q = auto_arima_model.order
model = SARIMAX(train['Qty'].values, order=(p,d,q))
model_fit = model.fit(disp=False)

# %%
# count rsme
from sklearn.metrics import mean_squared_error
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
rmse = mean_squared_error(test, predictions, squared=False)
rmse

# %%
# forecasting for next 90 days
period = 90
forecast = model_fit.forecast(steps=period)
index = pd.date_range(start='01-01-2023', periods=period)
df_forecast = pd.DataFrame(forecast, index=index, columns=['Qty'])

# %%
plt.figure(figsize=(12,8))
plt.title('Forecasting Sales')
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(df_forecast, label='Predicted')
plt.legend(loc='best')
plt.show()

# %%
# plot forecast
df_forecast.plot(figsize=(12,8), title='Forecasting Sales', xlabel='Date', ylabel='Total Qty', legend=False)

# %%
# forecast product for next 90 days
warnings.filterwarnings('ignore')

product_reg_df = merged_df[['Qty', 'Date', 'Product Name']]
new = product_reg_df.groupby("Product Name")

forecast_product_df = pd.DataFrame({'Date': pd.date_range(start='2023-01-01', periods=90)})

for product_name, group_data in new:
    target_var = group_data['Qty']
    model = SARIMAX(target_var.values, order=(p,d,q))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(90)
    forecast_product_df[product_name] = forecast

forecast_product_df.set_index('Date', inplace=True)
forecast_product_df.head()

# %%
# plot forecast for products
plt.figure(figsize=(12,8))
for i in forecast_product_df.columns:
    plt.plot(forecast_product_df[i], label=i)
plt.legend(loc=6, bbox_to_anchor=(1,.82))
plt.title('Forecasting Product')
plt.xlabel('Date')
plt.ylabel('Total Qty')
plt.show()


