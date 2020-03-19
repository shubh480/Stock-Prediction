#!/usr/bin/env python
# coding: utf-8

# In[284]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[285]:


data=pd.read_csv("C:\\Users\\Shubham\\nse_data.csv")
data.head()


# In[286]:


data.groupby("TIMESTAMP").count()


# In[287]:


data["TIMESTAMP"]=pd.to_datetime(data["TIMESTAMP"])
data["TIMESTAMP"]


# In[288]:


data["year"]=data["TIMESTAMP"].dt.year
data["month"]=data["TIMESTAMP"].dt.month
data["day"]=data["TIMESTAMP"].dt.day
data


# In[289]:


data.drop(["TIMESTAMP"], axis=1)


# In[290]:


data.groupby("year").count()


# In[291]:


df1=pd.read_csv("C:\\Users\\Shubham\\nse_2016.csv")
df1.head()


# In[292]:


df1.shape


# In[293]:


df1.drop(["TIMESTAMP"], axis=1, inplace=True)


# In[294]:


df1.drop(["year"], axis=1, inplace=True)


# In[295]:


df1.drop(["ISIN"], axis=1, inplace=True)


# In[296]:


df1.drop(["SERIES"], axis=1, inplace=True)


# In[297]:


df1["TOTALTRADES"].plot()
plt.show()


# In[298]:


#Calculate the moving average
#it means it takes price of one day and its past 59 days --> calculates the average
#specify in detail
df1["mean"]=df1["OPEN"].rolling(window=60, min_periods=0).mean()
print(df1.head())


# # task to do
# program having command line which sorts 1 company --> (we give company name)
# then it plots the below graph of specific company stocks and its mean.

# In[299]:


df2=df1.groupby(["SYMBOL"])


# In[300]:


for SYMBOL, SYMBOL_df1 in df2:
    print(SYMBOL)
    print(SYMBOL_df1)


# Selecting the Group User enters

# In[301]:


g= df2.get_group(input("Enter Stock name: "))
g.head()


# In[302]:


traning_set= g.iloc[:,1:2].values


# In[303]:


g.iloc[:,1:2]


# In[304]:


traning_set.shape


# In[305]:


from sklearn.preprocessing import MinMaxScaler


# In[306]:


sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled= sc.fit_transform(traning_set)


# In[307]:


training_set_scaled


# In[308]:


X_train=[]
y_train=[]
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)


# In[309]:


X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


# In[310]:


X_train


# In[311]:


X_train.shape


# Using Recurrent Neural Network

# In[312]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM


# In[313]:


regressor=Sequential()


# In[314]:


regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer="adam", loss="mean_squared_error")

regressor.fit(X_train, y_train, epochs=100, batch_size=217)
print(regressor.summary())


# In[315]:


dataset_test=pd.read_csv("C:\\Users\\Shubham\\nse_2017.csv")
dataset_test.head()


# In[316]:


dataset_test.drop(["TIMESTAMP"], axis=1, inplace=True)

dataset_test.drop(["year"], axis=1, inplace=True)

dataset_test.drop(["ISIN"], axis=1, inplace=True)

dataset_test.drop(["SERIES"], axis=1, inplace=True)


# In[317]:


dataset_test.head()


# In[318]:


df3=dataset_test.groupby(["SYMBOL"])


# In[319]:


g1= df3.get_group(input("Enter Stock name: "))
g1.head()


# In[320]:


g1.shape


# In[321]:


real_stock_price=g1.iloc[:,1:2].values


# In[322]:


g1.iloc[:,1:2]


# In[323]:


total_dataset=pd.concat((g["OPEN"], g1["OPEN"]), axis=0)
inputs=total_dataset[len(total_dataset) - len(g1) - 60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)


# In[324]:


X_test=[]
for i in range(60, 308):
    X_test.append(inputs[i-60:i, 0])
X_test=np.array(X_test)
X_test=np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

predicted_stock_price=regressor.predict(X_test)
predicted_stock_price=sc.inverse_transform(predicted_stock_price)


# In[325]:


X_test.shape


# In[326]:


predicted_stock_price


# In[327]:


plt.plot(real_stock_price,color="red",label="Real Stock Price")
plt.plot(predicted_stock_price,color="blue",label="Predicted Stock Price")
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




