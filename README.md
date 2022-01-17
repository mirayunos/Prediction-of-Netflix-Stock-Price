# Prediction-of-Netflix-Stock-Price
Predicting Netflix Stock Price with Linear Regression Algorithm. Special thanks to Dr. Mohammed Al-Obaydee for the guidance.

## Business Understanding

Case Study: To design a stock prediction model for the closing price of Netflix.

* Goal: To predict the closing price of Netflix.
* Objective: To build a predictive model using Linear Regression algorithm.

*Closing price: Last price to buy before the closing of business hour or After-Hour Trading (AHT).*


## Analytic Approach

Linear Regression algorithm is used to predict the closing price of Netflix stock.

*Simple Linear Regression algorithm is a supervised machine learning algorithm to model linear relationship between to variables.*

## Data Requirements

* Opening Price
* Closing Price
* Low Price
* High Price
* Volume of Stock

## Data Collection

The dataset is sourced from Kaggle.com. Dataset is downloaded and uploaded back to Jupyter Notebook. This dataset is based on Yahoo Finance: Netflix historical price 12/16/2015 until 12/16/2019 daily price and volume. There are 7 columns; Date, open, high, low, close, volume, adj close (2001, 7) each of stock.

## Data Understanding and Preparation

* Importing Packages

```
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
```

* Reading the Dataset

```
df = pd.read_csv("NFLX.csv")
df.head()
```

```
df.tail()
```

```
df.shape
```

(1007 rows, 7 columns)

* Summary Statistics

```
df.describe()
```

```
df.describe(include="all")
```
*'all' : All columns of the input will be included in the output.*

* Finding Correlation

To find the correlation among the columns in the dataframe using ‘Pearson’ method.

```
df.corr(method="pearson")
```

```
corr = df.corr()
corr
```

* Exploratory Data Analysis (EDA)

Using heatmap for visualisation.

```
import seaborn as sb
sb.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns,
 cmap='RdBu_r', annot=True, linewidth=0.5)
 ```
 ![1](https://user-images.githubusercontent.com/93753467/149455943-b2d33f1f-29aa-4ebd-aa7f-568fdd2e7446.png)

 Preparing the dataset to work with, using only the required columns.
 
 ```
nflx_df=df[["Date","High","Open","Low","Close"]]
nflx_df.head(10)
```

Plotting the Data

```
plt.figure(figsize=(16,8))
plt.title("Netflix Stocks Closing Price History 2015-2019")
plt.plot(nflx_df["Date"],nflx_df["Close"])
plt.xlabel("Date",fontsize=18)
plt.ylabel("Close Price US($)",fontsize=18)
plt.style.use("fivethirtyeight")
plt.show()
```
![2](https://user-images.githubusercontent.com/93753467/149796035-5a898fb2-ab1f-4e1e-bcd2-99819ec6934e.png)

```
#Plot Open vs Close
nflx_df[['Open','Close']].head(20).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
```
![3](https://user-images.githubusercontent.com/93753467/149796363-3b871156-6b93-4c29-b00f-08d46fad5a73.png)

```
#Plot High vs Close
nflx_df[['High','Close']].head(20).plot(kind='bar',figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
```
![4](https://user-images.githubusercontent.com/93753467/149796410-9b6ba001-87e9-4c9c-9ebe-65656edf8c8a.png)

```
#Plot Low vs Close
nflx_df[["Low","Close"]].head(20).plot(kind="bar",figsize=(16,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
```
![5](https://user-images.githubusercontent.com/93753467/149796480-2fb237a6-ce86-49e1-afec-821375d30c39.png)


 

